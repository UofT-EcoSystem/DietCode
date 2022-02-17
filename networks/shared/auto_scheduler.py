import torch
import tvm
from tvm import relay
from tvm.auto_scheduler import ApplyHistoryBest, RecordToFile, TaskScheduler, \
                               TuningOptions, extract_tasks
from tvm.contrib import graph_executor

import abc
import logging
import os

from io import UnsupportedOperation

logger = logging.getLogger(__name__)

from ...shared import CUDATarget, CUDAContext
from ...shared.auto_scheduler import auto_sched_ntrials, get_log_filename, \
                                     local_runner
from ...shared.logger import AutoSchedTimer, AvgStdMedianLogger
from .timer import py_benchmark


# temporarily mutate the compile_engine and autotvm logger
compile_engine_logger = logging.getLogger('compile_engine')
compile_engine_logger.setLevel(logging.ERROR)
autotvm_logger = logging.getLogger('autotvm')
autotvm_logger.setLevel(logging.ERROR)


class AnsorEnv:
    __slots__ = ['apply_history_best']

    def __init__(self, sched_log_fname):
        os.environ['USE_ANSOR_SCHED_LOG_FORMAT'] = '1'
        self.apply_history_best = ApplyHistoryBest(sched_log_fname)
    
    def __enter__(self):
        self.apply_history_best.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.apply_history_best.__exit__(exc_type, exc_value, exc_tb)
        del os.environ['USE_ANSOR_SCHED_LOG_FORMAT']


class _AutoScheduler(abc.ABC):

    def _get_model_name(self, model_prefix, model_args):
        return model_prefix + '_' + 'x'.join(['{}'.format(arg) for arg in model_args])

    def extract_tasks(self, model_fixture):
        logger.info("Extracting search tasks ...")
        mod, params = relay.frontend.from_pytorch(
                          model_fixture.scripted_model,
                          [(model_fixture.input_name, model_fixture.input_data_np.shape)]
                      )
        tasks, task_weights = \
                extract_tasks(mod["main"], target=CUDATarget, params=params)

        logger.info("Extracted search tasks: {}"
                        .format([(t[0].desc, t[1]) for t in zip(tasks, task_weights)]))
        return mod, params, tasks, task_weights

    @abc.abstractmethod
    def train(self):
        pass

    def infer(self, fmodel_fixture, model_args, sched_log_fname,
              sched_results_log_fname="temp_workspace",
              append_log=False, latency_logger=None, model_fixture=None,
              mod=None, params=None):
        if latency_logger is None:
            latency_logger = AvgStdMedianLogger(sched_results_log_fname + '.csv', append_log)

        if model_fixture is None:
            model_fixture = fmodel_fixture(*model_args)

        if mod is None:
            mod, params, _, _ = self.extract_tasks(model_fixture)

        func = model_fixture.model.cuda()
        input_data = model_fixture.input_data_torch.cuda()

        stmt = "func(input_data)"
        cuda_sync = "torch.cuda.synchronize()"

        with torch.no_grad(), torch.jit.optimized_execution(True):
            avg_latency = py_benchmark(stmt, {**globals(), **locals()},
                                       setup=cuda_sync, finish=cuda_sync)
            latency_logger.write('PyTorch', model_args, avg_latency)
        model_fixture.model.cpu()

        logger.info("Compiling ...")
        with AnsorEnv(sched_log_fname), \
             tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=CUDATarget, params=params)

            module = graph_executor.GraphModule(lib["default"](CUDAContext))
            module.set_input(model_fixture.input_name,
                             tvm.nd.array(model_fixture.input_data_np))


        logger.info("Evaluating ...")
        time_evaluator = module.module.time_evaluator("run", CUDAContext, number=5,
                                                      repeat=3, min_repeat_ms=500)
        latency_logger.write('Ansor', model_args, time_evaluator().results)


class DietCodeEnv:
    __slots__ = ['dietcode_sched_log_dir']

    def __init__(self, dietcode_sched_log_dir):
        self.dietcode_sched_log_dir = dietcode_sched_log_dir

    def __enter__(self):
        os.environ['DIETCODE_SCHED_LOG_DIR'] = self.dietcode_sched_log_dir

    def __exit__(self, exc_type, exc_value, exc_tb):
        del os.environ['DIETCODE_SCHED_LOG_DIR']


class AnsorAutoScheduler(_AutoScheduler):

    def train(self, fmodel_fixture, model_args,
              sched_results_log_fname="temp_workspace", append_log=False):
        model_fixture = fmodel_fixture(*model_args)

        mod, params, tasks, task_weights = self.extract_tasks(model_fixture)

        model_name = self._get_model_name(model_fixture.name.lower(), model_args)
        logger.info("Tuning {} ...".format(model_name))
        sched_log_fname = get_log_filename('ansor', model_name)

        with AutoSchedTimer("{}_autosched_timer.csv".format('ansor'),
                            append_log, model_name):
            task_scheduler = TaskScheduler(tasks, task_weights, strategy="round-robin")
            tune_option = TuningOptions(
                              num_measure_trials=auto_sched_ntrials * len(tasks),
                              runner=local_runner,
                              measure_callbacks=[RecordToFile(sched_log_fname)]
                          )
            task_scheduler.tune(tune_option)

        return self.infer(fmodel_fixture=fmodel_fixture,
                          model_args=model_args,
                          model_fixture=model_fixture,
                          sched_log_fname=sched_log_fname,
                          sched_results_log_fname=sched_results_log_fname, 
                          append_log=append_log,
                          mod=mod, params=params)


class DietCodeAutoScheduler(_AutoScheduler):

    def train(self):
        logger.error("DietCode does not support training yet")
        raise UnsupportedOperation

    def infer(self, fmodel_fixture, model_args, ansor_sched_log_fname,
              dietcode_sched_log_dir, sched_results_log_fname="temp_workspace",
              append_log=False):
        model_fixture = fmodel_fixture(*model_args)
        mod, params, _, _ = self.extract_tasks(model_fixture)

        latency_logger = AvgStdMedianLogger(sched_results_log_fname + '.csv', append_log)

        super().infer(fmodel_fixture=fmodel_fixture, model_args=model_args,
                      sched_log_fname=ansor_sched_log_fname,
                      sched_results_log_fname=sched_results_log_fname,
                      append_log=append_log, latency_logger=latency_logger,
                      model_fixture=model_fixture, mod=mod, params=params)

        logger.info("Compiling ...")
        with DietCodeEnv(dietcode_sched_log_dir), \
             tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=CUDATarget, params=params)

            module = graph_executor.GraphModule(lib["default"](CUDAContext))
            module.set_input(model_fixture.input_name,
                             tvm.nd.array(model_fixture.input_data_np))
            module.run()

        logger.info("Evaluating ...")
        time_evaluator = module.module.time_evaluator("run", CUDAContext, number=5,
                                                      repeat=3, min_repeat_ms=500)
        latency_logger.write('DietCode', model_args, time_evaluator().results)
