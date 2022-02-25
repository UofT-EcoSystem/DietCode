
from tvm.auto_scheduler import ComputeDAG, SearchTask, \
                               SketchPolicy, TuningOptions, XGBModel, \
                               RecordToFile, load_best_record, load_records

import abc
import logging
import re

from math import prod

logger = logging.getLogger(__name__)

from shared import CUDATarget, rand_seed
from shared.auto_scheduler import auto_sched_ntrials, get_log_filename, \
                                  local_rpc_measure_ctx
from shared.logger import ScopedTimer
from ops.shared.logger import PySchedLogger, TFLOPSLogger
from ops.shared.utils  import get_time_evaluator_results_rpc_wrapper

_default_fflop_estimator = lambda args: 2.0 * prod(args)


class _AutoScheduler(abc.ABC):
    """
    Auto-Scheduler Interface
    """
    def _get_wkl_folder(self, wkl_func):
        if wkl_func.__name__ == 'Dense':
            return 'dense'
        elif wkl_func.__name__.startswith('BatchMatmul'):
            return 'batch_matmul'
        elif wkl_func.__name__.startswith('Conv2d'):
            return 'conv2d'
        else:
            assert False, "Un-handled wkl_func={}".format(wkl_func.__name__)

    def _get_sched_func_name(self, wkl_func, wkl_func_args_or_mk):
        sched_func_name = \
                re.sub(r"([a-z])([A-Z])", r"\1_\2", wkl_func.__name__).lower() + '_'
        if isinstance(wkl_func_args_or_mk, str):
            sched_func_name += wkl_func_args_or_mk
        if isinstance(wkl_func_args_or_mk, tuple):
            sched_func_name += 'x'.join(['{}'.format(arg) for arg in wkl_func_args_or_mk])
        logger.info("Returning sched_func_name={}".format(sched_func_name))
        return sched_func_name

    def _train(self, func, args, sched_log_fname, shape_vars=None,
               wkl_insts=None, wkl_inst_weights=None):
        if shape_vars is not None:
            search_task = SearchTask(func=func, args=args,
                                     shape_vars=shape_vars, wkl_insts=wkl_insts,
                                     wkl_inst_weights=wkl_inst_weights,
                                     target=CUDATarget)
        else:
            search_task = SearchTask(func=func, args=args, target=CUDATarget)
        
        tune_option = TuningOptions(
                          num_measure_trials=auto_sched_ntrials,
                          runner=local_rpc_measure_ctx.runner,
                          measure_callbacks=[RecordToFile(sched_log_fname)]
                      )

        cost_model = XGBModel(seed=rand_seed)
        search_policy = SketchPolicy(search_task, cost_model, seed=rand_seed)
        return search_task, tune_option, search_policy
    
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def infer(self):
        pass


class AnsorAutoScheduler(_AutoScheduler):
    def train(self, wkl_func, wkl_func_args, fvendor_fixture, sched_preproc=None,
              sched_results_log_fname="temp_workspace", append_log=False,
              fflop_estimator=_default_fflop_estimator):
        logger.info("{}{}".format(wkl_func.__name__, wkl_func_args))

        sched_func_name = self._get_sched_func_name(wkl_func, wkl_func_args)

        pysched_logger = PySchedLogger(sched_results_log_fname + ".py", append_log)

        try:
            eval(sched_func_name)
        except NameError:
            pass
        else:
            logger.warn("Kernel {} has already been auto-scheduled before".format(sched_func_name))
            # return

        with ScopedTimer("ansor_autosched_timer.csv", append_log,
                         sched_func_name):
            sched_log_fname = get_log_filename('ansor', sched_func_name)
            search_task, tune_option, search_policy = \
                    self._train(func=wkl_func, args=wkl_func_args,
                                sched_log_fname=sched_log_fname
                                )
            search_task.tune(tune_option, search_policy)
            best_input, _ = load_best_record(sched_log_fname, search_task.workload_key, False)
            best_state = best_input.state
            
        pysched_logger.write_best_state(sched_func_name, search_task, best_state,
                                        sched_preproc)

        self.infer(wkl_func=wkl_func, wkl_func_args=wkl_func_args,
                   fvendor_fixture=fvendor_fixture,
                   search_task=search_task, best_state=best_state,
                   sched_preproc=sched_preproc,
                   sched_results_log_fname=sched_results_log_fname,
                   append_log=append_log,
                   fflop_estimator=fflop_estimator)

    def infer(self, wkl_func, wkl_func_args, fvendor_fixture,
              search_task=None, best_state=None, sched_preproc=None,
              sched_results_log_fname="temp_workspace", append_log=False,
              fflop_estimator=_default_fflop_estimator):
        logger.info("{}{}".format(wkl_func.__name__, wkl_func_args))

        tflops_logger = TFLOPSLogger(sched_results_log_fname + ".csv", append_log)

        vendor_fixture = fvendor_fixture(*wkl_func_args)

        FLOPs = fflop_estimator(wkl_func_args)

        # cuBLAS & cuDNN
        vendor_results = \
                get_time_evaluator_results_rpc_wrapper(
                    wkl_func=vendor_fixture,
                    wkl_func_args=wkl_func_args,
                    fixture=vendor_fixture
                )
        tflops_logger.write('Vendor', wkl_func_args, vendor_results, FLOPs)

        # Ansor
        assert search_task is not None
        compute_dag = ComputeDAG(wkl_func(*wkl_func_args))
        sched_func_or_str = compute_dag.print_python_code_from_state(best_state)
        if sched_preproc is not None:
            sched_func_or_str = sched_preproc + sched_func_or_str

        ansor_results = \
                get_time_evaluator_results_rpc_wrapper(
                    wkl_func=wkl_func,
                    wkl_func_args=wkl_func_args,
                    fixture=vendor_fixture,
                    sched_func_or_str=sched_func_or_str,
                    verify_correctness=True
                )
        tflops_logger.write('Ansor', wkl_func_args, ansor_results, FLOPs)


class DietCodeAutoScheduler(_AutoScheduler):
    def train(self, wkl_func, wkl_func_args, shape_vars, wkl_insts, wkl_inst_weights,
              fvendor_fixture, sched_func_name_prefix, sched_preproc=None,
              sched_results_log_fname="temp_workspace", append_log=False,
              fflop_estimator=_default_fflop_estimator):
        logger.info("{}{}".format(wkl_func.__name__, wkl_func_args))

        pysched_logger = PySchedLogger(sched_results_log_fname + ".py", append_log)

        with ScopedTimer("dietcode_autosched_timer.csv", append_log,
                         sched_func_name_prefix):
            search_task, dietcode_dispatcher = \
                    self._train(func=wkl_func, args=wkl_func_args,
                                sched_log_fname=get_log_filename('dietcode', sched_func_name_prefix),
                                shape_vars=shape_vars, wkl_insts=wkl_insts,
                                wkl_inst_weights=wkl_inst_weights)

        pysched_logger.write_dietcode_dispatcher(sched_func_name_prefix,
                                                 search_task,
                                                 dietcode_dispatcher)

        self.infer(wkl_func=wkl_func,
                   wkl_func_args=wkl_func_args,
                   shape_vars=shape_vars, wkl_insts=wkl_insts,
                   fvendor_fixture=fvendor_fixture, 
                   dietcode_dispatcher=dietcode_dispatcher,
                   sched_preproc=sched_preproc,
                   sched_results_log_fname=sched_results_log_fname,
                   append_log=append_log,
                   fflop_estimator=fflop_estimator)

    def infer(self, wkl_func, wkl_func_args, shape_vars, wkl_insts,
              fvendor_fixture, dietcode_dispatcher=None, sched_log_fname=None, sched_preproc=None,
              sched_results_log_fname='temp_workspace', append_log=False,
              fflop_estimator=_default_fflop_estimator):
        tflops_logger = TFLOPSLogger(sched_results_log_fname + ".csv", append_log)

        from tvm.auto_scheduler import instantiate_dyn_args

        if dietcode_dispatcher is None:
            logger.info("Loading the dispatcher from {}".format(sched_log_fname))
            dietcode_dispatcher = load_records(sched_log_fname)[1][-1]

        for wkl_inst_i, wkl_inst in enumerate(wkl_insts):
            logger.info("Workload Instance={}".format(wkl_inst))
            
            instantiated_wkl_func_args = \
                    instantiate_dyn_args(wkl_func_args, shape_vars, wkl_inst)
            vendor_fixture = fvendor_fixture(*instantiated_wkl_func_args)

            FLOPs = fflop_estimator(instantiated_wkl_func_args)

            # cuBLAS & cuDNN
            vendor_results = \
                    get_time_evaluator_results_rpc_wrapper(
                        wkl_func=vendor_fixture,
                        wkl_func_args=instantiated_wkl_func_args,
                        fixture=vendor_fixture
                    )
            tflops_logger.write('Vendor', instantiated_wkl_func_args, vendor_results, FLOPs)
            
            compute_dag = ComputeDAG(wkl_func(*instantiated_wkl_func_args))
            sched_str = compute_dag.print_python_code_from_state(
                            dietcode_dispatcher.dispatch_to_state(wkl_inst)
                        )
            if sched_preproc is not None:
                sched_str = sched_preproc + sched_str
            dietcode_jit_results = \
                    get_time_evaluator_results_rpc_wrapper(
                        wkl_func=wkl_func,
                        wkl_func_args=instantiated_wkl_func_args,
                        fixture=vendor_fixture,
                        sched_func_or_str=sched_str,
                        verify_correctness=True,
                        kernel_log_filename="scratchpad_JIT_{}.cu".format(wkl_inst_i)
                    )
            tflops_logger.write('DietCode', instantiated_wkl_func_args, dietcode_jit_results, FLOPs)
