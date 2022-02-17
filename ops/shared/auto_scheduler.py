import tvm
from tvm.auto_scheduler import ComputeDAG, SearchTask, RecordToFile, \
                               SketchPolicy, TuningOptions, XGBModel, \
                               load_best_record, load_records

import abc
import logging
import numpy as np
import os
import re

from math import prod

logger = logging.getLogger(__name__)

from ...shared import CUDATarget, CUDAContext, platform, rand_seed, use_tvm_base
from ...shared.auto_scheduler import auto_sched_ntrials, get_log_filename, \
                                     measure_ctx
from ...shared.logger import AutoSchedTimer
from .logger import PySchedLogger, TFLOPSLogger
from .utils  import get_time_evaluator_results

exec('from ..batch_matmul.saved_schedules_{} import *'.format(platform))
exec('from ..dense.saved_schedules_{} import *'.format(platform))

_default_fflop_estimator = lambda args: 2.0 * prod(args)


class _AutoScheduler(abc.ABC):
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
            task = SearchTask(func=func, args=args,
                              shape_vars=shape_vars, wkl_insts=wkl_insts,
                              wkl_inst_weights=wkl_inst_weights,
                              target=CUDATarget)
        else:
            task = SearchTask(func=func, args=args, target=CUDATarget)
        
        tune_option = TuningOptions(
                          num_measure_trials=auto_sched_ntrials,
                          runner=measure_ctx.runner,
                          measure_callbacks=[RecordToFile(sched_log_fname)]
                      )

        cost_model = XGBModel(seed=rand_seed)
        search_policy = SketchPolicy(task, cost_model, seed=rand_seed)

        if use_tvm_base:
            task.tune(tune_option, search_policy)
            best_input, _ = \
                    load_best_record(sched_log_fname, task.workload_key, False)
            best_state = best_input.state
            return task, best_state, task.compute_dag.apply_steps_from_state(best_state)

        if shape_vars is None:
            # static workload
            best_state, sched, in_args = task.tune(tune_option, search_policy)
            return task, best_state, (sched, in_args)
        else:
            return task, task.tune(tune_option, search_policy)  # dyn_wkl_dispatcher
    
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def infer(self):
        pass


class AnsorAutoScheduler(_AutoScheduler):
    def train(self, wkl_func, wkl_func_args, fcublas_fixture,
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

        with AutoSchedTimer("ansor_autosched_timer.csv", append_log,
                            sched_func_name):
            search_task, best_state, sched_args_pair = \
                    self._train(func=wkl_func, args=wkl_func_args,
                                sched_log_fname=get_log_filename('ansor', sched_func_name)
                                )
        pysched_logger.write_best_state(sched_func_name, search_task, best_state)

        self.infer(wkl_func=wkl_func, wkl_func_args=wkl_func_args,
                   fcublas_fixture=fcublas_fixture,
                   sched_args_pair=sched_args_pair, micro_kernel=None,
                   sched_results_log_fname=sched_results_log_fname,
                   append_log=append_log,
                   fflop_estimator=fflop_estimator)

    def infer(self, wkl_func, wkl_func_args, fcublas_fixture,
              sched_args_pair=None, micro_kernel=None,
              sched_results_log_fname="temp_workspace", append_log=False,
              fflop_estimator=_default_fflop_estimator):
        logger.info("{}{}".format(wkl_func.__name__, wkl_func_args))

        tflops_logger = TFLOPSLogger(sched_results_log_fname + ".csv", append_log)

        cublas_fixture = fcublas_fixture(*wkl_func_args)
        module_data = cublas_fixture.module_data()

        FLOPs = fflop_estimator(wkl_func_args)

        # cuBLAS
        cublas_results = \
                get_time_evaluator_results(cublas_fixture.cublas_kernel,
                                           module_data, CUDAContext)
        tflops_logger.write('cuBLAS', wkl_func_args, cublas_results, FLOPs)

        # Ansor
        try:
            if sched_args_pair is not None:
                ansor_kernel = tvm.build(*sched_args_pair, target=CUDATarget)
            else:
                tensor_args = wkl_func(*wkl_func_args)
                sched = tvm.te.create_schedule(tensor_args[-1].op)
                sched_func = eval(self._get_sched_func_name(wkl_func, wkl_func_args))
                sched_func(*tensor_args, sched)
                ansor_kernel = tvm.build(sched, tensor_args, target=CUDATarget)
            ansor_results = \
                    get_time_evaluator_results(ansor_kernel, module_data,
                                               CUDAContext)
            tflops_logger.write('Ansor', wkl_func_args, ansor_results, FLOPs)
        except NameError:
            tflops_logger.write('Ansor', wkl_func_args, None, FLOPs)

        if micro_kernel is None:
            return
        logger.info("Loading micro-kernel={}".format(micro_kernel))

        tensor_args = wkl_func(*wkl_func_args)
        sched = tvm.te.create_schedule(tensor_args[-1].op)
        sched_func = eval(self._get_sched_func_name(wkl_func, micro_kernel))
        sched_func(*tensor_args, sched)
        dietcode_kernel = tvm.build(sched, tensor_args, target=CUDATarget)

        dietcode_kernel_src = dietcode_kernel.imported_modules[0].get_source()

        if os.getenv("VERBOSE", "0") == "1":
            logger.info("{}".format(tvm.lower(sched, tensor_args, simple_mode=True)))
            logger.info("{}".format(dietcode_kernel_src))
        with open("scratchpad.cu", 'w') as fout:
            fout.write("{}".format(dietcode_kernel_src))

        dietcode_kernel(*module_data)
        np.testing.assert_allclose(module_data[-1].asnumpy(),
                                   cublas_fixture.Y_np_expected,
                                   rtol=1e-3, atol=1e-3)
        dietcode_results = get_time_evaluator_results(dietcode_kernel, module_data,
                                                      CUDAContext)
        tflops_logger.write('DietCode', wkl_func_args, dietcode_results, FLOPs)



class NimbleEnv:
    def __enter__(self):
        os.environ['USE_NIMBLE'] = str(1)
        os.environ['DIETCODE_SCHED_OPT'] = str(0)
        os.environ['DIETCODE_SCHED_OPT_PARTITION_CONST_LOOPS'] = str(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del os.environ['USE_NIMBLE']
        os.environ['DIETCODE_SCHED_OPT'] = str(1)
        os.environ['DIETCODE_SCHED_OPT_PARTITION_CONST_LOOPS'] = str(0)



class DietCodeAutoScheduler(_AutoScheduler):
    def train(self, wkl_func, wkl_func_args, shape_vars, wkl_insts, wkl_inst_weights,
              fcublas_fixture, sched_func_name_prefix,
              sched_results_log_fname="temp_workspace", append_log=False,
              fflop_estimator=_default_fflop_estimator):
        logger.info("{}{}".format(wkl_func.__name__, wkl_func_args))

        pysched_logger = PySchedLogger(sched_results_log_fname + ".py", append_log)

        with AutoSchedTimer("dietcode_autosched_timer.csv", append_log,
                            sched_func_name_prefix):
            search_task, dyn_wkl_dispatcher = \
                    self._train(func=wkl_func, args=wkl_func_args,
                                sched_log_fname=get_log_filename('dietcode', sched_func_name_prefix),
                                shape_vars=shape_vars, wkl_insts=wkl_insts,
                                wkl_inst_weights=wkl_inst_weights)

        pysched_logger.write_dyn_wkl_disp(sched_func_name_prefix, search_task, dyn_wkl_dispatcher)

        self.infer(wkl_func=wkl_func, wkl_func_args=wkl_func_args,
                   shape_vars=shape_vars, wkl_insts=wkl_insts,
                   fcublas_fixture=fcublas_fixture, 
                   dyn_wkl_dispatcher=dyn_wkl_dispatcher,
                   sched_log_fname=None,
                   sched_results_log_fname=sched_results_log_fname, append_log=append_log,
                   fflop_estimator=fflop_estimator)

    def infer(self, wkl_func, wkl_func_args, shape_vars, wkl_insts,
              fcublas_fixture, dyn_wkl_dispatcher=None, sched_log_fname=None,
              sched_results_log_fname='temp_workspace', append_log=False,
              fflop_estimator=_default_fflop_estimator):
        tflops_logger = TFLOPSLogger(sched_results_log_fname + ".csv", append_log)

        from tvm.auto_scheduler import instantiate_dyn_args, replace_shape_vars

        if dyn_wkl_dispatcher is None:
            logger.info("Loading the dispatcher from {}".format(sched_log_fname))
            dyn_wkl_dispatcher = load_records(sched_log_fname)[1][-1]
            dyn_wkl_dispatcher = \
                    dyn_wkl_dispatcher.embed_compute_dag(
                        ComputeDAG(wkl_func(
                            *replace_shape_vars(wkl_func_args, shape_vars,
                                                dyn_wkl_dispatcher.search_task.shape_vars
                                                )
                        ))
                    )

        """DietCode AOT Compilation
        ir_mod, _ = tvm.lower_dyn_wkl_dispatcher(dyn_wkl_dispatcher, CUDATarget)
        dietcode_aot_kernel = tvm.build(ir_mod, target=CUDATarget)

        with open("scratchpad_AOT_host.ll", 'w') as hfout, \
             open("scratchpad_AOT_dev.cu", 'w') as dfout:
            hfout.write("{}".format(dietcode_aot_kernel.get_source()))
            dfout.write("{}".format(dietcode_aot_kernel.imported_modules[0].get_source()))
        """

        largest_wkl_inst = max(wkl_insts)
        logger.info("Picking the largest workload instance={}".format(largest_wkl_inst))
        instantiated_largest_wkl_func_args = \
                instantiate_dyn_args(wkl_func_args, shape_vars, largest_wkl_inst)

        for wkl_inst_i, wkl_inst in enumerate(wkl_insts):
            logger.info("Workload Instance={}".format(wkl_inst))
            
            instantiated_wkl_func_args = \
                    instantiate_dyn_args(wkl_func_args, shape_vars, wkl_inst)
            cublas_fixture = fcublas_fixture(*instantiated_wkl_func_args)
            module_data = cublas_fixture.module_data()

            FLOPs = fflop_estimator(instantiated_wkl_func_args)

            # cuBLAS
            cublas_results = \
                    get_time_evaluator_results(cublas_fixture.cublas_kernel,
                                               module_data, CUDAContext)
            tflops_logger.write('Vendor', instantiated_wkl_func_args, cublas_results, FLOPs)
            
            # Ansor
            ansor_results = None
            try:
                tensor_args = wkl_func(*instantiated_wkl_func_args)
                sched = tvm.te.create_schedule(tensor_args[-1].op)
                sched_func = eval(self._get_sched_func_name(wkl_func, instantiated_wkl_func_args))
                sched_func(*tensor_args, sched)

                os.environ["DIETCODE_ALLOW_REGISTER_SPILL"] = '1'
                ansor_kernel = tvm.build(sched, tensor_args, target=CUDATarget)
                os.environ["DIETCODE_ALLOW_REGISTER_SPILL"] = '0'
                with open("scratchpad_Ansor_{}.cu".format(wkl_inst_i), 'w') as fout:
                    fout.write("{}".format(ansor_kernel.imported_modules[0].get_source()))
                ansor_results = \
                        get_time_evaluator_results(ansor_kernel, module_data,
                                                   CUDAContext)
            except Exception as e:
                logger.warn("err_msg={}".format(e))
            tflops_logger.write('Ansor', instantiated_wkl_func_args, ansor_results, FLOPs)

            # Nimble
            nimble_results = None
            try:
                tensor_args = wkl_func(*instantiated_wkl_func_args)
                sched = tvm.te.create_schedule(tensor_args[-1].op)

                with NimbleEnv():
                    sched_func = eval(self._get_sched_func_name(wkl_func, instantiated_largest_wkl_func_args))
                    sched_func(*tensor_args, sched)
                    nimble_kernel = tvm.build(sched, tensor_args, target=CUDATarget)

                with open("scratchpad_Nimble_{}.cu".format(wkl_inst_i), 'w') as fout:
                    fout.write("{}".format(nimble_kernel.imported_modules[0].get_source()))
                nimble_results = \
                        get_time_evaluator_results(nimble_kernel, module_data,
                                                   CUDAContext)
            except Exception as e:
                logger.warn("err_msg={}".format(e))
            tflops_logger.write('Nimble', instantiated_wkl_func_args, nimble_results, FLOPs)

            # DietCode
            """DietCode AOT Compilation
            try:
                dietcode_aot_kernel(*module_data, *wkl_inst)
                np.testing.assert_allclose(module_data[-1].asnumpy(),
                                           cublas_fixture.Y_np_expected,
                                           rtol=1e-3, atol=1e-3)

                dietcode_aot_results = \
                        get_time_evaluator_results(dietcode_aot_kernel,
                                                   module_data + list(wkl_inst),
                                                   CUDAContext)
                tflops_logger.write('DietCode_AOT', instantiated_wkl_func_args,
                                    dietcode_aot_results, FLOPs)
            except Exception as e:
                logger.warn("err_msg={}".format(e))
                tflops_logger.write('DietCode_AOT', instantiated_wkl_func_args, None, FLOPs)
            """

            dietcode_jit_results = None
            try:
                sched, args = dyn_wkl_dispatcher.dispatch(wkl_inst)
                os.environ["DIETCODE_ALLOW_REGISTER_SPILL"] = '1'
                dietcode_jit_kernel = tvm.build(sched, args, target=CUDATarget)
                os.environ["DIETCODE_ALLOW_REGISTER_SPILL"] = '0'

                with open("scratchpad_JIT_{}.cu".format(wkl_inst_i), 'w') as fout:
                    fout.write("{}".format(dietcode_jit_kernel.imported_modules[0].get_source()))

                dietcode_jit_kernel(*module_data)
                np.testing.assert_allclose(module_data[-1].asnumpy(),
                                           cublas_fixture.Y_np_expected,
                                           rtol=1e-3, atol=1e-3)

                dietcode_jit_results = \
                        get_time_evaluator_results(dietcode_jit_kernel, module_data,
                                                   CUDAContext)
            except Exception as e:
                logger.warn("err_msg={}".format(e))
            tflops_logger.write('DietCode', instantiated_wkl_func_args, dietcode_jit_results, FLOPs)
