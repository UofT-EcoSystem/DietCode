import numpy as np
import tvm
from tvm import te
from tvm.contrib.popen_pool import PopenPoolExecutor, StatusKind

import logging

logger = logging.getLogger(__name__)

from shared import CUDAContext, CUDATarget


def get_time_evaluator_results(wkl_func, wkl_func_args, fixture,
                               sched_func_or_str=None,
                               print_kernel=False, log_kernel_filename=None,
                               verify_correctness=False):
    """
    Measure the given workload with the provided schedule string (generated from
    the auto-scheduler) or schedule function.

    Parameters
    ==========
    wkl_func            : Workload Function
    wkl_func_args       : Workload Function Arguments
    fixture             : Operator Fixture, used for checking correctness
    print_kernel        : Whether to print the kernel to the console
    log_kernel_filename : Filename to dump kernel output
    verify_correctness  : Whether to verify the correctness of the generated
                          kernel (using the fixture)
    """
    tensor_args = wkl_func(*wkl_func_args)
    s = te.create_schedule(tensor_args[-1].op)
    if sched_func_or_str is not None:
        if isinstance(sched_func_or_str, str):
            exec('{} = tensor_args'.format(', '.join([t.op.name for t in tensor_args])))
            exec(sched_func_or_str)
        else:
            assert callable(sched_func_or_str)
            sched_func_or_str(*tensor_args, s)
    kernel = tvm.build(s, tensor_args, target=CUDATarget)

    if print_kernel:
        kernel_src = kernel.imported_modules[0].get_source()

        logger.info("{}".format(tvm.lower(s, tensor_args, simple_mode=True)))
        logger.info("{}".format(kernel_src))

        if log_kernel_filename is not None:
            with open(log_kernel_filename, 'w') as fout:
                fout.write("{}".format(kernel_src))

    if hasattr(fixture, "module_data_fixture_view"):
        # Sometimes, the fixture has constraint on the module data layout (e.g.,
        # the cuDNN on the conv2d operator). Therefore, if
        # `module_data_fixture_view` is explicitly specified, we have to invoke
        # that method instead of `module_data`.
        module_data = [tvm.nd.array(d, device=CUDAContext) for d in fixture.module_data_fixture_view()]
    else:
        module_data = [tvm.nd.array(d, device=CUDAContext) for d in fixture.module_data()]

    kernel(*module_data)
    if verify_correctness:
        np.testing.assert_allclose(module_data[-1].asnumpy(), fixture.Y_np,
                                   rtol=1e-3, atol=1e-3)

    warmup_evaluator = kernel.time_evaluator(kernel.entry_name, CUDAContext,
                                             number=300, repeat=1, min_repeat_ms=300)
    warmup_evaluator(*module_data)  # <- GPU warm-up, needed for stabilizing performance
    time_evaluator = kernel.time_evaluator(kernel.entry_name, CUDAContext,
                                           number=100, repeat=10, min_repeat_ms=100)
    return time_evaluator(*module_data).results


def get_time_evaluator_results_rpc_wrapper(*args, **kwargs):
    """
    get_time_evaluator_results (RPC version)
    """
    def _get_time_evaluator_results_rpc_worker(args):
        """
        RPC worker wrapper for getting time evaluator results.
        """
        return get_time_evaluator_results(*args[0], **args[1])

    exec = PopenPoolExecutor(1)
    results_gen = exec.map_with_error_catching(
                      _get_time_evaluator_results_rpc_worker, [(args, kwargs)],
                  )
    results = [r for r in results_gen]
    if results[0].status != StatusKind.COMPLETE:
        logger.warn("Exception=({}) caught during measurements".format(results[0].value))
        return None
    else:
        return results[0].value
