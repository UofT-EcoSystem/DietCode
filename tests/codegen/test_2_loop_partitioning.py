import filecmp
from flaky import flaky
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

from shared import CUDAContext, dietcode_decor, NoLocalPadding, DoLoopPartitioning

from ops.shared.utils import get_time_evaluator_results_rpc_wrapper


@flaky(max_runs=3)
@dietcode_decor
def test_loop_partitioning():
    """
    This test case shows the performance improvement brought by loop
    partitioning. We compare the compute throughputs between two schedules, one
    without local padding (baseline) and the other with loop partitioning.
    """
    from ops.dense.sample_schedule import dense_128x128x4
    from ops.dense.fixture import Dense, cuBLASDenseFixture


    B, T, I, H = 16, 60, 770, 2304
    TFLOPs = 2 * B * T * I * H / 1e12
    
    wkl_func_args = (B * T, I, H)
    cublas_fixture = cuBLASDenseFixture(*wkl_func_args)

    # temporarily disable local padding
    with NoLocalPadding():
        baseline_perf_results = get_time_evaluator_results_rpc_wrapper(
                                    wkl_func=Dense,
                                    wkl_func_args=wkl_func_args,
                                    sched_func_or_str=dense_128x128x4,
                                    fixture=cublas_fixture,
                                    print_kernel=True
                                )

    with DoLoopPartitioning():
        nimble_perf_results = get_time_evaluator_results_rpc_wrapper(
                                   wkl_func=Dense,
                                   wkl_func_args=wkl_func_args,
                                   sched_func_or_str=dense_128x128x4,
                                   fixture=cublas_fixture,
                                   print_kernel=True,
                                   log_kernel_filename="temp_workspace.log",
                                   verify_correctness=True
                               )
    assert filecmp.cmp(os.path.dirname(os.path.realpath(__file__))
                            + "/saved_artifacts/test_loop_partitioning.cu",
                       "temp_workspace.log")

    baseline_tflops = TFLOPs / np.average(baseline_perf_results)
    dietcode_tflops = TFLOPs / np.average(nimble_perf_results)
    logger.info(f"Baseline vs. DietCode: {baseline_tflops} vs. {dietcode_tflops} (TFLOPS)")

    if CUDAContext.device_name == 'NVIDIA GeForce RTX 3090':
        np.testing.assert_allclose(baseline_tflops, 0.98, atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(dietcode_tflops, 11.3, atol=1e-1, rtol=1e-1)


@flaky(max_runs=3)
@dietcode_decor
def test_loop_partitioning_ii():
    from ops.batch_matmul.sample_schedule import batch_matmul_nt_1x128x128x8
    from ops.batch_matmul.fixture import BatchMatmulNT, cuBLASBatchMatmulNTFixture
    

    B, T, H, NH = 16, 120, 768, 12
    TFLOPs = 2 * B * T * T * H / 1e12
    
    wkl_func_args = (B * NH, T, H // NH, T)
    cublas_fixture = cuBLASBatchMatmulNTFixture(*wkl_func_args)

    # temporarily disable local padding
    with NoLocalPadding():
        baseline_perf_results = get_time_evaluator_results_rpc_wrapper(
                                    wkl_func=BatchMatmulNT,
                                    wkl_func_args=wkl_func_args,
                                    sched_func_or_str=batch_matmul_nt_1x128x128x8,
                                    fixture=cublas_fixture,
                                    print_kernel=True,
                                    log_kernel_filename="temp_workspace_ii.log",
                                )

    with DoLoopPartitioning():
        nimble_perf_results = get_time_evaluator_results_rpc_wrapper(
                                  wkl_func=BatchMatmulNT,
                                  wkl_func_args=wkl_func_args,
                                  sched_func_or_str=batch_matmul_nt_1x128x128x8,
                                  fixture=cublas_fixture,
                                  print_kernel=True,
                                  log_kernel_filename="temp_workspace.log",
                                  verify_correctness=True
                              )

    baseline_tflops = TFLOPs / np.average(baseline_perf_results)
    nimble_tflops = TFLOPs / np.average(nimble_perf_results)
    logger.info(f"Baseline vs. DietCode: {baseline_tflops} vs. {nimble_tflops} (TFLOPS)")
