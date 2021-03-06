"""
.. seealso::

    :py:mod:`codegen.test_1_local_padding`
"""
import filecmp
from flaky import flaky
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

from shared import CUDAContext, dietcode_decor, NoLocalPadding, \
                   DoLoopPartitioning, tolerance

from ops.shared.utils import get_time_evaluator_results_rpc_wrapper


@flaky(max_runs=3)
@dietcode_decor
def test_loop_partitioning():
    """
    This test case shows the performance improvement brought by loop
    partitioning, used as an optimization technique in the [Nimble]_ paper.
    Similar to local padding, the goal of loop partitioning is also to mitigate
    the performance overhead brought by out-of-boundary checks.
    
    Given below is an example that illustrates how loop partitioning works:

    .. code-block:: C++
    
        for (i = 0; i < ceil(T / t); ++i)
          for (j = 0: j < t; ++j)
            if (i * t + j < T) // do something
    
    The above code snippet can be transformed into

    .. code-block:: C++

        for (i = 0; i < floor(T / t); ++i)
          for (j = 0: j < t; ++j)
            // do something
        for (j = 0; j < T - floor(T / t) * t; ++j)
          // do something

    which does not have any predicates.

    We compare the compute throughputs between four schedules:

      1. without any optimization (baseline)
      2. loop partitioning
      3. local padding
      4. local padding + loop partitioning
     
    The benchmark we use is the same as the one in local padding
    (:py:func:`codegen.test_1_local_padding.test_local_padding`). Our
    evaluations show that 
    
      1. Loop partitioning can also significantly boost the performance of the
         generated CUDA kernel by as much as :math:`10\\times` (the same order
         of speedup as local padding).
      2. Although local padding and loop partitioning are orthogonal to each
         other, applying them jointly is NOT able to boost the performance in
         this case.
    
    The table below illustrates the results we get from the CI workflow:

    =========== ======== ================= ============= ===================
    GPU         Baseline Loop Partitioning Local Padding | Local Padding + 
                                                         | Loop Partitioning
    =========== ======== ================= ============= ===================
    RTX 3090    ~0.98    ~11.2             ~11.6         ~11.0
    RTX 2080 Ti ~0.43    ~5.98             ~5.27
    =========== ======== ================= ============= ===================

    where the numbers denote the compute throughputs (in TFLOPs/sec), and hence
    the higher the better.
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

    with NoLocalPadding(), DoLoopPartitioning():
        loop_partitioning_perf_results = get_time_evaluator_results_rpc_wrapper(
                                             wkl_func=Dense,
                                             wkl_func_args=wkl_func_args,
                                             sched_func_or_str=dense_128x128x4,
                                             fixture=cublas_fixture,
                                             print_kernel=True,
                                             log_kernel_filename="temp_workspace.log",
                                             verify_correctness=True
                                         )
    with DoLoopPartitioning():
        partition_and_pad_perf_results = \
            get_time_evaluator_results_rpc_wrapper(
                wkl_func=Dense,
                wkl_func_args=wkl_func_args,
                sched_func_or_str=dense_128x128x4,
                fixture=cublas_fixture,
                print_kernel=True,
                log_kernel_filename="temp_workspace_ii.log",
                verify_correctness=True
            )

    baseline_tflops = TFLOPs / np.average(baseline_perf_results)
    loop_partitioning_tflops = TFLOPs / np.average(loop_partitioning_perf_results)
    partition_and_pad_tflops = TFLOPs / np.average(partition_and_pad_perf_results)
    logger.info( "Baseline vs. Loop Partitioning vs. Loop Partition + Local Padding: "
                f"{baseline_tflops} vs. {loop_partitioning_tflops} vs. "
                f"{partition_and_pad_tflops} (TFLOPS)"
                )

    assert filecmp.cmp(os.path.dirname(os.path.realpath(__file__))
                            + "/saved_artifacts/test_loop_partitioning.cu",
                       "temp_workspace.log")

    if CUDAContext.device_name == 'NVIDIA GeForce RTX 3090':
        np.testing.assert_allclose(baseline_tflops, 0.98, **tolerance)
        np.testing.assert_allclose(loop_partitioning_tflops, 11.2, **tolerance)
        np.testing.assert_allclose(partition_and_pad_tflops, 11.0, **tolerance)
    if CUDAContext.device_name == 'NVIDIA GeForce RTX 2080 Ti':
        np.testing.assert_allclose(baseline_tflops, 0.42, **tolerance)
        np.testing.assert_allclose(loop_partitioning_tflops, 5.98, **tolerance)


@flaky(max_runs=3)
@dietcode_decor
def test_loop_partitioning_ii():
    """
    Therefore, one natural question to ask is:

    **What is the difference between local padding and loop partitioning, given
    that they share the same objective?**

    Although the two techniques are indeed similar in what they are going to
    achieve, we pick local padding primarily due to the following reasons:

    - Due to the *partition* nature, **loop partitioning needs to duplicate the
      body statements** (as can be seen in the simple example illustrated
      before, where the comment ``// do something`` appears twice). Depending on
      the number of spatial axes, this duplication can happen multiple times.
      This can significantly elongate the CUDA kernel body, and further lead to
      a gigantic kernel in the case when the original kernel body is already
      long (e.g., because of a large unrolling factor), **which eventually
      increases the compilation time for the kernel** (can be several minutes
      for a single kernel by our measurements).
    - **There are cases that can be handled by local padding but NOT by loop
      partitioning**. We refer to the example below, which is again the same as
      the one in local padding
      (:py:func:`codegen.test_1_local_padding.test_local_padding_ii`). In this
      example, all the thread blocks possess the same predicates, which prevents
      the loop partitioning from taking place (since there is no region that can
      be partitioned out without predicates).

    Nevertheless, there are still advantages of loop partitioning that local
    padding does not have: Since **loop partitioning does not involve padding**,
    it does not have to perform any redundant computations, which explains why
    its performance can be better than local padding on RTX 2080 Ti (as can be
    seen in the previous table). However, this is NOT a problem for the DietCode
    auto-scheduling framework since we expect that the auto-scheduling outcomes
    should have relatively low padding ratio.

    .. [Nimble] H. Shen et al. *Nimble: Efficiently Compiling Dynamic Neural
                Networks for Model Inference*. MLSys 2021
    """
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
                                    log_kernel_filename="temp_workspace.log",
                                )

    with NoLocalPadding(), DoLoopPartitioning():
        loop_partitioning_perf_results = get_time_evaluator_results_rpc_wrapper(
                                             wkl_func=BatchMatmulNT,
                                             wkl_func_args=wkl_func_args,
                                             sched_func_or_str=batch_matmul_nt_1x128x128x8,
                                             fixture=cublas_fixture,
                                             print_kernel=True,
                                             log_kernel_filename="temp_workspace_ii.log",
                                             verify_correctness=True
                                         )
    # Loop partitioning is not able to optimize for this case, hence there
    # should NOT be any difference in the generated code.
    assert filecmp.cmp("temp_workspace_ii.log", "temp_workspace.log")

    baseline_tflops = TFLOPs / np.average(baseline_perf_results)
    loop_partitioning_tflops = TFLOPs / np.average(loop_partitioning_perf_results)
    logger.info(f"Baseline vs. Loop Partitioning: {baseline_tflops} vs. {loop_partitioning_tflops} (TFLOPS)")
