import filecmp
from flaky import flaky
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

from shared import CUDAContext, dietcode_decor, NoLocalPadding

from ops.shared.utils import get_time_evaluator_results_rpc_wrapper


@flaky(max_runs=3)
@dietcode_decor
def test_local_padding():
    """
    This test case shows the performance improvement brought by local padding.
    We compare the compute throughputs between two schedules, one without local
    padding (baseline) and the other with local padding (DietCode).
    
    We use the dense layer as an example workload:
    
    .. math::
     
        Y = XW^T, X : [B\\times T, I], W : [H, I], Y : [B\\times T, H]

    In ``tvm.te`` form:

    .. code-block:: C++
    
        for (i = 0; i < B * T; ++i)
          for (j = 0: j < H; ++j)
            for (k = 0; k < I; ++k)
              Y[i, j] += X[i, k] * W[j, k];
    
    where :math:`(B, T, I, H)` stand for (batch size, sequence length, input
    size, hidden dimension) respectively (namings adopted from the [BERT]_
    model).

    We adopt a sample schedule ``dense_128x128x4`` for this workload. The
    schedule, as its name suggests, has a tile size of :math:`(128, 128, 4)`
    along the :math:`(i, j, k)` dimension respectively. Furthermore, we
    deliberately set the value of the sequence length :math:`T` to :math:`60`,
    and the input size :math:`I` to :math:`770`, so that
    
    .. math::

        (B\\times T = 16\\times 60 = 960) \% 128 \\ne 0, (I = 770) \% 4 \\ne 0

    which is common in the case of handling dynamic-shape workloads.

    Due to the imperfect tiling, out-of-boundary checks (a.k.a., predicates)
    have to be injected inside the loop body, which can greatly hurt the
    performance. However, we make the following key observations:

    - The schedule generated by the auto-scheduler usually consists of 3 main
      stages, namely
      
      - **Fetch**: Obtain the input data from the global to the shared memory.

        .. code-block:: CUDA

            X_shared[...] = X[...]; W_shared[...] = W[...];

      - **Compute**: Compute output results using the shared memory variables
        and write to the registers.

        .. code-block:: CUDA

            Y_local[...] = X_shared[...] * W_shared[...];

      - **Writeback**: Write the results of the registers back to the global
        memory.
    
        .. code-block:: CUDA

            Y[...] = Y_local[...];

    - The predicates at the **Compute** stage have the biggest impact on the
      runtime performance, but the good news is they duplicate those at the
      Fetch and the Writeback stage and hence can be *safely* removed. This is
      in essence padding the compute by the size of the local workspace, hence
      the name **Local Padding**.

    Our evaluations show that local padding can significantly boost the
    performance of the generated CUDA kernel by as much as :math:`10\\times` in
    this case (on modern NVIDIA RTX GPUs). The table below illustrates the
    results we get from the CI workflow:

    =========== ======== ========
    GPU         Baseline DietCode
    =========== ======== ========
    RTX 3090    ~0.98    ~11
    RTX 2080 Ti ~0.43    ~5.2
    =========== ======== ========

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

    dietcode_perf_results = get_time_evaluator_results_rpc_wrapper(
                                wkl_func=Dense,
                                wkl_func_args=wkl_func_args,
                                sched_func_or_str=dense_128x128x4,
                                fixture=cublas_fixture,
                                print_kernel=True,
                                log_kernel_filename="temp_workspace.log",
                                verify_correctness=True
                            )
    assert filecmp.cmp(os.path.dirname(os.path.realpath(__file__))
                           + "/saved_artifacts/test_local_padding.cu",
                       "temp_workspace.log")

    baseline_tflops = TFLOPs / np.average(baseline_perf_results)
    dietcode_tflops = TFLOPs / np.average(dietcode_perf_results)
    logger.info(f"Baseline vs. DietCode: {baseline_tflops} vs. {dietcode_tflops} (TFLOPS)")

    if CUDAContext.device_name == 'NVIDIA GeForce RTX 3090':
        np.testing.assert_allclose(baseline_tflops, 0.98, atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(dietcode_tflops, 11.6, atol=1e-1, rtol=1e-1)
    if CUDAContext.device_name == 'NVIDIA GeForce RTX 2080 Ti':
        np.testing.assert_allclose(baseline_tflops, 0.43, atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(dietcode_tflops, 5.2,  atol=1e-1, rtol=1e-1)


@flaky(max_runs=3)
@dietcode_decor
def test_local_padding_ii():
    """
    This test case shows the necessity of NOT shrinking the local workspace,
    which is required for local padding.

    In TVM, the size of the local workspace (e.g., shared memory or registers)
    will be automatically shrinked if the code generator detects that the
    workspace is not fully utilized by the tensor programs. However, this
    optimization can be NOT desirable.

    For example, consider the batched matrix multiply workload:

    .. math::
    
        Y = \\operatorname{BatchMatmulNT}(X, W), X : \\left[B\\times NH, T,
        \\frac{H}{A}\\right], W : \\left[B\\times NH, T, \\frac{H}{NH}\\right],
        Y : [B\\times NH, T, T]
    
    .. code-block:: C++

        for (i = 0; i < B * NH; ++i)
          for (j = 0: j < T; ++j)
            for (k = 0; k < T; ++k)
              for (l = 0; l < H / NH; ++l)
                Y[i, j, k] += X[i, j, l] * W[i, k, l]

    where :math:`NH` stands for the number of attention heads (also adopted from
    the [BERT]_ model, other parameters are the same as above).

    We adopt a sample schedule ``batch_matmul_nt_1x128x128x8`` for this
    workload. The schedule has a tile size of :math:`(1, 128, 128, 8)` along the
    :math:`(i, j, k, l)` dimension respectively. By the tile sizes, the local
    workspace (i.e., shared memory allocations) for :math:`X` and :math:`W` are
    :math:`1\\times 128\\times 8` and :math:`1\\times 128\\times 8`
    respectively. 
    
    The allocations work fine in the case when :math:`T=128`, but if :math:`T`
    is just a little smaller (e.g., :math:`T=120`), the TVM code generator will
    find out that all the thread blocks are not fully utilizing the shared
    memory variables and shrink them, which prevents local padding from taking
    place (as it requests access to the whole workspace).

    Our evaluations show that preserving the local workspace can boost the
    performance of the generated CUDA kernel by as much as :math:`2.5\\times` in
    this case. The table below illustrates the results we get from the CI
    workflow:

    =========== ======== ========
    GPU         Baseline DietCode
    =========== ======== ========
    RTX 3090    ~3.5     ~8.7
    RTX 2080 Ti ~1.8     ~5.5
    =========== ======== ========

    where the numbers again denote the compute throughputs (in TFLOPs/sec), and
    hence the higher the better.

    .. [BERT] J. Devlin et al. *BERT: Pre-training of Deep Bidirectional
              Transformers for Language Understanding*. NAACL-NLT 2019
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
                                    print_kernel=True
                                )

    dietcode_perf_results = get_time_evaluator_results_rpc_wrapper(
                                wkl_func=BatchMatmulNT,
                                wkl_func_args=wkl_func_args,
                                sched_func_or_str=batch_matmul_nt_1x128x128x8,
                                fixture=cublas_fixture,
                                print_kernel=True,
                                log_kernel_filename="temp_workspace.log",
                                verify_correctness=True
                            )
    assert filecmp.cmp(os.path.dirname(os.path.realpath(__file__))
                           + "/saved_artifacts/test_local_padding_ii.cu",
                       "temp_workspace.log")

    baseline_tflops = TFLOPs / np.average(baseline_perf_results)
    dietcode_tflops = TFLOPs / np.average(dietcode_perf_results)
    logger.info(f"Baseline vs. DietCode: {baseline_tflops} vs. {dietcode_tflops} (TFLOPS)")

    if CUDAContext.device_name == 'NVIDIA GeForce RTX 3090':
        np.testing.assert_allclose(baseline_tflops, 3.5, atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(dietcode_tflops, 8.7, atol=1e-1, rtol=1e-1)
    if CUDAContext.device_name == 'NVIDIA GeForce RTX 2080 Ti':
        np.testing.assert_allclose(baseline_tflops, 1.8, atol=1e-1, rtol=1e-1)
        np.testing.assert_allclose(dietcode_tflops, 5.5, atol=1e-1, rtol=1e-1)
