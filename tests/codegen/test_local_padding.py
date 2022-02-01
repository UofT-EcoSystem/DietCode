import numpy as np

from ...shared import dietcode_decor


@dietcode_decor
def test_local_padding():
    from ..ops.dense.sample_schedule import dense_2048x768x2304
    from ..ops.dense.fixture import Dense
    from ..ops.shared.utils import get_time_evaluator_results_rpc_wrapper

    B = 16
    T = 60
    I = 768
    H = 2304
    TFLOPS = 2 * B * T * I * H
    
    wkl_func_args = (B * T, I, H)
    cublas_fixture = cuBLASDenseFixture(*wkl_func_args)

    perf_results = get_time_evaluator_results_rpc_wrapper(
                       wkl_func=Dense, wkl_func_args=wkl_func_args,
                       sched_func_or_str=dense_2048x768x2304,
                       fixture=cublas_fixture
                   )
    print("Average Throughput={}".format(
              TFLOPS / np.average(perf_results) / 1e12
          ))
