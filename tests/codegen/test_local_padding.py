import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

from ...shared import dietcode_decor

from ..ops.dense.sample_schedule import dense_2048x768x2304
from ..ops.dense.fixture import Dense
from ..ops.shared.utils import get_time_evaluator_results_rpc_wrapper


@dietcode_decor
def test_local_padding():
    """
    This test case shows the performance improvement brought by local padding.
    We compare the compute throughputs between two schedules, one without local
    padding (denoted as 'baseline') and the other with local padding.
    """

    B, T, I, H = 16, 60, 768, 2304
    TFLOPS = 2 * B * T * I * H
    
    wkl_func_args = (B * T, I, H)
    cublas_fixture = cuBLASDenseFixture(*wkl_func_args)

    os.environ['DIETCODE_CODEGEN_OPT'] = '0'
    baseline_perf_results = get_time_evaluator_results_rpc_wrapper(
                                wkl_func=Dense, wkl_func_args=wkl_func_args,
                                sched_func_or_str=dense_2048x768x2304,
                                fixture=cublas_fixture
                            )
    os.environ['DIETCODE_CODEGEN_OPT'] = '1'

    dietcode_perf_results = get_time_evaluator_results_rpc_wrapper(
                                wkl_func=Dense, wkl_func_args=wkl_func_args,
                                sched_func_or_str=dense_2048x768x2304,
                                fixture=cublas_fixture
                            )

    logger.info("Baseline vs. DietCode: {} vs. {}".format(
                    TFLOPS / np.average(baseline_perf_results) / 1e12,
                    TFLOPS / np.average(dietcode_perf_results) / 1e12
                ))
