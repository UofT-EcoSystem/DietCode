import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

from ...shared import dietcode_decor

from ..ops.dense.sample_schedule import dense_128x128
from ..ops.dense.fixture import Dense, cuBLASDenseFixture
from ..ops.shared.utils import get_time_evaluator_results_rpc_wrapper


@dietcode_decor
def test_local_padding():
    """
    This test case shows the performance improvement brought by local padding.
    We compare the compute throughputs between two schedules, one without local
    padding (baseline) and the other with local padding (DietCode).
    """
    B, T, I, H = 16, 60, 768, 2304
    TFLOPs = 2 * B * T * I * H / 1e12
    
    wkl_func_args = (B * T, I, H)
    cublas_fixture = cuBLASDenseFixture(*wkl_func_args)

    # temporarily disable local padding
    os.environ['DIETCODE_DO_LOCAL_PADDING'] = '0'
    baseline_perf_results = get_time_evaluator_results_rpc_wrapper(
                                wkl_func=Dense, wkl_func_args=wkl_func_args,
                                sched_func_or_str=dense_128x128,
                                fixture=cublas_fixture
                            )
    os.environ['DIETCODE_DO_LOCAL_PADDING'] = '1'

    dietcode_perf_results = get_time_evaluator_results_rpc_wrapper(
                                wkl_func=Dense, wkl_func_args=wkl_func_args,
                                sched_func_or_str=dense_128x128,
                                fixture=cublas_fixture
                            )

    logger.info("Baseline vs. DietCode: {} vs. {} (TFLOPS)".format(
                    TFLOPs / np.average(baseline_perf_results),
                    TFLOPs / np.average(dietcode_perf_results)
                ))
