from tvm import auto_scheduler

import os
import logging

from .logger import remove_log_file

logger = logging.getLogger(__name__)

auto_sched_ntrials = int(os.getenv('AUTO_SCHED_NTRIALS', '20'))
logger.info("Auto-scheduler is doing {} trials".format(auto_sched_ntrials))


def get_log_filename(auto_scheduler_name, wkl_name,):
    log_filename = "{}_autosched_{}.json".format(auto_scheduler_name, wkl_name)
    remove_log_file(log_filename)
    return log_filename

runner_kwargs = {'repeat' : 3, 'min_repeat_ms' : 100, 'timeout' : 10}
measure_ctx = auto_scheduler.LocalRPCMeasureContext(**runner_kwargs)
local_runner = auto_scheduler.LocalRunner(**runner_kwargs)
