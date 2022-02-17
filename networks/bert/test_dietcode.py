import logging
import os

logger = logging.getLogger(__name__)

from ...shared import tvm_dev_decor
from ..shared.auto_scheduler import DietCodeAutoScheduler
from .fixture import PyTorchBERTFixture

dietcode_auto_scheduler = DietCodeAutoScheduler()


@tvm_dev_decor
def test_infer(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    
    dietcode_auto_scheduler.infer(fmodel_fixture=PyTorchBERTFixture, model_args=(B, T),
                                  dietcode_sched_log_dir=os.path.dirname(os.path.abspath(__file__)) + '/../../ops/',
                                  ansor_sched_log_fname=pytestconfig.getoption('sched_log_fname'),
                                  )


@tvm_dev_decor
def test_infer_dynT():
    B = 16

    curr_working_dir = os.path.dirname(os.path.abspath(__file__))

    for i, T in enumerate(list(range(5, 128, 19)) + [128]):
        dietcode_auto_scheduler.infer(fmodel_fixture=PyTorchBERTFixture, model_args=(B, T),
                                      dietcode_sched_log_dir=curr_working_dir + '/../../ops/',
                                      ansor_sched_log_fname=curr_working_dir + '/saved_schedules_G4/ansor_autosched_bert_16x{}.json'.format(T),
                                      append_log=False if i == 0 else True
                                      )
