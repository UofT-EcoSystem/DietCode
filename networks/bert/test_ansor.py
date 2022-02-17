import logging

logger = logging.getLogger(__name__)

from ...shared import tvm_base_decor
from ..shared.auto_scheduler import AnsorAutoScheduler
from .fixture import PyTorchBERTFixture

ansor_auto_scheduler = AnsorAutoScheduler()


@tvm_base_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    ansor_auto_scheduler.train(fmodel_fixture=PyTorchBERTFixture, model_args=(B, T))


@tvm_base_decor
def test_infer(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    
    ansor_auto_scheduler.infer(fmodel_fixture=PyTorchBERTFixture, model_args=(B, T),
                               sched_log_fname=pytestconfig.getoption('sched_log_fname'))


@tvm_base_decor
def test_train_dynT():
    B = 16
    for i, T in enumerate(list(range(5, 128, 19)) + [128]):
        ansor_auto_scheduler.train(fmodel_fixture=PyTorchBERTFixture, model_args=(B, T),
                                   append_log=False if i == 0 else True)
