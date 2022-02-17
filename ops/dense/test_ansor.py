import logging

logger = logging.getLogger(__name__)

from ...shared import tvm_base_decor, tvm_dev_decor
from ..shared.auto_scheduler import AnsorAutoScheduler
from .fixture import Dense, cuBLASDenseFixture

auto_scheduler = AnsorAutoScheduler()


@tvm_base_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * T, I, H),
                         fcublas_fixture=cuBLASDenseFixture
                         )


@tvm_base_decor
def test_train_dynT():
    B, I, H = 16, 768, 2304
    for i, T in enumerate(list(range(5, 128, 19)) + [128]):
        auto_scheduler.train(wkl_func=Dense,
                             wkl_func_args=(B * T, I, H),
                             fcublas_fixture=cuBLASDenseFixture,
                             append_log=False if i == 0 else True
                             )


@tvm_dev_decor
def test_infer(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    auto_scheduler.infer(wkl_func=Dense,
                         wkl_func_args=(B * T, I, H),
                         micro_kernel=pytestconfig.getoption('micro_kernel'),
                         fcublas_fixture=cuBLASDenseFixture
                         )
