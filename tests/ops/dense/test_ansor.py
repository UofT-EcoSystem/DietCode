from shared import base_decor
from ops.shared.auto_scheduler import AnsorAutoScheduler
from ops.dense.fixture import Dense, cuBLASDenseFixture

auto_scheduler = AnsorAutoScheduler()


@base_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * T, I, H),
                         fvendor_fixture=cuBLASDenseFixture
                         )
