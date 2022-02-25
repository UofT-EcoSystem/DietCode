from tvm import tir

import logging

logger = logging.getLogger(__name__)

from ...shared import dietcode_decor
from ..shared.auto_scheduler import DietCodeAutoScheduler
from .fixture import Dense, cuBLASDenseFixture

auto_scheduler = DietCodeAutoScheduler()


@dietcode_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    DynT, DynI, DynH = tir.Var('T'), tir.Var('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH],
                         wkl_insts=[(T, I, H)],
                         wkl_inst_weights=[1.],
                         fvendor_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}x{}x{}'.format(B * T, I, H)
                         )
