from tvm import tir

import logging

logger = logging.getLogger(__name__)

from ...shared import tvm_dev_decor
from ..shared.auto_scheduler import DietCodeAutoScheduler
from ..shared.utils import cross_product
from .fixture import Dense, cuBLASDenseFixture

auto_scheduler = DietCodeAutoScheduler()


@tvm_dev_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=[(T, I, H)],
                         wkl_inst_weights=[1.],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}x{}x{}'.format(B * T, I, H)
                         )


@tvm_dev_decor
def test_infer(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.infer(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=[(T, I, H)],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )


@tvm_dev_decor
def test_train_dynT():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)
    I = 768
    H = 2304

    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H)
                         )


@tvm_dev_decor
def test_infer_dynT(pytestconfig):
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)
    I = 768
    H = 2304

    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.infer(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )


@tvm_dev_decor
def test_train_BERT_H768():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 768), (3072, 768)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTxIx768'.format(B)
                         )


@tvm_dev_decor
def test_train_BERT_H3072():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 3072)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx768x3072'.format(B)
                         )
