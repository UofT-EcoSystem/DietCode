from tvm import tir

import logging

logger = logging.getLogger(__name__)

from ...shared import tvm_dev_decor
from ..shared.auto_scheduler import DietCodeAutoScheduler
from .fixture import BatchMatmulNT, cuBLASBatchMatmulNTFixture, \
                     BatchMatmulNN, cuBLASBatchMatmulNNFixture

auto_scheduler = DietCodeAutoScheduler()


@tvm_dev_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    NH = pytestconfig.getoption('NH')
    T = pytestconfig.getoption('T')
    H = pytestconfig.getoption('H')

    DynT = tir.DynShapeVar('T')

    auto_scheduler.train(wkl_func=BatchMatmulNT,
                         wkl_func_args=(B * NH, DynT, H // NH, DynT),
                         shape_vars=[DynT], wkl_insts=[(T,)],
                         wkl_inst_weights=[1.],
                         fcublas_fixture=cuBLASBatchMatmulNTFixture,
                         sched_func_name_prefix='batch_matmul_nt_{}x{T}x{}x{T}'.format(B * NH, H // NH, T=T)
                         )


@tvm_dev_decor
def test_infer(pytestconfig):
    B = pytestconfig.getoption('B')
    NH = pytestconfig.getoption('NH')
    T = pytestconfig.getoption('T')
    H = pytestconfig.getoption('H')

    DynT = tir.DynShapeVar('T')

    auto_scheduler.infer(wkl_func=BatchMatmulNT,
                         wkl_func_args=(B * NH, DynT, H // NH, DynT),
                         shape_vars=[DynT], wkl_insts=[(T,)],
                         fcublas_fixture=cuBLASBatchMatmulNTFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )


@tvm_dev_decor
def test_train_dynT_1():
    B = 16
    NH = 12
    T = list(range(5, 128, 19))
    T.append(128)
    H = 768

    logger.info("Workload Instances: {}".format(T))
    wkl_insts = [(t,) for t in T]

    DynT = tir.DynShapeVar('T')

    auto_scheduler.train(wkl_func=BatchMatmulNT,
                         wkl_func_args=(B * NH, DynT, H // NH, DynT),
                         shape_vars=[DynT], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASBatchMatmulNTFixture,
                         sched_func_name_prefix='batch_matmul_nt_{}xTx{}xT'.format(B * NH, H // NH)
                         )


@tvm_dev_decor
def test_train_dynT_2():
    B = 16
    NH = 12
    T = list(range(5, 128, 19))
    T.append(128)
    H = 768

    logger.info("Workload Instances: {}".format(T))
    wkl_insts = [(t,) for t in T]

    DynT = tir.DynShapeVar('T')

    auto_scheduler.train(wkl_func=BatchMatmulNT,
                         wkl_func_args=(B * NH, DynT, DynT, H // NH),
                         shape_vars=[DynT], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASBatchMatmulNTFixture,
                         sched_func_name_prefix='batch_matmul_nt_{}xTxTx{}'.format(B * NH, H // NH)
                         )


@tvm_dev_decor
def test_train_nn_dynT():
    B = 16
    NH = 12
    T = list(range(5, 128, 19))
    T.append(128)
    H = 768

    logger.info("Workload Instances: {}".format(T))
    wkl_insts = [(t,) for t in T]

    DynT = tir.DynShapeVar('T')

    auto_scheduler.train(wkl_func=BatchMatmulNN,
                         wkl_func_args=(B * NH, DynT, DynT, H // NH),
                         shape_vars=[DynT], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASBatchMatmulNNFixture,
                         sched_func_name_prefix='batch_matmul_nn_{}xTxTx{}'.format(B * NH, H // NH)
                         )


@tvm_dev_decor
def test_infer_dynT(pytestconfig):
    B = 16
    NH = 12
    T = list(range(5, 128, 19))
    T.append(128)
    H = 768

    logger.info("Workload Instances: {}".format(T))
    wkl_insts = [(t,) for t in T]

    DynT = tir.DynShapeVar('T')

    auto_scheduler.infer(wkl_func=BatchMatmulNT,
                         wkl_func_args=(B * NH, DynT, H // NH, DynT),
                         shape_vars=[DynT], wkl_insts=wkl_insts,
                         fcublas_fixture=cuBLASBatchMatmulNTFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )

@tvm_dev_decor
def test_infer_nn_dynT(pytestconfig):
    B = 16
    NH = 12
    T = list(range(5, 128, 19))
    T.append(128)
    H = 768

    logger.info("Workload Instances: {}".format(T))
    wkl_insts = [(t,) for t in T]

    DynT = tir.DynShapeVar('T')

    auto_scheduler.infer(wkl_func=BatchMatmulNN,
                         wkl_func_args=(B * NH, DynT, DynT, H // NH),
                         shape_vars=[DynT], wkl_insts=wkl_insts,
                         fcublas_fixture=cuBLASBatchMatmulNNFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )
