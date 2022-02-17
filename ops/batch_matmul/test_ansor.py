from ...shared import tvm_base_decor
from ..shared.auto_scheduler import AnsorAutoScheduler
from .fixture import BatchMatmulNT, cuBLASBatchMatmulNTFixture, \
                     BatchMatmulNN, cuBLASBatchMatmulNNFixture

auto_scheduler = AnsorAutoScheduler()


@tvm_base_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    NH = pytestconfig.getoption('NH')
    T = pytestconfig.getoption('T')
    H = pytestconfig.getoption('H')

    auto_scheduler.train(wkl_func=BatchMatmulNT,
                         wkl_func_args=(B * NH, T, H // NH, T),
                         fcublas_fixture=cuBLASBatchMatmulNTFixture
                         )


@tvm_base_decor
def test_train_dynT():
    B = 16
    NH = 12
    H = 768
    for i, T in enumerate(list(range(5, 128, 19)) + [128]):
        auto_scheduler.train(wkl_func=BatchMatmulNT,
                             wkl_func_args=(B * NH, T, H // NH, T),
                             fcublas_fixture=cuBLASBatchMatmulNTFixture,
                             append_log=False if i == 0 else True
                             )


@tvm_base_decor
def test_train_nn_dynT():
    B = 16
    NH = 12
    H = 768
    for i, T in enumerate(list(range(5, 128, 19)) + [128]):
        auto_scheduler.train(wkl_func=BatchMatmulNN,
                             wkl_func_args=(B * NH, T, T, H // NH),
                             fcublas_fixture=cuBLASBatchMatmulNNFixture,
                             append_log=False if i == 0 else True
                             )
