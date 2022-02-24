

@tvm_dev_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH],
                         wkl_insts=[(T, I, H)],
                         wkl_inst_weights=[1.],
                         fvendor_fixture=cuBLASDenseFixture,
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
                         shape_vars=[DynT, DynI, DynH],
                         wkl_insts=[(T, I, H)],
                         fvendor_fixture=cuBLASDenseFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )
