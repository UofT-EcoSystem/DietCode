# <bojian/DietCode>
import tvm
from tvm import te


def matmul(lhs, rhs, transa=False, transb=False, dtype=None):
    n = lhs.shape[1] if transa else lhs.shape[0]
    m = rhs.shape[0] if transb else rhs.shape[1]
    dtype = dtype if dtype is not None else lhs.dtype
    return te.extern(
        (n, m),
        [lhs, rhs],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cutlass.matmul", ins[0], ins[1], outs[0], transa, transb
        ),
        dtype=dtype,
        name="matmul_cutlass",
    )
