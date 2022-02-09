import tvm
from tvm import auto_scheduler, te, topi

import numpy as np

from shared import CUDATarget, CUDAContext


@auto_scheduler.register_workload
def Dense(B, I, H):
    """
    TE Definition of Dense([B, I], [H, I])

    The workload definition is used by both TVM and DietCode.
    """
    X = te.placeholder((B, I), name='X')
    W = te.placeholder((H, I), name='W')
    Y = topi.nn.dense(X, W)
    return [X, W, Y]


class cuBLASDenseFixture:
    """
    cuBLAS Dense Fixture

    This fixture performs the computation
    
        Dense([B, I], [H, I])
    
    and is using the NVIDIA cuBLAS library under the hood. It is used for
    checking the correctness and testing the performance of DietCode.
    """
    __slots__ = 'B', 'I', 'H', 'X_np', 'W_np', 'Y', 'Y_np'

    def __init__(self, B, I, H):
        self.B, self.I, self.H = B, I, H
        self.X_np = np.random.uniform(-0.1, 0.1, size=(B, I)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(H, I)).astype(np.float32)

        tensor_args = self()
        self.Y = tensor_args[-1]
        sched = te.create_schedule(self.Y.op)
        cublas_kernel = tvm.build(sched, tensor_args, CUDATarget)

        module_data = [tvm.nd.array(d, device=CUDAContext) for d in self.module_data()]
        cublas_kernel(*module_data)
        self.Y_np = module_data[-1].asnumpy()

    def module_data(self):
        return [self.X_np, self.W_np,
                np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                         dtype=np.float32)
                ]

    def __call__(self):
        X = te.placeholder((self.B, self.I), name='X')
        W = te.placeholder((self.H, self.I), name='W')
        Y = tvm.contrib.cublas.matmul(X, W, transa=False, transb=True)
        return [X, W, Y]
