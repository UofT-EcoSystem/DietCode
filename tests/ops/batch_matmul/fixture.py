import tvm
from tvm import auto_scheduler, te, topi

import numpy as np

from shared import CUDATarget, CUDAContext


@auto_scheduler.register_workload
def BatchMatmulNT(B, M, K, N):
    X = te.placeholder((B, M, K), name='X')
    W = te.placeholder((B, N, K), name='W')
    Y = topi.nn.batch_matmul(X, W, transpose_b=True)
    return [X, W, Y]

@auto_scheduler.register_workload
def BatchMatmulNN(B, M, K, N):
    X = te.placeholder((B, M, K), name='X')
    W = te.placeholder((B, K, N), name='W')
    Y = topi.nn.batch_matmul(X, W, transpose_b=False)
    return [X, W, Y]


def _cublas_batch_matmul(B, M, K, N, transpose_b):
    X = te.placeholder((B, M, K), name='X')
    W = te.placeholder((B, N, K) if transpose_b else (B, K, N), name='W')
    Y = tvm.contrib.cublas.batch_matmul(X, W, transa=False, transb=transpose_b)
    sched = te.create_schedule(Y.op)
    vendor_kernel = tvm.build(sched, [X, W, Y], CUDATarget)
    return vendor_kernel, Y


class cuBLASBatchMatmulNTFixture:
    """
    cuBLAS Dense Fixture

    This fixture performs the computation
        
        BatchMatmul([B, M, K], [B, N, K])
        
    and is using the NVIDIA cuBLAS library under the hood. It is used for
    checking the correctness and testing the performance of DietCode.
    """
    __slots__ = 'B', 'M', 'K', 'N', 'X_np', 'W_np', 'Y', 'Y_np'

    def __init__(self, B, M, K, N):
        self.B, self.M, self.K, self.N = B, M, K, N
        self.X_np = np.random.uniform(-0.1, 0.1, size=(B, M, K)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(B, N, K)).astype(np.float32)

        cublas_kernel, self.Y = \
                _cublas_batch_matmul(B=B, M=M, K=K, N=N, transpose_b=True)

        module_data = self.module_data()
        cublas_kernel(*module_data)
        self.Y_np = module_data[-1].asnumpy()

    def module_data(self):
        return [tvm.nd.array(self.X_np, device=CUDAContext),
                tvm.nd.array(self.W_np, device=CUDAContext),
                tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                      dtype=np.float32), device=CUDAContext)]


class cuBLASBatchMatmulNNFixture:
    __slots__ = 'B', 'M', 'K', 'N', 'cublas_kernel', 'X_np', 'W_np', 'Y', 'Y_np'

    def __init__(self, B, M, K, N):
        self.B, self.M, self.K, self.N = B, M, K, N
        self.X_np = np.random.uniform(-0.1, 0.1, size=(B, M, K)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(B, K, N)).astype(np.float32)

        self.cublas_kernel, self.Y = \
                _cublas_batch_matmul(B=B, M=M, K=K, N=N, transpose_b=False)

        module_data = self.module_data()
        self.cublas_kernel(*module_data)
        self.Y_np_expected = module_data[-1].asnumpy()

    def module_data(self):
        return [tvm.nd.array(self.X_np, device=CUDAContext),
                tvm.nd.array(self.W_np, device=CUDAContext),
                tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                      dtype=np.float32), device=CUDAContext)]
