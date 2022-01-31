import torch
import tvm

import ctypes
import logging
import numpy as np
import os
import pytest
import random


logger = logging.getLogger(__name__)

use_tvm_base = (int(os.getenv('USE_TVM_BASE', '0')) == 1)

if use_tvm_base:
    logger.info("Using TVM base branch")
else:
    logger.info("Using TVM dev branch")
    os.environ["DIETCODE_SCHED_OPT"] = '1'
    os.environ["DIETCODE_PRINT_LAUNCHBOUND"] = '1'
    logger.info("!!! Enabling all the code generation optimizations !!!")

# decorators used by pytest's
tvm_base_decor = pytest.mark.skipif(not use_tvm_base,
                                    reason="TVM base branch must be set: USE_TVM_BASE=1")
tvm_dev_decor  = pytest.mark.skipif(use_tvm_base, reason="TVM dev branch must be set: USE_TVM_BASE=0")

rand_seed = 0

random.seed(rand_seed)
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

CUDAContext = tvm.cuda()

libcuda = ctypes.CDLL('libcuda.so')
# needed to retrieve the device name
assert libcuda.cuInit(0) == 0, "Failed to initialize CUDA driver APIs"

cuda_device_name = CUDAContext.device_name

if cuda_device_name == 'Tesla T4':
    cuda_target_abbrev = 't4'
    cuda_target = 'nvidia/nvidia-t4'
elif cuda_device_name == 'NVIDIA GeForce RTX 2080 Ti':
    cuda_target_abbrev = 'rtx_2080_ti'
    cuda_target = 'nvidia/geforce-rtx-2080-ti'
elif cuda_device_name == 'NVIDIA GeForce RTX 3090':
    cuda_target_abbrev = 'rtx_3090'
    cuda_target = 'nvidia/geforce-rtx-3090'
elif cuda_device_name == 'Tesla V100-SXM2-16GB':
    cuda_target_abbrev = 'v100'
    cuda_target = 'nvidia/nvidia-v100'
else:
    assert False, "Unknown CUDA device name={}".format(cuda_device_name)

CUDATarget = tvm.target.Target(cuda_target)
