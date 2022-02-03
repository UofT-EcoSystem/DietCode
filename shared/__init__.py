import torch
import tvm

import ctypes
import logging
import numpy as np
import os
import pytest
import random


logger = logging.getLogger(__name__)

use_dietcode = (int(os.getenv('USE_DIETCODE', '0')) == 1)

if use_dietcode:
    # enable all the code generation optimizations
    os.environ["DIETCODE_CODEGEN_OPT"] = '1'

# decorators used for filtering tests
base_decor = pytest.mark.skipif(use_dietcode, reason="Base branch must be set: "
                                                     "source environ/activate_base.sh")
dietcode_decor = pytest.mark.skipif(not use_dietcode,
                                    reason="DietCode branch must be set: "
                                           "source environ/activate_dietcode.sh"
                                    )

rand_seed = 0

random.seed(rand_seed)
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

CUDAContext = tvm.cuda()

libcuda = ctypes.CDLL('libcuda.so')
# needed to retrieve the device name
assert libcuda.cuInit(0) == 0, "Failed to initialize CUDA driver APIs"

CUDATarget = tvm.target.Target('cuda')
