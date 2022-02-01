import torch
import tvm

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
CUDATarget = tvm.target.Target('cuda')
