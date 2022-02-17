import tvm

import torch
import transformers

import logging
import os

logger = logging.getLogger(__name__)

from ..shared.fixture import ModelFixture


# PyTorch
torch.backends.cudnn.benchmark = True
# check the availability of cuDNN
if "CuDNN" in torch.__config__.show():
    logger.info("cuDNN is enabled in pytorch")
else:
    logger.warning("cuDNN is not enabled in pytorch!")


class PyTorchBERTFixture(ModelFixture):
    def __init__(self, B, T):
        self.input_data_torch = torch.randint(30000, (B, T), dtype=torch.int64)
        self.input_data_np = self.input_data_torch.detach().numpy()

        model_class = transformers.BertModel
        model = model_class.from_pretrained(
                    os.path.dirname(os.path.abspath(__file__)) + '/bert_base_uncased'
                )
        self.model = model.eval()
        self.scripted_model = torch.jit.trace(model, [self.input_data_torch], strict=False)

    @property
    def name(self):
        return 'BERT'

    @property
    def input_name(self):
        return 'input_ids'
