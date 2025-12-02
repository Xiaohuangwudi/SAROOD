import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .base_preprocessor import BasePreprocessor
from .transform import Convert


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config):
        super(TestStandardPreProcessor, self).__init__(config)
        self.transform = tvs_trans.Compose([
            tvs_trans.Resize((64, 64)),
            tvs_trans.ToTensor(),
        ])


