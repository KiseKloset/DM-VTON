import torch.nn as nn
from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.modeling import build_model


class DensePose(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        cfg = get_cfg()
        add_densepose_config(cfg)
        self.model = build_model(cfg)

    def forward(self, image):
        self.model([image])
