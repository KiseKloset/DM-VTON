import torch.nn as nn

from ClothFlow.models.networks import FlowEstimator

class ClothFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FlowEstimator()

    def forward(self, cloth, mask, parse_cloth):
        return self.model(cloth, mask, parse_cloth)