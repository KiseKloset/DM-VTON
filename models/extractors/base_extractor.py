import torch.nn as nn


class BaseExtractorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze(self):
        for p in self.extractor.parameters():
            p.requires_grad = False