import torch
from torch import nn


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class Swish(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)
