"""
Residual block V2: https://paperswithcode.com/paper/identity-mappings-in-deep-residual-networks
Edit from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor


# 3x3 convolution
def conv3x3(
    in_channels: int, 
    out_channels: int, 
    stride: int = 1, 
    padding: int = 1, 
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(out)
        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)

        return out


class ResidualBlockV2(ResidualBlock):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            in_channels, 
            out_channels, 
            stride,
            norm_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity

        return out

