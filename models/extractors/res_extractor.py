from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor
from base_extractor import BaseExtractorNetwork
from models.common.fpn import FeaturePyramidNetwork
from models.common.resnet import ResidualBlockV2

from . import EXTRACT_REGISTRY


class DownSample(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        stride: int = 2,
        padding: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.bn = norm_layer(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)

        return out
    

class ResNetFeatureEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels_list: list[int],
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.out_channels_list = [in_channels] + out_channels_list
        for i in range(len(self.out_channels_list) - 1):
            encoder = nn.Sequential(
                        DownSample(self.out_channels_list[i], self.out_channels_list[i+1]),
                        ResidualBlockV2(self.out_channels_list[i+1], self.out_channels_list[i+1]),
                        ResidualBlockV2(self.out_channels_list[i+1], self.out_channels_list[i+1])
                    )
            
            self.encoders.append(encoder)

    def forward(self, x: Tensor) -> list[Tensor]:
        out = []
        for encoder in self.encoders:
            out.append(encoder(x))

        return out
    

@EXTRACT_REGISTRY.register()
class ResNetExtractor(BaseExtractorNetwork):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_channels_list: list[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        freeze: bool = False
    ) -> None:
        super().__init__()
        self.feature_encoder = ResNetFeatureEncoder(
            in_channels=in_channels, 
            out_channels_list=feature_channels_list
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=feature_channels_list, 
            out_channels=out_channels, 
            norm_layer=norm_layer
        )

        if freeze:
            self.freeze()
        
    def forward(self, x: Tensor) -> list[Tensor]:
        return self.fpn(self.feature_encoder(x))

