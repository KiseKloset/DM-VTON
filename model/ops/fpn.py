from typing import Callable, List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation


# Edit from: https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self, 
        in_channels_list: List[int], 
        out_channels: int = 256,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.in_channels_list = in_channels_list
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block_module = Conv2dNormActivation(
                in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=None
            )
            layer_block_module = Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=None
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        last_inner = self.inner_blocks[-1](x[-1], -1)
        out = []
        out.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            feat_shape = inner_lateral.shape[-2:]
            # 2x scale up
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            out.insert(0, self.layer_blocks[idx](last_inner))

        return out