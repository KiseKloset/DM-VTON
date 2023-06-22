"""
Residual UNet Skip Connection
Edit from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""

import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.common.resnet import ResidualBlock


"""
Defines the Unet submodule with skip connection.
X -------------------identity----------------------
|-- downsampling -- |submodule| -- upsampling --|
"""
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(
        self, 
        inner_nc: int,
        outer_nc: int,  
        in_channels: int = None,
        outermost: bool = False, 
        innermost: bool = False, 
        use_dropout: bool = False,
        submodule: Optional[Callable[..., nn.Module]] = None, 
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
    ) -> None:
        """Construct a Unet submodule with skip connections.
        Parameters:
            inner_nc (int)          -- the number of filters in the inner conv layer
            outer_nc (int)          -- the number of filters in the outer conv layer
            in_channels (int)       -- the number of channels in input images/features
            submodule (nn.Module)   -- previously defined submodules
            outermost (bool)        -- if this module is the outermost module
            innermost (bool)        -- if this module is the innermost module
            norm_layer (nn.Module)  -- normalization layer
            use_dropout (bool)      -- if use dropout layers.
        """
        super().__init__()
        self.outermost = outermost
        if norm_layer == None:
            norm_layer == nn.BatchNorm2d
        use_bias = norm_layer == nn.InstanceNorm2d
        in_channels = outer_nc if inner_nc is None else in_channels
        up_scale = 2

        downconv = nn.Conv2d(
                    in_channels, 
                    inner_nc, 
                    kernel_size=3,
                    stride=2, 
                    padding=1, 
                    bias=use_bias)
        res_downconv = nn.Sequential(
                        ResidualBlock(inner_nc, inner_nc, norm_layer), 
                        ResidualBlock(inner_nc, inner_nc, norm_layer)
                    )
        res_upconv = nn.Sequential(
                        ResidualBlock(outer_nc, outer_nc, norm_layer), 
                        ResidualBlock(outer_nc, outer_nc, norm_layer)
                    )
        downnorm = norm_layer(inner_nc)
        upnorm = norm_layer(outer_nc)
        downact = nn.ReLU(True)
        upact = nn.ReLU(True)

        if outermost:
            down_block = nn.Sequential(downconv, downact, res_downconv)
            up_block = nn.Sequential(
                        nn.Upsample(scale_factor=up_scale, mode='nearest'), 
                        nn.Conv2d(inner_nc * up_scale, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    )
            self.model = nn.Sequential(down_block, submodule, up_block)
        elif innermost:
            down_block = nn.Sequential(downconv, downact, res_downconv)
            up_block = nn.Sequential(
                        nn.Upsample(scale_factor=up_scale, mode='nearest'),
                        nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                        upnorm,
                        upact,
                        res_upconv,
                    )
            self.model = nn.Sequential(down_block, up_block)
        else:
            down_block = nn.Sequential(downconv, downnorm, downact, res_downconv)
            up_block = nn.Sequential(
                        nn.Upsample(scale_factor=up_scale, mode='nearest'),
                        nn.Conv2d(inner_nc*up_scale, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                        upnorm,
                        upact,
                        res_upconv,
                    )
            if use_dropout:
                self.model = nn.Sequential(down_block, submodule, up_block, nn.Dropout(0.5))
            else:
                self.model = nn.Sequential(down_block, submodule, up_block)

    def forward(self, x: Tensor) -> Tensor:
        if self.outermost:
            out = self.model(x)
        else:
            # Concatenate
            out = torch.cat([x, self.model(x)], 1)
            
        return out