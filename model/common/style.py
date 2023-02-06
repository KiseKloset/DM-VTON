"""
StyleGan
Edit from: 
https://github.com/rosinality/progressive-gan-pytorch/blob/master/model.py
https://github.com/InterDigitalInc/FeatureStyleEncoder/blob/main/pixel2style2pixel/models/stylegan2/model.py
"""

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH



# TODO: Try StyleGAN2
class EqualLR:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

    def compute_weight(self, module: nn.Module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module: nn.Module, name: str):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        bias: bool = True,
    ) -> None:
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)


class EqualConv2d(nn.Module):
    def __init__(
        self,  
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0, 
        bias: bool = True,
    ) -> None:
        super().__init__()

        conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channel=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


class PixelNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.rmath.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# Disable upsample/downsample
# TODO: Enable upsample/downsample
# TODO: Check self.scale
class ModulatedConv2d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,  
        style_dim: int,
        eps: float = 1e-5,
        padding_layer: Optional[Callable[..., nn.Module]] = None,
        demodulate: bool = True, 
        norm_mlp: bool = False,
    ) -> None:
        super().__init__()
        if padding_layer is None:
            padding_layer = nn.ZeroPad2d

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        fan_in = in_channels * kernel_size ** 2
        self.scale = math.sqrt(2 / fan_in)
        self.eps = eps
        padding = kernel_size // 2
        self.pad = padding_layer(padding)

        self.weight = nn.Parameter(
            torch.Tensor(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(
            torch.Tensor(1, out_channels, 1, 1)
        )

        self.demodulate = False if kernel_size==1 else demodulate

        if norm_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(style_dim, in_channels), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(style_dim, in_channels)
        
        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        batch, in_channels, height, width = input.shape
        
        s = self.mlp_class_std(style).view(batch, 1, in_channels, 1, 1) 
        weight = s * self.weight * self.scale # batch x out_channels x in_channels x kernel_size x kernel_size
        # fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        # weight = self.weight * sqrt(2 / fan_in)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps).view(batch, self.out_channels, 1, 1, 1)
            weight = weight * demod
        
        weight = weight.view(-1, in_channels, self.kernel_size, self.kernel_size)

        out = input.view(1, -1, height, width)
        out = self.pad(out)
        out = F.conv2d(out, weight, groups=batch)
        out = out.view(batch, self.out_channels, height, width) + self.bias

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        style_dim: int,  
        padding_layer: Optional[Callable[..., nn.Module]] = None,
        modulated_conv: bool = False,
        demodulate: bool = True,
        norm_affine_output: bool = False, 
        act_layer: str = 'lrelu',
    ) -> None:
        super().__init__()
        if padding_layer is None:
            padding_layer = nn.ZeroPad2d
        if act_layer is None:    
            act_layer = nn.LeakyReLU(0.2, True)
        self.act = act_layer
        self.act_gain = math.sqrt(2) if modulated_conv else 1.0
        self.modulated_conv = modulated_conv
        
        if self.modulated_conv:
            self.conv1 = ModulatedConv2d(
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=3, 
                            style_dim=style_dim,
                            padding_layer=padding_layer,
                            demodulate=demodulate,
                            norm_mlp=norm_affine_output,
                        )
            self.conv2 = ModulatedConv2d(
                            in_channels=out_channels, 
                            out_channels=out_channels, 
                            kernel_size=3, 
                            style_dim=style_dim,
                            padding_layer=padding_layer,
                            demodulate=demodulate,
                            norm_mlp=norm_affine_output,
                        )
            
        else:
            self.conv1 = nn.Sequential(
                            padding_layer(1), 
                            EqualConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
                        )
            self.conv2 = nn.Sequential(
                            padding_layer(1), 
                            EqualConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
                        )
            
    def forward(self, input: Tensor, style: Tensor = None) -> Tensor:
        if self.modulated_conv:
            out = self.conv1(input, style)
            out = self.act(out) * self.act_gain
            self.conv2(out, style)
            out = self.act(out) * self.act_gain
        else:
            out = self.conv1(input)
            out = self.act(out) * self.act_gain
            self.conv2(out)
            out = self.act(out) * self.act_gain

        return out


# Add middle dim of two conv (128) and disable last activation layer
# TODO: Check this class's effect
class StyledFConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        style_dim: int,  
        padding_layer: Optional[Callable[..., nn.Module]] = None,
        modulated_conv: bool = False,
        demodulate: bool = True,
        norm_affine_output: bool = False, 
        act_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if padding_layer is None:
            padding_layer = nn.ZeroPad2d
        if act_layer is None:
            act_layer = nn.LeakyReLU(0.2, True)
        self.act = act_layer
        self.act_gain = math.sqrt(2) if modulated_conv else 1.0
        self.modulated_conv = modulated_conv
        
        if self.modulated_conv:
            self.conv1 = ModulatedConv2d(
                            in_channels=in_channels, 
                            out_channels=128, 
                            kernel_size=3, 
                            style_dim=style_dim,
                            padding_layer=padding_layer,
                            demodulate=demodulate,
                            norm_mlp=norm_affine_output,
                        )
            self.conv2 = ModulatedConv2d(
                            in_channels=128, 
                            out_channels=out_channels, 
                            kernel_size=3, 
                            style_dim=style_dim,
                            padding_layer=padding_layer,
                            demodulate=demodulate,
                            norm_mlp=norm_affine_output,
                        )
            
        else:
            self.conv1 = nn.Sequential(
                            padding_layer(1), 
                            EqualConv2d(in_channels=in_channels, out_channels=128, kernel_size=3),
                        )
            self.conv2 = nn.Sequential(
                            padding_layer(1), 
                            EqualConv2d(in_channels=128, out_channels=out_channels, kernel_size=3),
                        )
    
    def forward(self, input: Tensor, style: Tensor = None) -> Tensor:
        if self.modulated_conv:
            out = self.conv1(input, style)
            out = self.act(out) * self.act_gain
            self.conv2(out, style)
        else:
            out = self.conv1(input)
            out = self.act(out) * self.act_gain
            self.conv2(out)

        return out
    