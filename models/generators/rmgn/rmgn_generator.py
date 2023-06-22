import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
    
from models.common.factory import get_act, get_norm
from models.common.conv_norm_activation import CustomeConv2dNormActivation, CustomeConvTranspose2dNormActivation
from models.pf.rmgn.base_network import BaseNetwork
from models.pf.rmgn.rmff import RMFFResNetwork


# TODO: Try AttrDilatedEncoder
class RMGNGenerator(BaseNetwork):
    def __init__(
        self, 
        in_channels_person = 3,
        in_channels_cloth = 4,
        out_channels: int = 4,
        num_features: int = 64,
        SR_scale: int = 1,
        multilevel: bool = False, 
        predmask: bool = True,
        aei_encoder_head: bool = False
    ):
        super().__init__()
        num_head = int(np.log2(SR_scale)) + 1 if aei_encoder_head or SR_scale > 1 else 0
        
        self.person_encoder = AttrEncoder(num_features=num_features, in_channels=in_channels_person, num_head=num_head)
        self.cloth_encoder = AttrEncoder(num_features=num_features, in_nc=in_channels_cloth, num_head=num_head)
        self.generator = RMFFResNetwork(
                            num_features=num_features, 
                            out_channels=out_channels, 
                            SR_scale=SR_scale, 
                            multilevel=multilevel, 
                            predmask=predmask
                        )
        
        self.init_weights()

    def forward(
        self, 
        person: Tensor, 
        cloth: Tensor,
    ) -> tuple:
        person_attr_list = self.person_encoder(person)
        cloth_attr_list = self.get_ref_attr(cloth)
        out, out_L1, out_L2, M_list = self.generator(person_attr_list, cloth_attr_list)
        return out, out_L1, out_L2, M_list


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


def get_head(
    in_channels: int, 
    out_channels: int, 
    num_layers: int = 1, 
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d, 
    act_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
    act_params: Optional[dict] = {}, 
) -> nn.Module:
    assert num_layers >= 1
    layer = CustomeConv2dNormActivation(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                norm_layer=norm_layer,
                activation_layer=act_layer,
                activation_params=act_params,
            )
    head = [layer]

    for _ in range(num_layers - 1):
        layer = CustomeConv2dNormActivation(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    norm_layer=norm_layer,
                    activation_layer=act_layer,
                    activation_params=act_params,
                )
        head.append(layer)

    head = nn.Sequential(*head)

    return head


class Conv4x4(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dilation: int = 1, 
        padding: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d, 
        act_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
        act_params: Optional[dict] = {}, 
    ) -> None:
        super().__init__()
        self.conv4x4_block = CustomeConv2dNormActivation(
                                in_channels,
                                out_channels,
                                kernel_size=4,
                                stride=2,
                                padding=padding,
                                dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=act_layer,
                                activation_params=act_params,
                            )
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv4x4_block(x)
        return out


class Deconv4x4(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dilation: int = 1, 
        padding: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d, 
        act_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
        act_params: Optional[dict] = {}, 
    ):
        super().__init__()
        self.deconv4x4_block = CustomeConvTranspose2dNormActivation(
                                in_channels,
                                out_channels,
                                kernel_size=4,
                                stride=2,
                                padding=padding,
                                dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=act_layer,
                                activation_params=act_params,
                            )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        out = self.deconv_block(x)
        out = torch.cat((out, skip), dim=1)
        return out
        

class AttrEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,
        num_features: int = 64,  
        num_head: int = 0,
        norm: str = 'IN', 
        act: str = 'lrelu',
        act_params: dict = {},
    ) -> None:
        super().__init__()
        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        if num_head > 0:
            head_conv = get_head(
                            in_channels, 
                            num_features // 2,  
                            num_head, 
                            norm=norm,
                            act=act, 
                            act_params=act_params,
                        )
            conv4x4 = Conv4x4(num_features // 2, num_features // 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
            self.conv1 = nn.Sequential(head_conv, conv4x4)
        else:
            self.conv1 = Conv4x4(in_channels, num_features // 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
        
        self.conv2 = Conv4x4(num_features // 2, num_features * 1, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
        self.conv3 = Conv4x4(num_features * 1, num_features * 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
        self.conv4 = Conv4x4(num_features * 2, num_features * 4, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
        self.conv5 = Conv4x4(num_features * 4, num_features * 8, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)

        self.deconv1 = Deconv4x4(num_features * 8, num_features * 4, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
        self.deconv2 = Deconv4x4(num_features * 8, num_features * 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params) 
        self.deconv3 = Deconv4x4(num_features * 4, num_features * 1, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
        self.deconv4 = Deconv4x4(num_features * 2, num_features // 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  

    def forward(self, x: Tensor) -> tuple[Tensor]:
        feat1 = self.conv1(x)  # num_features//2, h/2, w/2
        feat2 = self.conv2(feat1)  # num_features, h/4, w/4
        feat3 = self.conv3(feat2)  # num_features*2, h/8, w/8
        feat4 = self.conv4(feat3)  # num_features*4, h/16, w/16
        feat5 = self.conv5(feat4)  # num_features*8, h/32, w/32

        attr1 = self.deconv1(feat5, feat4)
        # 512x8x8 -> 256x16x16 -> 512x16x16
        attr2 = self.deconv2(attr1, feat3)
        # 512x16x16 -> 128x32x32 -> 256x32x32
        attr3 = self.deconv3(attr2, feat2)
        # 256x32x32 -> 64x64x64 -> 128x64x64
        attr4 = self.deconv4(attr3, feat1)
        # 128x64x64 -> 32x128x128 -> 64x128x128
        attr5 = F.interpolate(attr4, scale_factor=2, mode='bilinear', align_corners=True)
        # 64x128x128 -> 64x256x256
        return attr1, attr2, attr3, attr4, attr5


class AttrDilatedEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        num_features: int = 64, 
        num_head: int = 0,
        norm: str = 'IN', 
        act: str = 'lrelu',
        act_params: dict = {},
    ):
        super().__init__()
        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        if num_head > 0:
            head_conv = get_head(in_channels, num_features // 2, num_head)
            conv4x4 = Conv4x4(num_features // 2, num_features // 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
            self.conv1 = nn.Sequential(head_conv, conv4x4)
        else:
            self.conv1 = Conv4x4(in_channels, num_features // 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)

        self.conv2 = Conv4x4(num_features // 2, num_features, dilation=8, padding=get_pad(256, 4, 2, 8) + 1, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
        self.conv3 = Conv4x4(num_features, num_features * 2, dilation=4, padding=get_pad(128, 4, 2, 4) + 1, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params) 
        self.conv4 = Conv4x4(num_features * 2, num_features * 4, dilation=2, padding=get_pad(64, 4, 2, 2)  + 1, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
        self.conv5 = Conv4x4(num_features * 4, num_features * 8, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
        
        self.deconv1 = Deconv4x4(num_features * 8, num_features * 4, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
        self.deconv2 = Deconv4x4(num_features * 8, num_features * 2, dilation=3, padding=get_pad(64, 4, 2, 2) + 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)
        self.deconv3 = Deconv4x4(num_features * 4, num_features, dilation=5, padding=get_pad(128, 4, 2, 4) + 2, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params) 
        self.deconv4 = Deconv4x4(num_features * 2, num_features // 2, dilation=1, padding=get_pad(256, 4, 2, 8) - 1, norm_layer=norm_layer, activation_layer=act_layer, activation_params=act_params)  
    
    def forward(self, x: Tensor) -> Tensor:
        feat1 = self.conv1(x)  # num_features//2, h/2, w/2
        feat2 = self.conv2(feat1)  # num_features, h/4, w/4
        feat3 = self.conv3(feat2)  # num_features*2, h/8, w/8
        feat4 = self.conv4(feat3)  # num_features*4, h/16, w/16
        feat5 = self.conv5(feat4)  # num_features*8, h/32, w/32
        
        attr1 = self.deconv1(feat5, feat4)
        # 512x8x8 -> 256x16x16 -> 512x16x16
        attr2 = self.deconv2(attr1, feat3)
        # 512x16x16 -> 128x32x32 -> 256x32x32
        attr3 = self.deconv3(attr2, feat2)
        # 256x32x32 -> 64x64x64 -> 128x64x64
        attr4 = self.deconv4(attr3, feat1)
        # 128x64x64 -> 32x128x128 -> 64x128x128
        attr5 = F.interpolate(attr4, scale_factor=2, mode='bilinear', align_corners=True)
        # 64x128x128 -> 64x256x256
        return attr1, attr2, attr3, attr4, attr5


# def conv3x3(in_c, out_c, stride=2, norm=nn.InstanceNorm2d):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=stride, padding=1, bias=False),
#         norm(out_c),
#         nn.LeakyReLU(0.1, inplace=True)
#     )
#
#
# class AttrEncoderV2(nn.Module):
#     def __init__(self, num_features=64, in_channels=3):
#         super(AttrEncoderV2, self).__init__()
#         self.conv_head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
#         self.conv1 = conv3x3(num_features, num_features, stride=1)  # num_features * h * w
#         self.conv2 = conv3x3(num_features, num_features, stride=2)  # num_features * h/2 * w/2
#         self.conv3 = conv3x3(num_features, num_features*2, stride=2)  # 2num_features * h/4 * w/4
#         self.conv4 = conv3x3(num_features*2, num_features*4, stride=2)  # 4num_features * h/8 * w/8
#         self.conv5 = conv3x3(num_features*4, num_features*8, stride=2)  # 8num_features * h/16 * w/16

#     def forward(self, x):
#         feat = self.conv_head(x)
#         feat1 = self.conv1(feat)
#         feat2 = self.conv2(feat1)
#         feat3 = self.conv3(feat2)
#         feat4 = self.conv4(feat3)
#         feat5 = self.conv5(feat4)
#         return feat5, feat4, feat3, feat2, feat1


