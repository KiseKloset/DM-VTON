import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from base.base_model import BaseModel
from model.ops.fpn import FeaturePyramidNetwork
from model.ops.resnet import ResidualBlockV2
from model.pb.flow_style.style import StyledFConvBlock, StyledConvBlock


def apply_offset(offset) -> Tensor:
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))] 

    return torch.stack(grid_list, dim=-1)


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


class FeatureEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels_list: List[int],
    ) -> None:
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
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

    def forward(self, x: Tensor) -> List[Tensor]:
        out = []
        for encoder in self.encoders:
            out.append(encoder(x))

        return out


# TODO: Check B is batchsize (forward)
# TODO: Check edge == None
# Class Appearance Flow Estimation Network
class AFEN(nn.Module):
    def __init__(
        self, 
        n_pyramid: int, 
        fpn_dim: int = 256,
        padding_layer: Optional[Callable[..., nn.Module]] = None,
        modulated_conv: bool = True,
        demodulate: bool = True,
        norm_mlp: bool = False,
        act: str = 'lrelu',
    ) -> None:
        super().__init__()

        self.refine_net = nn.ModuleList()
        self.styled_net = nn.ModuleList()
        self.styled_f_net = nn.ModuleList()

        for _ in range(n_pyramid):
            refine_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            styled_block = StyledConvBlock(
                            in_channels=256, 
                            out_channels=49, 
                            style_dim=256,
                            padding_layer=padding_layer, 
                            modulated_conv=modulated_conv,
                            demodulate=demodulate,
                            norm_affine_output=norm_mlp,
                            act=act,
                        )

            styled_f_block = StyledFConvBlock(
                                in_channels=49, 
                                out_channels=2, 
                                style_dim=256,
                                padding_layer=padding_layer,
                                modulated_conv=modulated_conv,
                                demodulate=demodulate,
                                norm_affine_output=norm_mlp,
                                act=act,
                            )

            self.refine_net.append(refine_block)
            self.styled_net.append(styled_block)
            self.styled_f_net.append(styled_f_block)

        self.cond_style = torch.nn.Sequential(
                            torch.nn.Conv2d(256, 128, kernel_size=(8,6), stride=1, padding=0), 
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )
        self.img_style = torch.nn.Sequential(
                            torch.nn.Conv2d(256, 128, kernel_size=(8,6), stride=1, padding=0), 
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

    def create_filter(self):
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]] 
        
        return filter_x, filter_y, filter_diag1, filter_diag2

    def create_weight(self):
        filter_x, filter_y, filter_diag1, filter_diag2 = self.create_filter()
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        weight_array = torch.cuda.FloatTensor(weight_array).permute(3,2,0,1)
        return weight_array

    def forward(
        self, 
        x: Tensor, 
        x_edge: Tensor, 
        x_warps: List[Tensor], 
        x_conds: List[Tensor], 
        warp_feature=True
    ) -> Tuple:
        last_flow, last_flow_all = None, []
        x_all, x_edge_all, delta_list, delta_x_all, delta_y_all, cond_fea_all = [], [], [], [], [], []

        weight_array = self.create_weight()
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        B = x_conds[len(x_warps)-1].shape[0]

        cond_style = self.cond_style(x_conds[len(x_warps) - 1]).view(B, -1)
        image_style = self.img_style(x_warps[len(x_warps) - 1]).view(B, -1)
        style = torch.cat([cond_style, image_style], 1)

        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]
            cond_fea_all.append(x_cond)

            if last_flow is not None and warp_feature:
                x_warp_after = F.grid_sample(
                                x_warp, 
                                last_flow.detach().permute(0, 2, 3, 1),
                                mode='bilinear', 
                                padding_mode='border',
                            )
            else:
                x_warp_after = x_warp

            stylemap = self.styled_net[i](x_warp_after, style)
            flow = self.styled_f_net[i](stylemap, style)
            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp = F.grid_sample(x_warp, flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            concat = torch.cat([x_warp,x_cond],1)
            flow = self.refine_net[i](concat)
            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)
            cur_x = F.interpolate(x, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            x_all.append(cur_x_warp)
            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)
            flow_x,flow_y = torch.split(last_flow,1,dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y,self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

        x_warp = F.grid_sample(x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

        return x_warp, last_flow, cond_fea_all, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all


# TODO: Check lr
# Class Appearance Flow Warping Module 
class AFWM(BaseModel):
    def __init__(
        self, 
        cond_in_channels: int,
        filters: List[int] = [64, 128, 256, 256, 256],
    ) -> None:
        super().__init__()
        self.image_features = FeatureEncoder(in_channels=3, out_channels_list=filters) 
        self.cond_features = FeatureEncoder(in_channels=cond_in_channels, out_channels_list=filters)
        self.image_FPN = FeaturePyramidNetwork(in_channels_list=filters, out_channels=256)
        self.cond_FPN = FeaturePyramidNetwork(in_channels_list=filters, out_channels=256)
        self.aflow_net = AFEN(len(filters))
        # self.old_lr = opt.lr
        # self.old_lr_warp = opt.lr*0.2

    def forward(
        self, 
        cond_input: Tensor, 
        image_input: Tensor, 
        image_edge: Tensor,
    ) -> Tuple:
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))
        image_pyramids = self.image_FPN(self.image_features(image_input))

        x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all \
            = self.aflow_net(image_input, image_edge, image_pyramids, cond_pyramids)

        return x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all


    # def update_learning_rate(self,optimizer):
    #     lrd = opt.lr / opt.niter_decay
    #     lr = self.old_lr - lrd
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     if opt.verbose:
    #         print('update learning rate: %f -> %f' % (self.old_lr, lr))
    #     self.old_lr = lr

    # def update_learning_rate_warp(self,optimizer):
    #     lrd = 0.2 * opt.lr / opt.niter_decay
    #     lr = self.old_lr_warp - lrd
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     if opt.verbose:
    #         print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
    #     self.old_lr_warp = lr