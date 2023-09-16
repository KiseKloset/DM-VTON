from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.extractors.res_fpn import FeatureEncoder, RefinePyramid


def apply_offset(offset):
    # Offset(B x 2 x H x W)
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)  # Two cordinates grid tensor

    # Apply offset
    grid_list = [
        grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)
    ]  # 2 tensor (B x H x W): [0: size - 1]

    # Normalize
    grid_list = [
        grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))
    ]  # 2 tensor (B x H x W): [-1, 1]

    return torch.stack(grid_list, dim=-1)


# backbone
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        fin,
        fout,
        kernel_size,
        padding_type='zero',
        upsample=False,
        downsample=False,
        latent_dim=512,
        normalize_mlp=False,
    ):
        super().__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        padding_size = kernel_size // 2

        if kernel_size == 1:
            self.demodulate = False
        else:
            self.demodulate = True

        self.weight = nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))

        # TODO: Add PixelNorm
        self.mlp_class_std = EqualLinear(latent_dim, fin)
        # if normalize_mlp:
        #     self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, fin), PixelNorm())
        # else:
        #     self.mlp_class_std = EqualLinear(latent_dim, fin)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(
            1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        s = self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demodulate:
            d = torch.rsqrt((weight**2).sum(4).sum(3).sum(2) + 1e-5).view(
                -1, self.out_channels, 1, 1, 1
            )
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        batch, _, height, width = input.shape

        input = input.view(1, -1, height, width)
        input = self.padding(input)
        out = (
            F.conv2d(input, weight, groups=batch).view(batch, self.out_channels, height, width)
            + self.bias
        )

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        fin,
        fout,
        latent_dim=256,
        padding='zero',
        actvn='lrelu',
        normalize_affine_output=False,
        modulated_conv=False,
    ):
        super().__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        # TODO: Add EqualConv2d
        conv2d = ModulatedConv2d
        # if modulated_conv:
        #     conv2d = ModulatedConv2d
        # else:
        #     conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2, True)

        if self.modulated_conv:
            self.conv0 = conv2d(
                fin,
                fout,
                kernel_size=3,
                padding_type=padding,
                upsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output,
            )
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)

            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(
                fout,
                fout,
                kernel_size=3,
                padding_type=padding,
                downsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output,
            )
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input, latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out, latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain

        return out


class Styled_F_ConvBlock(nn.Module):
    def __init__(
        self,
        fin,
        fout,
        latent_dim=256,
        padding='zero',
        actvn='lrelu',
        normalize_affine_output=False,
        modulated_conv=False,
    ):
        super().__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        # TODO: Add EqualConv2d
        conv2d = ModulatedConv2d
        # if modulated_conv:
        #     conv2d = ModulatedConv2d
        # else:
        #     conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2, True)

        if self.modulated_conv:
            self.conv0 = conv2d(
                fin,
                128,
                kernel_size=3,
                padding_type=padding,
                upsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output,
            )
        else:
            conv0 = conv2d(fin, 128, kernel_size=3)

            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(
                128,
                fout,
                kernel_size=3,
                padding_type=padding,
                downsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output,
            )
        else:
            conv1 = conv2d(128, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        # self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input, latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out, latent)
        else:
            out = self.conv1(out)

        # out = self.actvn1(out) * self.actvn_gain

        return out


class AFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256, align_corners=True):
        super().__init__()

        padding_type = 'zero'
        actvn = 'lrelu'
        normalize_mlp = False
        modulated_conv = True
        self.align_corners = align_corners

        self.netRefine = []

        self.netStyle = []

        self.netF = []

        for i in range(num_pyramid):
            netRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(
                    in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            )

            style_block = StyledConvBlock(
                256,
                49,
                latent_dim=256,
                padding=padding_type,
                actvn=actvn,
                normalize_affine_output=normalize_mlp,
                modulated_conv=modulated_conv,
            )

            style_F_block = Styled_F_ConvBlock(
                49,
                2,
                latent_dim=256,
                padding=padding_type,
                actvn=actvn,
                normalize_affine_output=normalize_mlp,
                modulated_conv=modulated_conv,
            )

            self.netRefine.append(netRefine_layer)
            self.netStyle.append(style_block)
            self.netF.append(style_F_block)

        self.netRefine = nn.ModuleList(self.netRefine)
        self.netStyle = nn.ModuleList(self.netStyle)
        self.netF = nn.ModuleList(self.netF)

        self.cond_style = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(8, 6), stride=1, padding=0),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

        self.image_style = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(8, 6), stride=1, padding=0),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

    def forward(self, x, x_warps, x_conds, x_edge=None, warp_feature=True, phase='train'):
        if phase == 'train':
            last_flow = None
            last_flow_all = []
            delta_list = []
            x_all = []
            x_edge_all = []
            cond_fea_all = []
            warp_fea_all = []
            delta_x_all = []
            delta_y_all = []
            filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
            filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
            filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
            filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
            weight_array = np.ones([3, 3, 1, 4])
            weight_array[:, :, 0, 0] = filter_x
            weight_array[:, :, 0, 1] = filter_y
            weight_array[:, :, 0, 2] = filter_diag1
            weight_array[:, :, 0, 3] = filter_diag2

            weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
            self.weight = nn.Parameter(data=weight_array, requires_grad=False)

            # import ipdb; ipdb.set_trace()

            B = x_conds[len(x_warps) - 1].shape[0]

            cond_style = self.cond_style(x_conds[len(x_warps) - 1]).view(B, -1)
            image_style = self.image_style(x_warps[len(x_warps) - 1]).view(B, -1)
            style = torch.cat([cond_style, image_style], 1)

            for i in range(len(x_warps)):
                x_warp = x_warps[len(x_warps) - 1 - i]
                x_cond = x_conds[len(x_warps) - 1 - i]
                cond_fea_all.append(x_cond)
                warp_fea_all.append(x_warp)

                if last_flow is not None and warp_feature:
                    x_warp_after = F.grid_sample(
                        x_warp,
                        last_flow.detach().permute(0, 2, 3, 1),
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=self.align_corners,
                    )
                else:
                    x_warp_after = x_warp

                flow = self.netStyle[i](x_warp_after, style)
                flow = self.netF[i](flow, style)
                delta_list.append(flow)
                flow = apply_offset(flow)
                if last_flow is not None:
                    flow = F.grid_sample(
                        last_flow,
                        flow,
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=self.align_corners,
                    )
                else:
                    flow = flow.permute(0, 3, 1, 2)

                last_flow = flow
                x_warp = F.grid_sample(
                    x_warp,
                    flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners,
                )
                concat = torch.cat([x_warp, x_cond], 1)
                flow = self.netRefine[i](concat)
                delta_list.append(flow)
                flow = apply_offset(flow)
                flow = F.grid_sample(
                    last_flow,
                    flow,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners,
                )

                last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
                last_flow_all.append(last_flow)
                cur_x = F.interpolate(
                    x, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear'
                )
                cur_x_warp = F.grid_sample(
                    cur_x,
                    last_flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners,
                )
                x_all.append(cur_x_warp)
                cur_x_edge = F.interpolate(
                    x_edge, scale_factor=0.5 ** (len(x_warps) - 1 - i), mode='bilinear'
                )
                cur_x_warp_edge = F.grid_sample(
                    cur_x_edge,
                    last_flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=self.align_corners,
                )
                x_edge_all.append(cur_x_warp_edge)
                flow_x, flow_y = torch.split(last_flow, 1, dim=1)
                delta_x = F.conv2d(flow_x, self.weight)
                delta_y = F.conv2d(flow_y, self.weight)
                delta_x_all.append(delta_x)
                delta_y_all.append(delta_y)

            x_warp = F.grid_sample(
                x,
                last_flow.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border',
                align_corners=self.align_corners,
            )
            return (
                x_warp,
                last_flow,
                cond_fea_all,
                warp_fea_all,
                last_flow_all,
                delta_list,
                x_all,
                x_edge_all,
                delta_x_all,
                delta_y_all,
            )

        elif phase == 'test':
            last_flow = None

            B = x_conds[len(x_warps) - 1].shape[0]

            cond_style = self.cond_style(x_conds[len(x_warps) - 1]).view(B, -1)
            image_style = self.image_style(x_warps[len(x_warps) - 1]).view(B, -1)
            style = torch.cat([cond_style, image_style], 1)

            for i in range(len(x_warps)):
                x_warp = x_warps[len(x_warps) - 1 - i]
                x_cond = x_conds[len(x_warps) - 1 - i]

                if last_flow is not None and warp_feature:
                    x_warp_after = F.grid_sample(
                        x_warp,
                        last_flow.detach().permute(0, 2, 3, 1),
                        mode='bilinear',
                        padding_mode='border',
                    )
                else:
                    x_warp_after = x_warp

                stylemap = self.netStyle[i](x_warp_after, style)

                flow = self.netF[i](stylemap, style)
                flow = apply_offset(flow)
                if last_flow is not None:
                    flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
                else:
                    flow = flow.permute(0, 3, 1, 2)

                last_flow = flow
                x_warp = F.grid_sample(
                    x_warp, flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border'
                )
                concat = torch.cat([x_warp, x_cond], 1)
                flow = self.netRefine[i](concat)
                flow = apply_offset(flow)
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

                last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')

            x_warp = F.grid_sample(
                x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border'
            )
            return x_warp, last_flow


class StyleAFWM(BaseModel):
    def __init__(self, input_nc, align_corners):
        super().__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.image_features = FeatureEncoder(3, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(num_filters)
        self.cond_FPN = RefinePyramid(num_filters)

        self.aflow_net = AFlowNet(len(num_filters), align_corners=align_corners)

    def forward(self, cond_input, image_input, image_edge=None, phase='train'):
        assert phase in ['train', 'test'], f'ERROR: phase can only be train or test, not {phase}'

        # TODO: Refactor to nn.Sequential
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))
        image_pyramids = self.image_FPN(self.image_features(image_input))

        if phase == 'train':
            assert image_edge is not None, 'ERROR: image_edge cannot be None when phase is train'
            (
                x_warp,
                last_flow,
                cond_fea_all,
                warp_fea_all,
                flow_all,
                delta_list,
                x_all,
                x_edge_all,
                delta_x_all,
                delta_y_all,
            ) = self.aflow_net(image_input, image_pyramids, cond_pyramids, image_edge, phase=phase)
            return (
                x_warp,
                last_flow,
                cond_fea_all,
                warp_fea_all,
                flow_all,
                delta_list,
                x_all,
                x_edge_all,
                delta_x_all,
                delta_y_all,
            )
        elif phase == 'test':
            x_warp, last_flow = self.aflow_net(image_input, image_pyramids, cond_pyramids, phase=phase)
            return x_warp, last_flow
