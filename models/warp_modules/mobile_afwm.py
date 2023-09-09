import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.common.correlation import FunctionCorrelation  # Custom cost volume layer
from models.extractors.mobile_fpn import MobileNetV2_dynamicFPN


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


class AFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256, align_corners=True):
        super().__init__()
        self.netMain = []
        self.netRefine = []
        self.align_corners = align_corners

        for i in range(num_pyramid):
            netMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1
                ),
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
            self.netMain.append(netMain_layer)
            self.netRefine.append(netRefine_layer)

        self.netMain = nn.ModuleList(self.netMain)
        self.netRefine = nn.ModuleList(self.netRefine)

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

                tenCorrelation = FunctionCorrelation(
                    tenFirst=x_warp_after, tenSecond=x_cond, intStride=1
                )
                tenCorrelation = F.leaky_relu(
                    input=tenCorrelation, negative_slope=0.1, inplace=False
                )
                flow = self.netMain[i](tenCorrelation)
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

            for i in range(len(x_warps)):
                x_warp = x_warps[len(x_warps) - 1 - i]
                x_cond = x_conds[len(x_warps) - 1 - i]

                if last_flow is not None and warp_feature:
                    # with style_dt[3]:
                    x_warp_after = F.grid_sample(
                        x_warp,
                        last_flow.detach().permute(0, 2, 3, 1),
                        mode='bilinear',
                        padding_mode='border',
                        align_corners=self.align_corners,
                    )
                else:
                    x_warp_after = x_warp

                # with style_dt[0]:
                tenCorrelation = FunctionCorrelation(
                    tenFirst=x_warp_after, tenSecond=x_cond, intStride=1
                )
                tenCorrelation = F.leaky_relu(
                    input=tenCorrelation, negative_slope=0.1, inplace=False
                )
                flow = self.netMain[i](tenCorrelation)

                # with style_dt[5]:
                flow = apply_offset(flow)

                if last_flow is not None:
                    # with style_dt[3]:
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

                # with style_dt[3]:
                x_warp = F.grid_sample(
                    x_warp,
                    flow.permute(0, 2, 3, 1),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners,
                )

                concat = torch.cat([x_warp, x_cond], 1)

                # with style_dt[4]:
                flow = self.netRefine[i](concat)

                # with style_dt[5]:
                flow = apply_offset(flow)

                # with style_dt[3]:
                flow = F.grid_sample(
                    last_flow,
                    flow,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners,
                )

                last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')

            # with style_dt[3]:
            x_warp = F.grid_sample(
                x,
                last_flow.permute(0, 2, 3, 1),
                mode='bilinear',
                padding_mode='border',
                align_corners=self.align_corners,
            )

            return x_warp, last_flow


class MobileAFWM(BaseModel):
    def __init__(self, input_nc, align_corners):
        super().__init__()
        num_filters = [64, 128, 256, 256, 256]
        self.image_mobile = MobileNetV2_dynamicFPN(3)
        self.cond_mobile = MobileNetV2_dynamicFPN(input_nc)

        self.aflow_net = AFlowNet(len(num_filters), align_corners=align_corners)

    def forward(self, cond_input, image_input, image_edge=None, phase='train'):
        assert phase in ['train', 'test'], f'ERROR: phase can only be train or test, not {phase}'

        cond_pyramids = self.cond_mobile(cond_input)
        image_pyramids = self.image_mobile(image_input)

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
            x_warp, last_flow = self.aflow_net(
                image_input, image_pyramids, cond_pyramids, phase=phase
            )
            return x_warp, last_flow
