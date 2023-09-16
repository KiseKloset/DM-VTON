import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from options.train_options import TrainOptions
opt = TrainOptions().parse()



def apply_offset(offset):
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


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))


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
    def __init__(self, fin, fout, kernel_size, padding_type='zero', upsample=False, downsample=False, latent_dim=512, normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        padding_size = kernel_size // 2

        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True

        self.weight = nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))
        #self.conv = F.conv2d

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, fin), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        #self.blur = Blur(fout)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)


        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        s = self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        

        batch,_,height,width = input.shape
        #input = input.view(1,-1,h,w)
        #input = self.padding(input)
        #out = self.conv(input, weight, groups=b).view(b, self.out_channels, h, w) + self.bias

        

        input = input.view(1,-1,height,width)
        input = self.padding(input)
        out = F.conv2d(input, weight, groups=batch).view(batch, self.out_channels, height, width) + self.bias

        return out


class StyledConvBlock(nn.Module):
    def __init__(self, fin, fout, latent_dim=256, padding='zero',
                 actvn='lrelu', normalize_affine_output=False, modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        
        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2,True)


        if self.modulated_conv:
            self.conv0 = conv2d(fin, fout, kernel_size=3, padding_type=padding, upsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)
  
            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(fout, fout, kernel_size=3, padding_type=padding, downsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain

        return out


class Styled_F_ConvBlock(nn.Module):
    def __init__(self, fin, fout, latent_dim=256, padding='zero',
                 actvn='lrelu', normalize_affine_output=False, modulated_conv=False):
        super(Styled_F_ConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        
        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2,True)


        if self.modulated_conv:
            self.conv0 = conv2d(fin, 128, kernel_size=3, padding_type=padding, upsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, 128, kernel_size=3)
  
            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(128, fout, kernel_size=3, padding_type=padding, downsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(128, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        #self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        #out = self.actvn1(out) * self.actvn_gain

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block=  nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)



class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64,128,256,256,256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                         ResBlock(out_chns),
                                         ResBlock(out_chns))
            
            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)


    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features

class RefinePyramid(nn.Module):
    def __init__(self, chns=[64,128,256,256,256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive 
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x
        
        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class AFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256, align_corners=True):
        super(AFlowNet, self).__init__()

        padding_type='zero'
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
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            style_block = StyledConvBlock(256, 49, latent_dim=256,
                                         padding=padding_type, actvn=actvn,
                                         normalize_affine_output=normalize_mlp,
                                         modulated_conv=modulated_conv)

            style_F_block = Styled_F_ConvBlock(49, 2, latent_dim=256,
                                              padding=padding_type, actvn=actvn,
                                              normalize_affine_output=normalize_mlp,
                                              modulated_conv=modulated_conv)

            self.netRefine.append(netRefine_layer)
            self.netStyle.append(style_block)
            self.netF.append(style_F_block)

        self.netRefine = nn.ModuleList(self.netRefine)
        self.netStyle = nn.ModuleList(self.netStyle)
        self.netF = nn.ModuleList(self.netF)

        self.cond_style = torch.nn.Sequential(torch.nn.Conv2d(256, 128, kernel_size=(8,6), stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))

        self.image_style = torch.nn.Sequential(torch.nn.Conv2d(256, 128, kernel_size=(8,6), stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))


    def forward(self, x, x_edge, x_warps, x_conds, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        cond_fea_all = []
        delta_x_all = []
        delta_y_all = []
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
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3,2,0,1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        #import ipdb; ipdb.set_trace()

        B = x_conds[len(x_warps)-1].shape[0]

        cond_style = self.cond_style(x_conds[len(x_warps) - 1]).view(B,-1)
        image_style = self.image_style(x_warps[len(x_warps) - 1]).view(B,-1)
        style = torch.cat([cond_style, image_style], 1)

        for i in range(len(x_warps)):
              x_warp = x_warps[len(x_warps) - 1 - i]
              x_cond = x_conds[len(x_warps) - 1 - i]
              cond_fea_all.append(x_cond)

              if last_flow is not None and warp_feature:
                  x_warp_after = F.grid_sample(x_warp, last_flow.detach().permute(0, 2, 3, 1),
                       mode='bilinear', padding_mode='border', align_corners=self.align_corners)
              else:
                  x_warp_after = x_warp

              flow = self.netStyle[i](x_warp_after, style)
              flow = self.netF[i](flow, style)
              delta_list.append(flow)
              flow = apply_offset(flow)
              if last_flow is not None:
                  flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border', align_corners=self.align_corners)
              else:
                  flow = flow.permute(0, 3, 1, 2)

              last_flow = flow
              x_warp = F.grid_sample(x_warp, flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border', align_corners=self.align_corners)
              concat = torch.cat([x_warp,x_cond],1)
              flow = self.netRefine[i](concat)
              delta_list.append(flow)
              flow = apply_offset(flow)
              flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border', align_corners=self.align_corners)

              last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
              last_flow_all.append(last_flow)
              cur_x = F.interpolate(x, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
              cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border', align_corners=self.align_corners)
              x_all.append(cur_x_warp)
              cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
              cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros', align_corners=self.align_corners)
              x_edge_all.append(cur_x_warp_edge)
              flow_x,flow_y = torch.split(last_flow,1,dim=1)
              delta_x = F.conv2d(flow_x, self.weight)
              delta_y = F.conv2d(flow_y,self.weight)
              delta_x_all.append(delta_x)
              delta_y_all.append(delta_y)

        x_warp = F.grid_sample(x, last_flow.permute(0, 2, 3, 1),
                     mode='bilinear', padding_mode='border', align_corners=self.align_corners)
        return x_warp, last_flow, cond_fea_all, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all


class AFWM(nn.Module):

    def __init__(self, opt, input_nc):
        super(AFWM, self).__init__()
        num_filters = [64,128,256,256,256]
        self.image_features = FeatureEncoder(3, num_filters) 
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(num_filters)
        self.cond_FPN = RefinePyramid(num_filters)

        # self.image_mobile = MobileNetV2_dynamicFPN()
        # self.cond_mobile = MobileNetV2_dynamicFPN()
        self.aflow_net = AFlowNet(len(num_filters), align_corners=opt.align_corners)
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge=None):

        #import ipdb; ipdb.set_trace()
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))
        image_pyramids = self.image_FPN(self.image_features(image_input))

        x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = self.aflow_net(image_input, image_edge, image_pyramids, cond_pyramids)

        return x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all


    def update_learning_rate(self,optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self,optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr

