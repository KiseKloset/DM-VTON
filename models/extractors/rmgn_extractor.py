import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv4x4(in_c, out_c, norm=nn.InstanceNorm2d, dilation=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            dilation=dilation,
            padding=padding,
            bias=False,
        ),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


class deconv4x4(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.InstanceNorm2d, dilation=1, padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn = norm(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


def head_conv3x3(in_c, out_c, stride=1, norm=nn.InstanceNorm2d):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        norm(out_c),
        nn.LeakyReLU(0.1, inplace=True),
    )


def get_head(in_c, out_c, num_layers=1):
    assert num_layers >= 1
    layers = [head_conv3x3(in_c, out_c, 1)]
    for _ in range(num_layers - 1):
        layers.append(head_conv3x3(out_c, out_c, 2))
    return nn.Sequential(*layers)


class AttrEncoder(nn.Module):
    def __init__(self, nf=64, in_nc=3, head_layers=0):
        super().__init__()

        if head_layers > 0:
            head_conv = get_head(in_nc, nf // 2, head_layers)
            conv1 = conv4x4(nf // 2, nf // 2)
            self.conv1 = nn.Sequential(head_conv, conv1)
        else:
            self.conv1 = conv4x4(in_nc, nf // 2)  # 2x

        self.conv2 = conv4x4(nf // 2, nf)  # 4x
        self.conv3 = conv4x4(nf, nf * 2)  # 8x
        self.conv4 = conv4x4(nf * 2, nf * 4)  # 16x
        self.conv5 = conv4x4(nf * 4, nf * 8)  # 32x

        self.deconv1 = deconv4x4(nf * 8, nf * 4)  # 16x
        self.deconv2 = deconv4x4(nf * 8, nf * 2)  # 8x
        self.deconv3 = deconv4x4(nf * 4, nf)  # 4x
        self.deconv4 = deconv4x4(nf * 2, nf // 2)  # 2x

    def forward(self, x):
        feat1 = self.conv1(x)  # nf//2, h/2, w/2
        feat2 = self.conv2(feat1)  # nf, h/4, w/4
        feat3 = self.conv3(feat2)  # nf*2, h/8, w/8
        feat4 = self.conv4(feat3)  # nf*4, h/16, w/16
        feat5 = self.conv5(feat4)  # nf*8, h/32, w/32

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


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class AttrDilatedEncoder(nn.Module):
    def __init__(self, nf=64, in_nc=3, head_layers=0):
        super().__init__()

        if head_layers > 0:
            head_conv = get_head(in_nc, nf // 2, head_layers)
            conv1 = conv4x4(nf // 2, nf // 2)
            self.conv1 = nn.Sequential(head_conv, conv1)
        else:
            self.conv1 = conv4x4(in_nc, nf // 2)  # 2x

        self.conv2 = conv4x4(nf // 2, nf, dilation=8, padding=get_pad(256, 4, 2, 8) + 1)  # 4x
        self.conv3 = conv4x4(nf, nf * 2, dilation=4, padding=get_pad(128, 4, 2, 4) + 1)  # 8x
        self.conv4 = conv4x4(nf * 2, nf * 4, dilation=2, padding=get_pad(64, 4, 2, 2) + 1)  # 16x
        self.conv5 = conv4x4(nf * 4, nf * 8)  # 32x

        self.deconv1 = deconv4x4(nf * 8, nf * 4)  # 16x
        self.deconv2 = deconv4x4(nf * 8, nf * 2, dilation=3, padding=get_pad(64, 4, 2, 2) + 2)  # 8x
        self.deconv3 = deconv4x4(nf * 4, nf, dilation=5, padding=get_pad(128, 4, 2, 4) + 2)  # 4x
        self.deconv4 = deconv4x4(
            nf * 2, nf // 2, dilation=7, padding=get_pad(256, 4, 2, 8) - 1
        )  # 2x

    def forward(self, x):
        feat1 = self.conv1(x)  # nf//2, h/2, w/2
        feat2 = self.conv2(feat1)  # nf, h/4, w/4
        feat3 = self.conv3(feat2)  # nf*2, h/8, w/8
        feat4 = self.conv4(feat3)  # nf*4, h/16, w/16
        feat5 = self.conv5(feat4)  # nf*8, h/32, w/32

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
#         nn.Conv2d(
#             in_channels=in_c,
#             out_channels=out_c,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False
#         ),
#         norm(out_c),
#         nn.LeakyReLU(0.1, inplace=True)
#     )
#
#
# class AttrEncoderV2(nn.Module):
#     def __init__(self, nf=64, in_nc=3):
#         super(AttrEncoderV2, self).__init__()
#         self.conv_head = nn.Conv2d(in_nc, nf, kernel_size=3, padding=1)
#         self.conv1 = conv3x3(nf, nf, stride=1)  # nf * h * w
#         self.conv2 = conv3x3(nf, nf, stride=2)  # nf * h/2 * w/2
#         self.conv3 = conv3x3(nf, nf*2, stride=2)  # 2nf * h/4 * w/4
#         self.conv4 = conv3x3(nf*2, nf*4, stride=2)  # 4nf * h/8 * w/8
#         self.conv5 = conv3x3(nf*4, nf*8, stride=2)  # 8nf * h/16 * w/16

#     def forward(self, x):
#         feat = self.conv_head(x)
#         feat1 = self.conv1(feat)
#         feat2 = self.conv2(feat1)
#         feat3 = self.conv3(feat2)
#         feat4 = self.conv4(feat3)
#         feat5 = self.conv5(feat4)
#         return feat5, feat4, feat3, feat2, feat1
