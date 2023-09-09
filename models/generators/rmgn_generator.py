import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from models.base_model import BaseModel
from models.extractors.rmgn_extractor import AttrEncoder


class AAD(nn.Module):
    def __init__(self, h_nc, a1_nc, a2_nc):
        super().__init__()
        ks = 1
        pw = ks // 2

        self.norm = nn.InstanceNorm2d(h_nc, affine=False)
        self.conv_h = nn.Conv2d(h_nc, 1, kernel_size=ks, padding=pw)
        self.conv_a1_gamma = nn.Conv2d(a1_nc, h_nc, kernel_size=ks, padding=pw)
        self.conv_a1_beta = nn.Conv2d(a1_nc, h_nc, kernel_size=ks, padding=pw)

        self.conv_a2_gamma = nn.Conv2d(a2_nc, h_nc, kernel_size=ks, padding=pw)
        self.conv_a2_beta = nn.Conv2d(a2_nc, h_nc, kernel_size=ks, padding=pw)

    def forward(self, h_in, a1, a2):
        h = self.norm(h_in)

        a1_gamma = self.conv_a1_gamma(a1)
        a1_beta = self.conv_a1_beta(a1)

        a2_gamma = self.conv_a2_gamma(a2)
        a2_beta = self.conv_a2_beta(a2)

        A1 = (a1_gamma + 1) * h + a1_beta
        A2 = (a2_gamma + 1) * h + a2_beta

        M = torch.sigmoid(self.conv_h(h))
        h_out = (1 - M) * A1 + M * A2

        return h_out, M


class AADResnetBlock(nn.Module):
    def __init__(self, inp_nc, out_nc, a1_nc, a2_nc):
        super().__init__()
        self.learned_shortcut = inp_nc != out_nc
        mid_nc = min(inp_nc, out_nc)

        multi = 1

        self.AAD_1 = AAD(inp_nc, a1_nc, a2_nc)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = spectral_norm(nn.Conv2d(inp_nc * multi, mid_nc, kernel_size=3, padding=1))

        self.AAD_2 = AAD(mid_nc, a1_nc, a2_nc)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = spectral_norm(nn.Conv2d(mid_nc * multi, out_nc, kernel_size=3, padding=1))

        if self.learned_shortcut:
            self.AAD_s = AAD(inp_nc, a1_nc, a2_nc)
            self.relu_s = nn.ReLU(inplace=True)
            self.conv_s = spectral_norm(nn.Conv2d(inp_nc * multi, out_nc, kernel_size=3, padding=1))

    def forward(self, inp, a1, a2):
        M_list = []
        out, M_1 = self.AAD_1(inp, a1, a2)
        M_list.append(M_1)
        out = self.conv_1(self.relu_1(out))

        out, M_2 = self.AAD_2(out, a1, a2)
        M_list.append(M_2)
        out = self.conv_2(self.relu_2(out))

        if self.learned_shortcut:
            inp_s, M_S = self.AAD_s(inp, a1, a2)
            M_list.append(M_S)
            inp_s = self.conv_s(self.relu_s(inp_s))
        else:
            inp_s = inp

        out = out + inp_s

        return out, M_list


class AADGenerator(nn.Module):
    def __init__(self, nf=64, out_nc=3, SR_scale=1, multilevel=False, predmask=True):
        super().__init__()

        self.conv_head = nn.Conv2d(nf * 16, nf * 16, 3, padding=1)

        self.AADResBlk_mid_0 = AADResnetBlock(nf * 16, nf * 16, nf * 8, nf * 8)
        self.AADResBlk_mid_1 = AADResnetBlock(nf * 4, nf * 4, nf * 2, nf * 2)
        self.AADResBlk_mid_2 = AADResnetBlock(nf * 1, nf * 1, nf, nf)

        self.AADResBlk_up_0 = AADResnetBlock(nf * 16, nf * 8, nf * 8, nf * 8)
        self.AADResBlk_up_1 = AADResnetBlock(nf * 8, nf * 4, nf * 4, nf * 4)
        self.AADResBlk_up_2 = AADResnetBlock(nf * 4, nf * 2, nf * 2, nf * 2)
        self.AADResBlk_up_3 = AADResnetBlock(nf * 2, nf * 1, nf, nf)

        self.conv_final = nn.Conv2d(nf, out_nc * (SR_scale**2), 3, padding=1)

        self.final = nn.PixelShuffle(SR_scale) if SR_scale > 1 else nn.Identity()

        self.multilevel = multilevel
        if self.multilevel:
            self.conv_L1 = nn.Conv2d(nf * 16, 3, 3, padding=1)
            self.conv_L2 = nn.Conv2d(nf * 4, 3, 3, padding=1)

        self.predmask = predmask

    @staticmethod
    def upsample(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, A1_list, A2_list):
        M_list = []
        x = torch.cat([A1_list[0], A2_list[0]], dim=1)
        x = self.conv_head(x)

        x, M_mid_0 = self.AADResBlk_mid_0(x, A1_list[0], A2_list[0])
        M_list += M_mid_0
        if self.multilevel:
            x_L1 = self.conv_L1(x)
        else:
            x_L1 = None

        x, M_up_0 = self.AADResBlk_up_0(x, A1_list[0], A2_list[0])
        M_list += M_up_0
        x = self.upsample(x)
        x, M_up_1 = self.AADResBlk_up_1(x, A1_list[1], A2_list[1])
        M_list += M_up_1
        x = self.upsample(x)

        x, M_mid_1 = self.AADResBlk_mid_1(x, A1_list[2], A2_list[2])
        M_list += M_mid_1
        if self.multilevel:
            x_L2 = self.conv_L2(x)
        else:
            x_L2 = None

        x, M_up_2 = self.AADResBlk_up_2(x, A1_list[2], A2_list[2])
        M_list += M_up_2
        x = self.upsample(x)
        x, M_up_3 = self.AADResBlk_up_3(x, A1_list[3], A2_list[3])
        M_list += M_up_3
        x = self.upsample(x)

        x, M_mid_2 = self.AADResBlk_mid_2(x, A1_list[4], A2_list[4])
        M_list += M_mid_2
        x = self.conv_final(x)

        x = self.final(x)

        return x, x_L1, x_L2, M_list


class RMGNGenerator(BaseModel):
    def __init__(self, in_person_nc=3, in_clothes_nc=4, nf=64, multilevel=False, predmask=True):
        super().__init__()
        out_nc = 4
        self.in_nc = [in_person_nc, in_clothes_nc]

        SR_scale = 1
        aei_encoder_head = False
        head_layers = int(np.log2(SR_scale)) + 1 if aei_encoder_head or SR_scale > 1 else 0

        self.inp_encoder = AttrEncoder(nf=nf, in_nc=in_person_nc, head_layers=head_layers)
        self.ref_encoder = AttrEncoder(nf=nf, in_nc=in_clothes_nc, head_layers=head_layers)
        self.generator = AADGenerator(
            nf=nf, out_nc=out_nc, SR_scale=SR_scale, multilevel=multilevel, predmask=predmask
        )

    def get_inp_attr(self, inp):
        inp_attr_list = self.inp_encoder(inp)
        return inp_attr_list

    def get_ref_attr(self, ref):
        ref_attr_list = self.ref_encoder(ref)
        return ref_attr_list

    def get_gen(self, inp_attr_list, ref_attr_list):
        out = self.generator(inp_attr_list, ref_attr_list)
        return out

    def forward(self, x):
        inp, ref = torch.split(x, split_size_or_sections=self.in_nc, dim=1)
        inp_attr_list = self.get_inp_attr(inp)
        ref_attr_list = self.get_ref_attr(ref)
        out, out_L1, out_L2, M_list = self.get_gen(inp_attr_list, ref_attr_list)
        return out  # , out_L1, out_L2, M_list
