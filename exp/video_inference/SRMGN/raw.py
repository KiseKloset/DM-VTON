import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from SRMGN.models.pfafn.afwm_test import AFWM
from SRMGN.models.mobile_unet_generator import MobileNetV2_unet
from SRMGN.options.test_options import TestOptions

from utils import blending_fn

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return
    checkpoint = torch.load(checkpoint_path)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)


class SRMGN(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        opt = TestOptions().parse()
        opt.align_corners = True

        self.warp_model = AFWM(opt, 3)
        self.gen_model = MobileNetV2_unet(7, 4)
        # self.gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
        if checkpoint != None:
            if checkpoint.get('warp') != None:
                load_checkpoint(self.warp_model, checkpoint['warp'])
            if checkpoint.get('gen') != None:
                load_checkpoint(self.gen_model, checkpoint['gen'])

        self.prev_mask = None


    def forward(self, person, cloth, cloth_edge):
        # print("C", person[0][0][100][100], cloth[0][0][100][100], cloth_edge[0][0][100][100])

        cloth_edge = (cloth_edge > 0.5).float()
        cloth = cloth * cloth_edge

        # Warp
        warped_cloth, last_flow, = self.warp_model(person, cloth)
        # print("D", warped_cloth[0][0][100][100], last_flow[0][0][100][100])
        warped_edge = F.grid_sample(cloth_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        # print("E", warped_edge[0][0][100][100])

        # Gen
        gen_inputs = torch.cat([person, warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        # print("F", gen_outputs[0][0][100][100])
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        # print("G", p_rendered[0][0][100][100], m_composite[0][0][100][100])
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge

        # Smoothing mask
        if self.prev_mask is not None:
            m_composite = blending_fn(self.prev_mask, m_composite)
        self.prev_mask = m_composite

        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        # TUNGPNT2
        # visualize each module
        # combine = torch.cat([
        #     person, cloth, cloth_edge.expand(-1, 3, -1, -1), 
        #     warped_cloth, warped_edge.expand(-1, 3, -1, -1),
        #     p_rendered, m_composite.expand(-1, 3, -1, -1), 
        #     p_tryon
        # ], -1).squeeze()

        # return combine # p_tryon
        return p_tryon