import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from RMGN.models.afwm import AFWM
from RMGN.models.rmgn_generator import RMGNGenerator
from RMGN.options.test_options import TestOptions


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return
    
    checkpoint = torch.load(checkpoint_path)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        param_new = param
        param_ckpt = param
        checkpoint_new[param_new] = checkpoint[param_ckpt]
    
    model.load_state_dict(checkpoint_new)


class RMGN(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.opt = TestOptions().parse()
        ##### CONFIG #####
        self.opt.align_corners = True
        self.opt.hr = True
        self.opt.predmask = True
        ##### CONFIG #####

        self.warp_model = AFWM(self.opt, 3)
        self.gen_model = RMGNGenerator(multilevel=self.opt.multilevel, predmask=self.opt.predmask)
        if checkpoint != None:
            if checkpoint.get('warp') != None:
                load_checkpoint(self.warp_model, checkpoint['warp'])
            if checkpoint.get('gen') != None:
                load_checkpoint(self.gen_model, checkpoint['gen'])


    def forward(self, person, cloth, cloth_edge):
        cloth_edge = (cloth_edge > 0.5).float()
        cloth = cloth * cloth_edge
        
        # Warp
        warped_cloth, last_flow, = self.warp_model(person, cloth)
        warped_edge = F.grid_sample(cloth_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

        # Gen
        gen_inputs_cloth = torch.cat([warped_cloth, warped_edge], 1)
        gen_outputs, out_L1, out_L2, M_list = self.gen_model(person, gen_inputs_cloth)

        if self.opt.predmask:
            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
            
        else:
            p_rendered = gen_outputs
            p_rendered = torch.tanh(p_rendered)
            p_tryon = p_rendered

        return p_tryon
