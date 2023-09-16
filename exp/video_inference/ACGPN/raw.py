import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ACGPN.options.test_options import TestOptions
from ACGPN.models.pix2pixHD_model import InferenceModel


def changearm(old_label):
    label = old_label
    arm1 = (old_label == 11).float()
    arm2 = (old_label == 13).float()
    noise = (old_label == 7).float()
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label


class ACGPN(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        opt = TestOptions().parse()

        self.model = InferenceModel()

        if checkpoint is not None:
            opt.load_pretrain = True
            opt.U_path = checkpoint.get('U', '')
            opt.G1_path = checkpoint.get('G1', '')
            opt.G2_path = checkpoint.get('G2', '')
            opt.G_path = checkpoint.get('G', '')
        else:
            opt.load_pretrain = False
        
        self.model.initialize(opt)

    def forward(self, person, cloth, cloth_edge, pose, parse):
        t_mask = (parse == 7).float() 
        #
        # parse = parse * (1 - t_mask) + t_mask * 4
        mask_clothes = (parse == 4).float() 
        mask_fore = (parse > 0).float() 
        img_fore = person * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(parse)

        ############## Forward Pass ######################
        fake_image, warped_cloth, refined_cloth = self.model(Variable(parse), Variable(cloth_edge), Variable(img_fore), Variable(
            mask_clothes), Variable(cloth), Variable(all_clothes_label), Variable(person), Variable(pose), Variable(person), Variable(mask_fore))

        return fake_image
