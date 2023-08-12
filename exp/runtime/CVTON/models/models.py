import torch
import torch.nn as nn
from torch.nn import init

import CVTON.models.generator as generators


class OASIS_model(nn.Module):
    
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.netG = generators.OASIS_Simple(opt)                    
        self.init_networks()        
        

    def forward(self, image, label, agnostic=None):        
        with torch.no_grad():
            fake = self.netG(image["I_m"], image["C_t"], label["body_seg"], label["cloth_seg"], label["densepose_seg"], agnostic=agnostic)
        return fake            


    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def preprocess_input(opt, data, device):
    data['cloth_label'] = data['cloth_label'].long()
    data['body_label'] = data['body_label'].long()
    data['densepose_label'] = data['densepose_label'].long()
    
    data['cloth_label'] = data['cloth_label'].to(device)
    data['body_label'] = data['body_label'].to(device)
    data['densepose_label'] = data['densepose_label'].to(device)
    
    for key in data['image'].keys():
        data['image'][key] = data['image'][key].to(device)
        
    label_body_map = data['body_label']
    bs, _, h, w = label_body_map.size()
    nc = opt.semantic_nc[0]
    input_body_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_body_semantics = input_body_label.scatter_(1, label_body_map, 1.0)
    
    label_cloth_map = data['cloth_label']
    bs, _, h, w = label_cloth_map.size()
    nc = opt.semantic_nc[1]
    input_cloth_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_cloth_semantics = input_cloth_label.scatter_(1, label_cloth_map, 1.0)
    
    label_densepose_map = data['densepose_label']
    bs, _, h, w = label_densepose_map.size()
    nc = opt.semantic_nc[2]
    input_densepose_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_densepose_semantics = input_densepose_label.scatter_(1, label_densepose_map, 1.0)
    
    
    return data['image'], {"body_seg": input_body_semantics, "cloth_seg": input_cloth_semantics, "densepose_seg": input_densepose_semantics}