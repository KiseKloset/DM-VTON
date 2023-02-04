import torch
import torch.nn as nn

def get_act_layer(act_type: str = 'relu') -> nn.Module:
    if act_type=='relu':
        return nn.ReLU(True)
    elif act_type=='lrelu':
        return nn.LeakyReLU(0.2, True)