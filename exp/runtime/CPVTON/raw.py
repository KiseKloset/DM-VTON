import torch
import torch.nn as nn

from .networks import GMM, UnetGenerator


class CPVTON(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.warp = GMM(256, 192, 5, device)
        self.gen = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)

    def forward(self, inputA, inputB):
        output1 = self.warp(inputA, inputB)
        output2 = torch.cat((inputA, inputB), dim=1)
        return self.gen(output2)
