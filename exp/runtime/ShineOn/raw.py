import torch.nn as nn

from ShineOn.models.warp_model import WarpModel
from ShineOn.models.unet_mask_model import UnetMaskModel

class ShineOn(nn.Module):
    def __init__(self):
        super().__init__()
        self.warp = WarpModel()
        self.gen = UnetMaskModel()

    def forward(self, cloth, person):
        grid, theta = self.warp(person, cloth)
        return self.gen(person, cloth)


