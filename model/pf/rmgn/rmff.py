"""
Regional Mask Feature Fusion: https://paperswithcode.com/paper/rmgn-a-regional-mask-guided-network-for
Idea: https://paperswithcode.com/method/spade  
Edit from: https://github.com/jokerlc/RMGN-VITON/blob/main/models/aad.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch import Tensor


# Regional Mask Feature Fusion
class RMFF(nn.Module):
    def __init__(
        self, 
        h_channels: int, 
        a1_channels: int, 
        a2_channels: int,
        kernel_size: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.InstanceNorm2d(h_channels, affine=False)
        self.conv_h = nn.Conv2d(h_channels, 1, kernel_size=kernel_size, padding=kernel_size)
        self.conv_a1_gamma = nn.Conv2d(a1_channels, h_channels, kernel_size=kernel_size, padding=padding)
        self.conv_a1_beta = nn.Conv2d(a1_channels, h_channels, kernel_size=kernel_size, padding=padding)
        self.conv_a2_gamma = nn.Conv2d(a2_channels, h_channels, kernel_size=kernel_size, padding=padding)
        self.conv_a2_beta = nn.Conv2d(a2_channels, h_channels, kernel_size=kernel_size, padding=padding) 

    def forward(
        self, 
        f_in: Tensor, 
        a1: Tensor, 
        a2: Tensor
    ) -> tuple[Tensor]:
            
        h = self.norm(f_in)

        a1_gamma = self.conv_a1_gamma(a1)
        a1_beta = self.conv_a1_beta(a1)

        a2_gamma = self.conv_a2_gamma(a2)
        a2_beta = self.conv_a2_beta(a2)

        A1 = (a1_gamma + 1) * h + a1_beta
        A2 = (a2_gamma + 1) * h + a2_beta

        M = torch.sigmoid(self.conv_h(h))
        out = (1 - M) * A1 + M * A2
        
        return out, M


# RMFF Resnet block
class RMFFResBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        a1_channels: int, 
        a2_channels: int,
        multi: int = 1,
    ) -> None:
        super().__init__()
        self.learned_shortcut = (in_channels != out_channels)
        mid_channels = min(in_channels, out_channels)
        
        self.rmff1 = RMFF(in_channels, a1_channels, a2_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels * multi, mid_channels, kernel_size=3, padding=1))
        self.rmff2 = RMFF(mid_channels, a1_channels, a2_channels)
        self.conv2 = spectral_norm(nn.Conv2d(mid_channels * multi, out_channels, kernel_size=3, padding=1))
        
        if self.learned_shortcut:
            self.rmff_s = RMFF(in_channels, a1_channels, a2_channels)
            self.conv_s = spectral_norm(nn.Conv2d(in_channels * multi, out_channels, kernel_size=3, padding=1))

    def forward(self, 
        f_in: Tensor, 
        a1: Tensor, 
        a2: Tensor,
    ) -> tuple:
        M_list = []
        out, M1 = self.rmff1(f_in, a1, a2)
        M_list.append(M1)
        out = self.conv1(self.relu(out))

        out, M2 = self.rmff2(out, a1, a2)
        M_list.append(M2)
        out = self.conv2(self.relu(out))

        if self.learned_shortcut:
            inp_s, M_s = self.rmff_s(f_in, a1, a2)
            M_list.append(M_s)
            inp_s = self.conv_s(self.relu(inp_s))
        else:
            inp_s = f_in
        out = out + inp_s

        return out, M_list

        
class RMFFResNetwork(nn.Module):
    def __init__(self, 
        num_features: int = 64, 
        out_nc: int = 3, 
        SR_scale: int = 1, 
        multilevel: bool = False, 
        predmask: bool = True
    ) -> None:
        super().__init__()
        self.multilevel = multilevel
        self.predmask = predmask

        self.conv_head = nn.Conv2d(num_features * 16, num_features * 16, 3, padding=1)

        self.rmff_res_block_mid1 = RMFFResBlock(num_features * 16, num_features * 16, num_features * 8, num_features * 8)
        self.rmff_res_block_mid2 = RMFFResBlock(num_features * 4, num_features * 4, num_features * 2, num_features * 2)
        self.rmff_res_block_mid3 = RMFFResBlock(num_features * 1, num_features * 1, num_features, num_features)

        self.rmff_res_block_up1 = RMFFResBlock(num_features * 16, num_features * 8, num_features * 8, num_features * 8)
        self.rmff_res_block_up2 = RMFFResBlock(num_features * 8, num_features * 4, num_features * 4, num_features * 4)
        self.rmff_res_block_up3 = RMFFResBlock(num_features * 4, num_features * 2, num_features * 2, num_features * 2)
        self.rmff_res_block_up4 = RMFFResBlock(num_features * 2, num_features * 1, num_features, num_features)

        self.conv_final = nn.Conv2d(num_features, out_nc * (SR_scale ** 2), 3, padding=1)
        
        self.final = nn.PixelShuffle(SR_scale) if SR_scale > 1 else nn.Identity()
        
        if self.multilevel:
            self.conv_L1 = nn.Conv2d(num_features * 16, 3, 3, padding=1)
            self.conv_L2 = nn.Conv2d(num_features * 4, 3, 3, padding=1)
        
    def upsample2x(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(
        self, 
        A1_list: list, 
        A2_list: list,
    ) -> tuple:
        M_list = []
        out = torch.cat([A1_list[0], A2_list[0]], dim=1)    
        out = self.conv_head(out)                             
        
        out, M_mid1 = self.rmff_res_block_mid1(out, A1_list[0], A2_list[0]) 
        M_list += M_mid1
        out_L1 = self.conv_L1(out) if self.multilevel else None

        out, M_up1 = self.rmff_res_block_up1(out, A1_list[0], A2_list[0])  
        M_list += M_up1
        out = self.upsample2x(out)    

        out, M_up2 = self.rmff_res_block_up2(out, A1_list[1], A2_list[1])   
        M_list += M_up2
        out = self.upsample2x(out)                              
        
        out, M_mid2 = self.rmff_res_block_mid2(out, A1_list[2], A2_list[2])
        M_list += M_mid2
        out_L2 = self.conv_L2(out) if self.multilevel else None

        out, M_up3 = self.rmff_res_block_up3(out, A1_list[2], A2_list[2])  
        M_list += M_up3
        out = self.upsample2x(out)  

        out, M_up4 = self.rmff_res_block_up4(out, A1_list[3], A2_list[3])   
        M_list += M_up4
        out = self.upsample2x(out)                              
        
        out, M_mid3 = self.rmff_res_block_mid3(out, A1_list[4], A2_list[4])   
        M_list += M_mid3
        out = self.conv_final(out)                           
        
        out = self.final(out)
        
        return out, out_L1, out_L2, M_list