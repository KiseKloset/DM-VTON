from math import exp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable


def gaussian(window_size: int, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        

# TODO: CHECK
'''
window = torch.tensor([[0.0448, 0.2856, 0.3001, 0.2856, 0.0448],
                       [0.2856, 0.9796, 1.0000, 0.9796, 0.2856],
                       [0.3001, 1.0000, 1.0000, 1.0000, 0.3001],
                       [0.2856, 0.9796, 1.0000, 0.9796, 0.2856],
                       [0.0448, 0.2856, 0.3001, 0.2856, 0.0448]])
'''
def ssim(imgs1: Tensor, imgs2: Tensor, window_size: int = 11, size_average: bool = True):
    def _ssim(
        imgs1: Tensor, 
        imgs2: Tensor, 
        window: Tensor, 
        window_size: int,
        channel: int, 
        size_average: bool = True
    ):  
        # compute mean, standard deviation, and cross-covariance of images
        mu1 = F.conv2d(imgs1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(imgs2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(imgs1*imgs1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(imgs2*imgs2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(imgs1*imgs2, window, padding = window_size//2, groups = channel) - mu1_mu2

        # set constants for SSIM calculation
        C1 = 0.01**2
        C2 = 0.03**2
        
        # calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(-1).mean(-1).mean(-1)
    
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    _, channel, _, _ = imgs1.size()
    window = create_window(window_size, channel)
    
    window = window.to(imgs1.get_device()).type_as(imgs1)
    
    return _ssim(imgs1, imgs2, window, window_size, channel, size_average)


