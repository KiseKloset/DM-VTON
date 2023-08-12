import warnings
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

'''
module                                  arguments   
BACKBONE                  
models.common.Conv                      [3, 32, 6, 2, 2]              
models.common.Conv                      [32, 64, 3, 2]                
models.common.C3                        [64, 64, 1]  

models.common.Conv                      [64, 128, 3, 2]               
models.common.C3                        [128, 128, 2] 

models.common.Conv                      [128, 256, 3, 2]              
models.common.C3                        [256, 256, 3] 

models.common.Conv                      [256, 512, 3, 2]              
models.common.C3                        [512, 512, 1]    

models.common.SPPF                      [512, 512, 5]   

HEAD
models.common.Conv                      [512, 256, 1, 1]              
torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
models.common.Concat                    [1]                           
models.common.C3                        [512, 256, 1, False]          
models.common.Conv                      [256, 128, 1, 1]              
torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
models.common.Concat                    [1]                           
models.common.C3                        [256, 128, 1, False]          
models.common.Conv                      [128, 128, 3, 2]              
models.common.Concat                    [1]                           
models.common.C3                        [256, 256, 1, False]          
models.common.Conv                      [256, 256, 3, 2]              
models.common.Concat                    [1]                           
models.common.C3                        [512, 512, 1, False]          
'''
class YOLOv5Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Change stride of conv0 to 1
        self.cv0 = Conv(*[3, 32, 6, 1, 2]) 
        self.cv1 = Conv(*[32, 64, 3, 2])
        self.c3_1 = C3(*[64, 64, 1])

        self.cv2 = Conv(*[64, 128, 3, 2])
        self.c3_2 = C3(*[128, 128, 2])

        self.cv3 = Conv(*[128, 256, 3, 2])
        self.c3_3 = C3(*[256, 256, 3])

        self.cv4 = Conv(*[256, 512, 3, 2])
        self.c3_4 = C3(*[512, 512, 1])

        self.sppf = SPPF(*[512, 512, 5])
    
    def forward(self, x) -> list:
        out = []

        feature = self.c3_1(self.cv1(self.cv0(x)))

        feature = self.c3_2(self.cv2(feature))
        out.append(feature) # 128 x size/4

        feature = self.c3_3(self.cv3(feature))
        out.append(feature) # 256 x size/8

        feature = self.sppf(self.c3_4(self.cv4(feature)))
        out.append(feature) # 512 x size/16

        return out


class YOLOv5Head(nn.Module):
    def __init__(self):
        super().__init__()
        # Add upsample + conv1 add before each output
        self.cv1 = Conv(*[512, 256, 1, 1])             
        self.upsample1 = nn.Upsample(*[None, 2, 'nearest'])       
        self.concat1 = Concat(*[1])            
        self.c3_1 = C3(*[512, 256, 1, False])       
        
        self.cv2 = Conv(*[256, 128, 1, 1])           
        self.upsample2 = nn.Upsample(*[None, 2, 'nearest'])        
        self.concat2 = Concat(*[1])                          
        self.c3_2 = C3(*[256, 128, 1, False])
        self.upsample2out = nn.Upsample(*[None, 2, 'nearest'])
        self.cv1x1_2 = nn.Conv2d(128, 256, 1)  

        self.cv3 = Conv(*[128, 128, 3, 2])              
        self.concat3 = Concat(*[1])                         
        self.c3_3 = C3(*[256, 256, 1, False])   
        self.upsample3out = nn.Upsample(*[None, 2, 'nearest'])  
        self.cv1x1_3 = nn.Conv2d(256, 256, 1)  

        self.cv4 = Conv(*[256, 256, 3, 2])          
        self.concat4 = Concat(*[1])                         
        self.c3_4 = C3(*[512, 512, 1, False])    
        self.upsample4out = nn.Upsample(*[None, 2, 'nearest']) 
        self.cv1x1_4 = nn.Conv2d(512, 256, 1)  

    def forward(self, x: list) -> list:
        out = []
        feature1 = self.cv1(x[2])
        now = self.c3_1(
            self.concat1(
                [x[1], self.upsample1(feature1)]
            )
        )
        
        feature2 = self.cv2(now)
        now = self.c3_2(
            self.concat2(
                [x[0], self.upsample2(feature2)]
            )
        )
        out.append(self.cv1x1_2(self.upsample2out(now)))

        now = self.c3_3(
            self.concat3(
                [feature2, self.cv3(now)]
            )
        )
        out.append(self.cv1x1_3(self.upsample3out(now)))

        now = self.c3_4(
            self.concat4(
                [feature1, self.cv4(now)]
            )
        )
        out.append(self.cv1x1_4(self.upsample4out(now)))

        return out



import contextlib
import os
import time
import torch.nn.functional as F
class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class.
    Usage: as a decorator with @Profile() or as a context manager with 'with Profile():'
    """

    def __init__(self, t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        """
        Start timing.
        """
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """
        Stop timing.
        """
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """
        Get current time.
        """
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()



if __name__=='__main__':
    dt = Profile()

    device = torch.device(f'cuda:1')
    model = nn.Sequential(YOLOv5Backbone(), YOLOv5Head()).to(device)
    model.eval()
    a = torch.rand(1, 3, 256, 192).to(device)

    x = model(a)
    num_sample = 1000
    with torch.no_grad():
        for i in range(num_sample):
            with dt:
                x = model(a)
    print(dt.t / num_sample * 1E3)
