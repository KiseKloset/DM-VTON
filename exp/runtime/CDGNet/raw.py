import argparse
import numpy as np
import torch

from torch.utils import data
from .CDGNet import Res_Deeplab
import os
import torchvision.transforms as transforms
from copy import deepcopy

from PIL import Image as PILImage

IGNORE_LABEL = 255
NUM_CLASSES = 20
INPUT_SIZE = (473,473)

# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
def get_lip_palette():
    palette = [0,0,0,
            128,0,0,
            255,0,0,
            0,85,0,
            170,0,51,
            255,85,0,
            0,0,85,
            0,119,221,
            85,85,0,
            0,85,85,
            85,51,0,
            52,86,128,
            0,128,0,
            0,0,255,
            51,170,221,
            0,255,255,
            85,255,170,
            170,255,85,
            255,255,0,
            255,170,0]
    return palette               
def get_palette(num_cls):
  """ Returns the color map for visualizing the segmentation mask.

  Inputs:
    =num_cls=
      Number of classes.

  Returns:
      The color map.
  """
  n = num_cls
  palette = [0] * (n * 3)
  for j in range(0, n):
    lab = j
    palette[j * 3 + 0] = 0
    palette[j * 3 + 1] = 0
    palette[j * 3 + 2] = 0
    i = 0
    while lab:
      palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
      palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
      palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
      i += 1
      lab >>= 3
  return palette


class CDGNet(torch.nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()
        self.model = Res_Deeplab(num_classes)
        self.interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)


    def forward(self, image):
        org_img = image
        normal_img = org_img
        flipped_img = torch.flip(org_img, [-1])
        fused_img = torch.cat((normal_img,flipped_img), dim=0)
        outputs = self.model(fused_img)
        prediction = self.interp(outputs[0][-1]).permute(0, 2, 3, 1) #N,H,W,C
        single_out = prediction
        single_out_flip = torch.zeros( single_out.shape ).to(image.device)
        single_out_tmp = prediction[:, :,:,:]
        for c in range(14):
            single_out_flip[:,:, :, c] = single_out_tmp[:, :, :, c]
        single_out_flip[:, :, :, 14] = single_out_tmp[:, :, :, 15]
        single_out_flip[:, :, :, 15] = single_out_tmp[:, :, :, 14]
        single_out_flip[:, :, :, 16] = single_out_tmp[:, :, :, 17]
        single_out_flip[:, :, :, 17] = single_out_tmp[:, :, :, 16]
        single_out_flip[:, :, :, 18] = single_out_tmp[:, :, :, 19]
        single_out_flip[:, :, :, 19] = single_out_tmp[:, :, :, 18]
        single_out_flip = torch.flip(single_out_flip, [2])  
        # Fuse two outputs
        single_out = ( single_out+single_out_flip ) / 2

        return single_out