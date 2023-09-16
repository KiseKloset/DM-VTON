import math
import os.path as osp
from typing import List

import torch
from torch import nn as nn
from torch.nn import functional as F

from ShineOn.models.networks.cpvton.unet import UnetGenerator


class UnetMaskModel(nn.Module):

    def __init__(self):
        super().__init__()
        n_frames = 1
        self.unet = UnetGenerator(
            input_nc= 10 * n_frames,
            output_nc= 4 * n_frames,
            num_downs=6,
            num_attention= 2,
            ngf=int(64 * (math.log(n_frames) + 1)),
            norm_layer=nn.InstanceNorm2d,
            use_self_attn=True,
            activation="relu",
        )


    def forward(self, person_representation, warped_cloths, flows=None, prev_im=None):
        # comment andrew: Do we need to interleave the concatenation? Or can we leave it
        #  like this? Theoretically the unet will learn where things are, so let's try
        #  simple concat for now.

        concat_tensor = torch.cat([person_representation, warped_cloths], 1)
        outputs = self.unet(concat_tensor)

        # teach the u-net to make the 1st part the rendered images, and
        # the 2nd part the masks
        boundary = 3
        weight_boundary = 4

        p_rendereds = outputs[:, 0:boundary, :, :]
        tryon_masks = outputs[:, boundary:weight_boundary, :, :]

        p_rendereds = F.tanh(p_rendereds)
        tryon_masks = F.sigmoid(tryon_masks)

        # chunk operation per individual frame
        warped_cloths_chunked = list(
            torch.chunk(warped_cloths, 1, dim=1)
        )
        p_rendereds_chunked = list(
            torch.chunk(p_rendereds, 1, dim=1)
        )
        tryon_masks_chunked = list(
            torch.chunk(tryon_masks, 1, dim=1)
        )

        # only use second frame for warping
        all_generated_frames = []
        for fIdx in range(1):
            p_rendered = p_rendereds_chunked[fIdx]

            p_tryon = (
                (1 - tryon_masks_chunked[fIdx]) * p_rendered  ##
                + tryon_masks_chunked[fIdx] * warped_cloths_chunked[fIdx]
            )

            all_generated_frames.append(p_tryon)

        p_tryons = torch.cat(all_generated_frames, dim=1)  # cat back to the channel dim

        return p_rendereds, tryon_masks, p_tryons
