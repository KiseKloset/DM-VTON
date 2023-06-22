from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor

from models.common.resunet import ResUnetSkipConnectionBlock


class ResUnetGenerator(nn.Module):
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        num_downs: int, 
        filters: list[int] = [64, 128, 256, 512],
        use_dropout: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
    ) -> None:
        """Construct a Unet generator
        Parameters:
            in_channels (int)   -- the number of channels in input images
            out_channels (int)  -- the number of channels in output images
            num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                    image of size 128x128 will become of size 1x1 # at the bottleneck
            filters (list[int]) -- the filters list of conv layers
            norm_layer          -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # inner most block
        unet_block = ResUnetSkipConnectionBlock(
                            inner_nc=filters[3], 
                            outer_nc=filters[3], 
                            input_nc=None, 
                            submodule=None, 
                            norm_layer=norm_layer, 
                            innermost=True)

        for _ in range(num_downs - 5):
            # add intermediate layers with filters in the last conv layer
            unet_block = ResUnetSkipConnectionBlock(
                            inner_nc=filters[3], 
                            outer_nc=filters[3], 
                            input_nc=None, 
                            submodule=unet_block, 
                            norm_layer=norm_layer, 
                            use_dropout=use_dropout
                        )
        
        # gradually reduce the number of filters from filters[3] -> filters[0]
        unet_block = ResUnetSkipConnectionBlock(filters[3], filters[2], input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(filters[2], filters[1], input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(filters[1], filters[0], input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        # outer most block
        outermost_block = ResUnetSkipConnectionBlock(
                            inner_nc=filters[0],             
                            outer_nc=out_channels, 
                            input_nc=in_channels, 
                            submodule=unet_block, 
                            outermost=True, 
                            norm_layer=norm_layer)

        self.model = outermost_block

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
