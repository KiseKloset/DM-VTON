# Credit probably goes to Junyan somewhere
import torch
from torch import nn as nn

from ShineOn.models.networks.attention.sagan import SelfAttention
from ShineOn.models.networks.activation import Sine, Swish


class UnetGenerator(nn.Module):
    """
    Defines the Unet generator.
    |num_downs|: number of downsamplings in UNet. For example,
    if |num_downs| == 7, image of size 128x128 will become of size 1x1
    at the bottleneck
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        num_attention,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        use_self_attn=False,
        activation=None
    ):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            self_attn=use_self_attn if use_self_attn and num_attention > 0 else None,
            activation=activation
        )
        num_attention -= 1
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                self_attn=use_self_attn if use_self_attn and num_attention > 0 else None,
                activation=activation
            )
            num_attention -= 1
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            self_attn=use_self_attn if use_self_attn and num_attention > 0 else None,
            activation=activation
        )
        num_attention -= 1
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            self_attn=use_self_attn if use_self_attn and num_attention > 0 else None,
            activation=activation
        )
        num_attention -= 1
        # Self_Attn(ngf * 2, 'relu')
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            self_attn=use_self_attn if use_self_attn and num_attention > 0 else None,
            activation=activation
        )
        num_attention -= 1
        # Self_Attn(ngf, 'relu')
        unet_block = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            self_attn=use_self_attn if use_self_attn and num_attention > 0 else None,
            activation=activation
        )

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the submodule with skip connection.
    X -------------------identity---------------------- X
      |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        self_attn=False,
        use_dropout=False,
        activation=None
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        down_activation = nn.LeakyReLU(0.2, True) if activation is None else _get_activation_fn(activation)
        downnorm = norm_layer(inner_nc)
        up_activation = nn.ReLU(True) if activation is None else _get_activation_fn(activation)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            upconv = nn.Conv2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            down = [downconv]
            up = [up_activation, upsample, upconv, upnorm]
            if self_attn:
                down.append(SelfAttention(inner_nc, "relu"))
                up.append(SelfAttention(outer_nc, "relu"))

            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            upconv = nn.Conv2d(
                inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias
            )
            down = [down_activation, downconv]
            up = [up_activation, upsample, upconv, upnorm]
            if self_attn:
                down.append(SelfAttention(inner_nc, "relu"))
                up.append(SelfAttention(outer_nc, "relu"))
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            upconv = nn.Conv2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            down = [down_activation, downconv, downnorm]
            up = [up_activation, upsample, upconv, upnorm]
            if self_attn:
                down.append(SelfAttention(inner_nc, "relu"))
                up.append(SelfAttention(outer_nc, "relu"))

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            try:
                x_prime = self.model(x)
            except Exception as e:
                print(x, type(x), x.size(), x.type())
                raise e

            return torch.cat([x, x_prime], 1)


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return Swish()
    elif activation == "sine":
        return Sine()
    else:
        raise RuntimeError(f"The selected activation should be relu/gelu/swish/sine, not {activation}")
