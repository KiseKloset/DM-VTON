import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .pretrained_networks import alexnet, squeezenet, vgg16


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    # in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(
        self,
        pretrained=True,
        net='alex',
        version='0.1',
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        model_path=None,
        eval_mode=True,
        verbose=False,
    ):
        """Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained normalization bug; corresponds to https://arxiv.org/abs/1801.03924v1
        model_path : 'str'
            [None] is default and loads the pretrained from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super().__init__()
        if verbose:
            print(
                'Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'
                % ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off')
            )

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if pretrained:
                if model_path is None:
                    import inspect
                    import os

                    model_path = os.path.abspath(
                        os.path.join(
                            inspect.getfile(self.__init__),
                            '..',
                            f'weights/v{version}/{net}.pth',
                        )
                    )

                if verbose:
                    print('Loading model from: %s' % model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == '0.1'
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [
                    upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)
                ]
        else:
            if self.spatial:
                res = [
                    upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                    for kk in range(self.L)
                ]

        val = 0
        for lidx in range(self.L):
            val += res[lidx]

        if retPerLayer:
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    '''A single linear layer which does a 1x1 conv'''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def calculate_lpips_given_paths(paths, device, backbone='vgg', version='0.1'):
    loss_fn = LPIPS(net=backbone, version=version).to(device)
    loss_fn.eval()

    files = os.listdir(paths[0])

    lpips_avg = 0
    for f in files:
        if not os.path.exists(os.path.join(paths[1], f)):
            raise FileNotFoundError(f'{os.path.join(paths[1], f)} does not exist')

        # Load images
        # img0 = im2tensor(load_image(os.path.join(paths[0], f))).to(device) # RGB image from [-1,1]
        # img1 = im2tensor(load_image(os.path.join(paths[1], f))).to(device)
        img0 = (
            np.array(Image.open(os.path.join(paths[0], f))).transpose(2, 0, 1).astype(np.float32)
            / 255
        ) * 2 - 1
        img1 = (
            np.array(Image.open(os.path.join(paths[1], f))).transpose(2, 0, 1).astype(np.float32)
            / 255
        ) * 2 - 1
        img0 = torch.from_numpy(img0).unsqueeze(0).to(device)
        img1 = torch.from_numpy(img1).unsqueeze(0).to(device)

        # Compute distance
        lpips_avg += loss_fn(img0, img1).item()

    lpips_avg /= len(files)

    return lpips_avg
