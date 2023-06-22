from typing import Callable, Optional, Union

import torch.nn as nn
from torchvision.ops.misc import ConvNormActivation


def get_act(act_type: str) -> nn.Module:
    if act_type is None:
        return None
    if isinstance(act_type, str):
        if len(act_type) == 0:
            return None
        act = {
                'relu': nn.ReLU,
                'lrelu': nn.LeakyReLU,
            }[act_type]

    return act


def get_norm(norm_type: str) -> nn.Module:
    if norm_type is None:
        return None
    if isinstance(norm_type, str):
        if len(norm_type) == 0:
            return None
        norm = {
                'BN': nn.BatchNorm2d,
                'IN': nn.InstanceNorm2d,
            }[norm_type]

    return norm



