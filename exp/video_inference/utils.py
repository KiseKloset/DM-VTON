import contextlib
import time
from PIL import Image

import torch
import torchvision.transforms as transforms


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


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    #flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}

    
def get_transform(method=Image.BICUBIC, normalize=True):
    transform_list = []
    base = float(2 ** 4)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def blending_fn(prev_mask, new_mask):
    c1 = 5.68842
    c2 = -0.748699
    c3 = -57.8051
    c4 = 291.309
    c5 = -624.717
    t = new_mask - 0.5
    x = t * t

    i = torch.ones_like(new_mask)
    uncertainty = i - torch.minimum(i, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))))

    return new_mask + (prev_mask - new_mask) * (uncertainty * 1)