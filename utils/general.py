import contextlib
import os
import time
from PIL import Image, ImageDraw

import torch
import yaml
from pathlib import Path
from torch import Tensor


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()
    

# https://github.dev/PaddlePaddle/PaddleOCR/ppocr/utils/utility.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def keypoints2posemaps(keypoints: Tensor, size: list = [256, 192], radius: int = 5) -> list[Image.Image]:
    h, w = size
    num_point = keypoints.shape[0]
    pose_map = []
    img_pose = Image.new('L', (w, h))

    pose_draw = ImageDraw.Draw(img_pose)
    for i in range(num_point):
        one_map = Image.new('L', (w, h))
        draw = ImageDraw.Draw(one_map)
        pointx = keypoints[i, 0]
        pointy = keypoints[i, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            pose_draw.rectangle((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
        one_map = one_map.convert('RGB')
        pose_map.append(one_map)
    return pose_map


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path