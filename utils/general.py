import contextlib
import time

import torch
import yaml
from pathlib import Path


class Profile(contextlib.ContextDecorator):
    """
    YOLOv5 Profile class. 
    Usage: 'with Profile():' or '@Profile()' decorator
    """

    def __init__(self, device=None, t=0.0):
        self.device = device
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
            torch.cuda.synchronize(self.device)
        return time.time()
    

class AverageMeter:
    """
    Computes and stores the average and current value
    Source: https://github.dev/PaddlePaddle/PaddleOCR/ppocr/utils/utility.py
    """
    
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


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def increment_path(path: str, exist_ok: bool = False, sep: str = '') -> Path:
    """
    Increments a path by adding a number to the end if it already exists.

    Args:
      path (str): Path to increment.
      exist_ok (bool): If True, the path will not be incremented and returned as-is.
      sep (str): Separator to use between the path and the incrementation number.

    Returns:
      Incremented path.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(1, 999):
            p = f'{path}{sep}{n}{suffix}'
            if not Path(p).exists():
                path = Path(p)
                break

    return path


def warm_up(model, **dummy_input):
    with torch.no_grad():
        for _ in range(10):
            _ = model(**dummy_input)


def print_log(log_path, content, to_print=True):
    with open(log_path, 'a') as f:
      f.write(content)
      f.write('\n')

    if to_print:
        print(content)

    