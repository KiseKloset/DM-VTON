import contextlib
import time

import torch


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
