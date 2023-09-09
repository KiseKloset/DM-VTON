import torch.nn as nn


class BaseVTONPipeline(nn.Module):
    """
    Base class for pipeline
    """

    def __init__(self, checkpoints=None, **kargs):
        super().__init__()
