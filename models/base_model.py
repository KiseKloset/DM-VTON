from abc import abstractmethod

import numpy as np
import torch.nn as nn


# TODO: Add init_weights function to base model
class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'

    # def init_weights(self, init_type='xavier', gain=0.02):
    #     from torch.nn import init
    #     def init_func(m):
    #         classname = m.__class__.__name__
    #         if classname.find('BatchNorm2d') != -1:
    #             print('!!!!!!!!!!!! Found BN !!!!!!!!!!!!!!!!')
    #             if hasattr(m, 'weight') and m.weight is not None:
    #                 init.normal_(m.weight.data, 1.0, gain)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)
    #         elif (hasattr(m, 'weight') \
    #             and (classname.find('Conv') != -1 or classname.find('Linear') != -1)\
    #         ):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, gain)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=gain)
    #             elif init_type == 'xavier_uniform':
    #                 init.xavier_uniform_(m.weight.data, gain=1.0)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=gain)
    #             elif init_type == 'none':  # Pytorch's default init method
    #                 m.reset_parameters()
    #             else:
    #                 raise NotImplementedError(f'Init method {init_type} is not implemented')
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)

    #     self.apply(init_func)
    #     # propagate to children
