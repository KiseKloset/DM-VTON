import torch


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))
