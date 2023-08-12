import os

import torch


# def select_device(device='', batch_size=0):
#     # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
#     device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
#     cpu = device == 'cpu'
#     if cpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
#     elif device:  # non-cpu device requested
#         os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
#         assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
#             f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

#     if not cpu and torch.cuda.is_available():  # prefer GPU if available
#         devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
#         n = len(devices)  # device count
#         if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
#             assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
#         arg = 'cuda:0'
#     else:  # revert to CPU
#         arg = 'cpu'

#     return torch.device(arg)


def select_device(device='', batch_size=0):
    cpu = device == 'cpu'
    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        arg = f'cuda:{device}'
    else:  # revert to CPU
        arg = 'cpu'

    # return torch.device(arg)
    return arg


def get_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu') if ckpt_path else None
    return ckpt


def load_ckpt(model, ckpt=None):
    if ckpt is not None:
        ckpt_new = model.state_dict()
        pretrained = ckpt.get('model') if ckpt.get('model') else ckpt
        for param in ckpt_new:
            ckpt_new[param] = pretrained[param]
        model.load_state_dict(ckpt_new)


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9):
    params = [p for p in model.parameters()]

    if name == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    return optimizer


def smart_resume(ckpt, optimizer, ckpt_path, epoch_num):
    # Resume training from a partially trained checkpoint
    best_fid = float('inf')
    start_epoch = ckpt.get('epoch') + 1 if  ckpt.get('epoch') else 1
    if ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
        if ckpt.get('best_fid') is not None:
            best_fid = ckpt['best_fid']
    print(f'Resume training from {ckpt_path} from epoch {start_epoch} to {epoch_num} total epochs')

    return start_epoch, best_fid

