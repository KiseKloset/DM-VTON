import os

import torch

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

def load_checkpoint(model, checkpoint_path, device):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]

    model.load_state_dict(checkpoint_new)

def load_checkpoint_parallel(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)

def load_checkpoint_part_parallel(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return
    checkpoint = torch.load(checkpoint_path,map_location=device)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        if 'cond_' not in param and 'aflow_net.netRefine' not in param or 'aflow_net.cond_style' in param:
            checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)