import os

import torch


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    # model.cuda(model.opt.gpu_ids[0])


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("Did not found checkpoint at", checkpoint_path)
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
