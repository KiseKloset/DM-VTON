import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader

from .data.viton_dataset import LoadVITONDataset
from .models.afwm import AFWM
from .models.networks import ResUnetGenerator, load_checkpoint
from .options.test_options import TestOptions


def run_test(dataroot, save_dir, batch_size, device):
    opt = TestOptions().parse()
    opt.dataroot = dataroot
    opt.batchSize = batch_size
    opt.warp_checkpoint = 'FS_VTON/checkpoints/non_aug/PFAFN_warp_epoch_101.pth'
    opt.gen_checkpoint = 'FS_VTON/checkpoints/non_aug/PFAFN_gen_epoch_101.pth'

    tryon_dir = Path(save_dir) / 'tryon'
    tryon_dir.mkdir(parents=True, exist_ok=True)

    test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    data_loader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=16)

    warp_model = AFWM(opt, 3)
    warp_model.eval()
    warp_model.to(device)
    load_checkpoint(warp_model, opt.warp_checkpoint)

    gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    gen_model.eval()
    gen_model.to(device)
    load_checkpoint(gen_model, opt.gen_checkpoint)

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            real_image = data['image']
            clothes = data['color']
            ##edge is extracted from the clothes image with the built-in function in python
            edge = data['edge']
            edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
            clothes = clothes * edge        

            #; ipdb.set_trace()

            flow_out = warp_model(real_image.to(device), clothes.to(device))
            warped_cloth, last_flow, = flow_out
            warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                            mode='bilinear', padding_mode='zeros', align_corners=True)

            gen_inputs = torch.cat([real_image.to(device), warped_cloth, warped_edge], 1)

            gen_outputs = gen_model(gen_inputs)

            p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            m_composite = m_composite * warped_edge
            p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            # Save images
            for j in range(len(data['p_name'])):
                p_name = data['p_name'][j]
                tv.utils.save_image(
                    p_tryon[j],
                    tryon_dir / p_name,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1,1),
                )
