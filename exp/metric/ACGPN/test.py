import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import numpy as np
import torch
from torch.autograd import Variable
import torchvision as tv


SIZE = 320
NC = 14


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((old_label.cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((old_label.cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((old_label.cpu().numpy() == 7).astype(np.int))
    label = label*(1-arm1)+arm1*4
    label = label*(1-arm2)+arm2*4
    label = label*(1-noise)+noise*4
    return label
            

def run_test(dataroot, save_dir, batch_size, device):
    opt = TestOptions().parse()
    opt.dataroot = dataroot
    opt.batchSize = batch_size
    opt.checkpoints_dir = 'ACGPN/checkpoints'
    opt.resize_or_crop = 'none'

    tryon_dir = Path(save_dir) / 'tryon'
    tryon_dir.mkdir(parents=True, exist_ok=True)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt, device)
    model.eval()

    for i, data in enumerate(dataset):
        # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor(
            (data['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor(
            (data['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = data['image'] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data['label'])

        ############## Forward Pass ######################
        fake_image, warped_cloth, refined_cloth = model(Variable(data['label'].to(device)), Variable(data['edge'].to(device)), Variable(img_fore.to(device)), Variable(
            mask_clothes.to(device)), Variable(data['color'].to(device)), Variable(all_clothes_label.to(device)), Variable(data['image'].to(device)), Variable(data['pose'].to(device)), Variable(data['image'].to(device)), Variable(mask_fore.to(device)))

        # save output
        for j in range(len(data['name'])):
            util.save_tensor_as_image(fake_image[j], tryon_dir / data['name'][j])
            