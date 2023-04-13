import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from models.afwm import AFWM, TVLoss 
from models.networks import ResUnetGenerator, VGGLoss
from models.rmgn_generator import RMGNGenerator
from options.train_options import TrainOptions
from utils.utils import load_checkpoint_parallel


def CreateDataset(opt):
    #training with augumentation
    #from data.aligned_dataset import AlignedDataset_aug
    #dataset = AlignedDataset_aug()
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset

opt = TrainOptions().parse()

device = torch.device(f'cuda:{opt.gpu_ids[0]}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=16, pin_memory=True)

# TUNGPNT2
# PF_warp_model = AFWM(opt, 3)
# PF_warp_model.eval()
# PF_warp_model.to(device)
# load_checkpoint_parallel(PF_warp_model, opt.PFAFN_warp_checkpoint, device)

# PF_gen_model = RMGNGenerator(multilevel=False, predmask=True)
# PF_gen_model.eval()
# PF_gen_model.to(device)
# load_checkpoint_parallel(PF_gen_model, opt.PFAFN_gen_checkpoint, device)

# PB_warp_model = AFWM(opt, 45)
# PB_warp_model.eval()
# PB_warp_model.to(device)
# load_checkpoint_parallel(PB_warp_model, opt.PBAFN_warp_checkpoint, device)

# PB_gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
# PB_gen_model.eval()
# PB_gen_model.to(device)
# load_checkpoint_parallel(PB_gen_model, opt.PBAFN_gen_checkpoint,  device)


path = os.path.join('artifact/', opt.name)
os.makedirs(path, exist_ok=True)
i = 0
with torch.no_grad():
    for idx, data in enumerate(tqdm(train_loader)):
        i += 1
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        edge_un = data['edge_un']
        pre_clothes_edge_un = torch.FloatTensor((edge_un.detach().numpy() > 0.5).astype(np.int64))
        clothes_un = data['color_un']
        clothes_un = clothes_un * pre_clothes_edge_un
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
        densepose_fore = data['densepose'] / 24
        face_mask = torch.FloatTensor((data['label'].cpu().numpy() == 1).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int64))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int64)) \
                                + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int64)) \
                                + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int64))
        face_img = face_mask * real_image
        other_clothes_img = other_clothes_mask * real_image
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

        # concat_un = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
        # flow_out_un = PB_warp_model(concat_un.to(device), clothes_un.to(device), pre_clothes_edge_un.to(device))
        # warped_cloth_un, last_flow_un, cond_un_all, flow_un_all, delta_list_un, x_all_un, x_edge_all_un, delta_x_all_un, delta_y_all_un = flow_out_un
        # warped_prod_edge_un = F.grid_sample(pre_clothes_edge_un.to(device), last_flow_un.permute(0, 2, 3, 1),
        #                                     mode='bilinear', padding_mode='zeros', align_corners=opt.align_corners)

        # flow_out_sup = PB_warp_model(concat_un.to(device), clothes.to(device), pre_clothes_edge.to(device))
        # warped_cloth_sup, last_flow_sup, cond_sup_all, flow_sup_all, delta_list_sup, x_all_sup, x_edge_all_sup, delta_x_all_sup, delta_y_all_sup = flow_out_sup

        arm_mask = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.float64)) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 3).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64)) \
                                + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64)) \
                                + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64)) \
                                + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 22))
        hand_img = (arm_mask * hand_mask) * real_image
        dense_preserve_mask = dense_preserve_mask.to(device) * (1 - person_clothes_edge.to(device))
        preserve_region = face_img + other_clothes_img + hand_img

        # gen_inputs_un = torch.cat([preserve_region.to(device), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1)
        # gen_outputs_un = PB_gen_model(gen_inputs_un)
        # p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
        # p_rendered_un = torch.tanh(p_rendered_un)
        # m_composite_un = torch.sigmoid(m_composite_un)
        # m_composite_un = m_composite_un * warped_prod_edge_un
        # p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

        # flow_out = PF_warp_model(p_tryon_un.detach(), clothes.to(device), pre_clothes_edge.to(device))
        # warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        # warped_prod_edge = x_edge_all[4]

        # skin_mask = warped_prod_edge_un.detach() * (1 - person_clothes_edge.to(device))

        # # gen_inputs = torch.cat([p_tryon_un.detach(), warped_cloth, warped_prod_edge], 1)
        # gen_inputs_clothes = torch.cat([warped_cloth, warped_prod_edge], 1)
        # gen_inputs_persons = p_tryon_un.detach()
        
        # gen_outputs, out_L1, out_L2, M_list = PF_gen_model(gen_inputs_persons, gen_inputs_clothes)

        # p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        # p_rendered = torch.tanh(p_rendered)
        # m_composite = torch.sigmoid(m_composite)
        # m_composite1 = m_composite * warped_prod_edge
        # m_composite = person_clothes_edge.to(device) * m_composite1
        # p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        ############## Display results ##########
        # a = real_image.float().to(device)
        # b = clothes.to(device)
        # c = person_clothes.to(device)
        # d = clothes_un.float().to(device)
        # e = warped_cloth_un
        # f = p_tryon_un.detach()
        # g = warped_cloth
        # h = torch.cat([skin_mask.to(device), skin_mask.to(device), skin_mask.to(device)], 1)
        # i = p_rendered
        # j = torch.cat([m_composite1, m_composite1, m_composite1], 1)
        # k = p_tryon
        # combine = torch.cat([a, b, c, d, e, f, g, h, i, j, k], -1).squeeze()

        a = real_image.to(device)
        b = person_clothes_edge.to(device).expand(-1, 3, -1, -1)
        c = person_clothes.to(device)
        d = face_img.to(device)
        e = preserve_region.to(device)
        f = dense_preserve_mask.to(device).expand(-1, 3, -1, -1)
        combine = torch.cat([a, b, c, d, e, f], -1).squeeze()

        utils.save_image(
            combine,
            os.path.join(path, f'{idx}.jpg'),
            nrow=1,
            normalize=True,
            value_range=(-1,1),
        )

        if i == 10:
            break

        # cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        # rgb = (cv_img * 255).astype(np.uint8)
        # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(path, f'{idx}.jpg'), bgr)