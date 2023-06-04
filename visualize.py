from pathlib import Path

import numpy as np
import torch
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.pfafn.afwm import AFWM 
from models.afwm_pb import AFWM as PBAFWM
from models.mobile_unet_generator import MobileNetV2_unet 
from models.networks import ResUnetGenerator
from opt.train_opt import TrainOptions
from utils.torch_utils import select_device, smart_pretrained
from utils.general import AverageMeter, print_log
from utils.lr_utils import MyLRScheduler
from data.dresscode_dataset import DressCodeDataset
from data.viton_dataset import LoadVITONDataset


def visualize_teacher(models, data_loader, device, save_dir):
    pb_warp_model, pb_gen_model, pf_warp_model, pf_gen_model \
        = models['pb_warp'], models['pb_gen'], models['pf_warp'], models['pf_gen']
    
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, data in enumerate(data_loader):
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

        concat_un = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
        flow_out_un = pb_warp_model(concat_un.to(device), clothes_un.to(device), pre_clothes_edge_un.to(device))
        warped_cloth_un, last_flow_un, cond_un_all, flow_un_all, delta_list_un, x_all_un, x_edge_all_un, delta_x_all_un, delta_y_all_un = flow_out_un
        warped_prod_edge_un = F.grid_sample(pre_clothes_edge_un.to(device), last_flow_un.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros', align_corners=opt.align_corners)

        flow_out_sup = pb_warp_model(concat_un.to(device), clothes.to(device), pre_clothes_edge.to(device))
        warped_cloth_sup, last_flow_sup, cond_sup_all, flow_sup_all, delta_list_sup, x_all_sup, x_edge_all_sup, delta_x_all_sup, delta_y_all_sup = flow_out_sup

        arm_mask = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.float64)) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 3).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64)) \
                                + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64)) \
                                + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64)) \
                                + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 22))
        hand_img = (arm_mask * hand_mask) * real_image
        dense_preserve_mask = dense_preserve_mask.to(device) * (1 - warped_prod_edge_un)
        preserve_region = face_img + other_clothes_img + hand_img

        gen_inputs_un = torch.cat([preserve_region.to(device), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1)
        gen_outputs_un = pb_gen_model(gen_inputs_un)
        p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
        p_rendered_un = torch.tanh(p_rendered_un)
        m_composite_un = torch.sigmoid(m_composite_un)
        m_composite_un = m_composite_un * warped_prod_edge_un
        p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

        flow_out = pf_warp_model(p_tryon_un.detach(), clothes.to(device), pre_clothes_edge.to(device))
        warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        gen_inputs = torch.cat([p_tryon_un.detach(), warped_cloth, warped_prod_edge], 1)
        gen_outputs = pf_gen_model(gen_inputs)
        # gen_inputs_clothes = torch.cat([warped_cloth, warped_prod_edge], 1)
        # gen_inputs_persons = p_tryon_un.detach()
        # gen_outputs, out_L1, out_L2, M_list = pf_gen_model(gen_inputs_persons, gen_inputs_clothes)

        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_prod_edge
        # TUNGPNT2
        # m_composite =  person_clothes_edge.to(device)*m_composite
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        # Visualize
        p_name = f'{idx}.jpg'
        a = real_image.float().to(device)
        b = person_clothes.to(device)
        c = clothes.to(device)
        d = torch.cat([densepose_fore.to(device),densepose_fore.to(device),densepose_fore.to(device)],1)
        e = warped_cloth
        f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
        g = preserve_region.to(device)
        h = torch.cat([dense_preserve_mask,dense_preserve_mask,dense_preserve_mask],1)
        i = p_rendered
        j = torch.cat([m_composite, m_composite, m_composite], 1)
        k = p_tryon
        combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0],g[0],h[0],i[0],j[0],k[0]], 2).squeeze()
        tv.utils.save_image(
            combine,
            save_dir / p_name,
            nrow=int(1),
            normalize=True,
            value_range=(-1,1),
        )
        tv.utils.save_image(
            combine,
            save_dir / f'{p_name}_img',
            nrow=int(1),
            normalize=True,
            value_range=(-1,1),
        )
        tv.utils.save_image(
            combine,
            save_dir / f'{p_name}_parse',
            nrow=int(1),
            normalize=True,
            value_range=(-1,1),
        )
        tv.utils.save_image(
            combine,
            save_dir / f'{p_name}_pose',
            nrow=int(1),
            normalize=True,
            value_range=(-1,1),
        )
    

def main(opt):
    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Model
    warp_model = PBAFWM(45, opt.align_corners)
    warp_model.eval()
    warp_model.to(device)
    smart_pretrained(warp_model, opt.pf_warp_checkpoint)
    gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
    gen_model.eval()
    gen_model.to(device)
    smart_pretrained(gen_model, opt.pf_gen_checkpoint)
    
    # Dataloader
    # train_data = DressCodeDataset(dataroot_path=opt.dataroot, phase='train', category=['upper_body'])
    train_data = LoadVITONDataset(path=opt.dataroot, phase='train', size=(256, 192))
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    log_path = Path(opt.save_dir) / 'log.txt'

    visualize_teacher(
        models={'warp': warp_model, 'gen': gen_model},
        data_loader=train_loader,
        device=device,
        save_dir=Path(opt.save_dir) / 'results'
    )

if __name__ == '__main__' :
    opt = TrainOptions().parse_opt()
    main(opt)