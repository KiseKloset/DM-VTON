import shutil
from pathlib import Path
from tqdm import tqdm

import cupy
import torch
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.viton_dataset import LoadVITONDataset
from data.dresscode_dataset import DressCodeDataset
from models.pfafn.afwm_test import AFWM
#from models.afwm_test import AFWM as FSAFWM
from models.mobile_unet_generator import MobileNetV2_unet
from models.networks import ResUnetGenerator
from opt.test_opt import TestOptions
from utils.torch_utils import select_device, get_ckpt, load_ckpt
from utils.general import Profile, warm_up, print_log
from utils.metrics.pytorch_fid.fid_score import calculate_fid_given_paths
from utils.metrics.lpips.lpips import calculate_lpips_given_paths


def run_test_pf(models, data_loader, align_corners, device, img_dir, save_dir, log_path, save_img=True):
    warp_model, gen_model = models['warp'], models['gen']
    metrics = {}

    tryon_dir = Path(save_dir) / 'results' / 'tryon'
    visualize_dir = Path(save_dir) / 'results' / 'visualize'
    tryon_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up gpu
    dummy_input = torch.randn(1, 7, 256, 192, dtype=torch.float).to(device)
    warm_up(gen_model, dummy_input)

    # testidate
    with torch.no_grad():
        seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))

        for idx, data in enumerate(tqdm(data_loader)):
            # Prepare data
            with dt[0]:
                real_image = data['image'].to(device)
                clothes = data['color'].to(device)
                edge = data['edge'].to(device)
                edge = (edge > 0.5).float()
                clothes = clothes * edge  
            
            # Warp
            with dt[1]:
                with cupy.cuda.Device(int(device.split(':')[-1])):
                    flow_out = warp_model(real_image, clothes)
                    warped_cloth, last_flow, = flow_out
                    warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros', align_corners=align_corners)
            
            # Gen
            with dt[2]:
                gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)                
                gen_outputs = gen_model(gen_inputs)
                p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                p_rendered = torch.tanh(p_rendered)
                m_composite = torch.sigmoid(m_composite)
                m_composite = m_composite * warped_edge
                p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

            seen += len(p_tryon)
            
            # Save images
            for j in range(len(data['p_name'])):
                p_name = data['p_name'][j]
                c_name = data['c_name'][j]

                tv.utils.save_image(
                    p_tryon[j],
                    tryon_dir / p_name,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1,1),
                )

                combine = torch.cat([real_image[j].float(), clothes[j], warped_cloth[j], p_tryon[j]], -1).squeeze()
                tv.utils.save_image(
                    combine,
                    visualize_dir / p_name,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1,1),
                )
        
    fid = calculate_fid_given_paths(
        paths=[str(img_dir), str(tryon_dir)],
        batch_size=50,
        device=device,
    )
    lpips = calculate_lpips_given_paths(paths=[str(img_dir), str(tryon_dir)], device=device)

    if not save_img:
        shutil.rmtree(Path(save_dir) / 'results')

    # FID
    metrics['fid'] = fid
    metrics['lpips'] = lpips

    # Speed
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    t = (sum(t), ) + t
    metrics['fps'] = 1000 / sum(t[1:]) # Data loading time is not included
    print_log(log_path, f'Speed: %.1fms all, %.1fms pre-process, %.1fms warp, %.1fms gen per image at shape {real_image.size()}' % t)

    # Log
    metrics_str = 'Metric, {}'.format(', '.join(['{}: {}'.format(k, v) for k, v in metrics.items()]))
    print_log(log_path, metrics_str)
    
    return metrics


def main(opt):
    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)
    log_path = Path(opt.save_dir) / 'log.txt'
    
    # Model
    warp_model = AFWM(3, opt.align_corners).to(device)
    warp_model.eval()
    warp_ckpt = get_ckpt(opt.pf_warp_checkpoint)
    load_ckpt(warp_model, warp_ckpt)
    print_log(log_path, f'Load pretrained parser-free warp from {opt.pf_warp_checkpoint}')
    # from SRMGNLastHope.models.mobile_unet_generator import MobileNetV2_unet as LMobileNetV2_unet
    gen_model = MobileNetV2_unet(7, 4).to(device)
    gen_model.eval()
    gen_ckpt = get_ckpt(opt.pf_gen_checkpoint)
    load_ckpt(gen_model, gen_ckpt)
    print_log(log_path, f'Load pretrained parser-free gen from {opt.pf_gen_checkpoint}')
    
    # Dataloader
    test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    # test_data = DressCodeDataset(dataroot_path=opt.dataroot, phase='test', category=['upper_body'])
    data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    run_test_pf(
        models={'warp': warp_model, 'gen': gen_model},
        data_loader=data_loader,
        align_corners=opt.align_corners,
        device=device,
        log_path=log_path,
        save_dir=opt.save_dir,
        img_dir=Path(opt.dataroot) / 'test_img',
        save_img=True,
    )


if __name__ == "__main__":
    opt = TestOptions().parse_opt()
    main(opt)