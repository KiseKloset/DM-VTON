from pathlib import Path
from tqdm import tqdm

import cupy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.viton_dataset import LoadVITONDataset
from models.pfafn.afwm_test import AFWM
from models.mobile_unet_generator import MobileNetV2_unet
from opt.test_opt import TestOptions
from utils.torch_utils import smart_pretrained, select_device
from utils.general import Profile, warm_up, print_log
from utils.metrics import PytorchFID


def run_val_pf(models, val_loader, align_corners, device, log_path):
    warp_model, gen_model = models['warp'], models['gen']
    metrics = {}
    pytorch_fid = PytorchFID()

    # Warm-up gpu
    dummy_input = torch.randn(1, 7, 256, 192, dtype=torch.float).to(device)
    warm_up(gen_model, dummy_input)

    # Validate
    real_imgs = []
    tryon_imgs = []
    with torch.no_grad():
        seen, dt = 0, (Profile(device=device), Profile(device=device), Profile(device=device))

        for idx, data in enumerate(tqdm(val_loader)):
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
            real_imgs.append((real_image + 1) / 2)
            tryon_imgs.append((p_tryon + 1) / 2)
            # real_imgs.append(make_grid(real_image, normalize=True, value_range=(-1,1)))
            # tryon_imgs.append(make_grid(p_tryon, normalize=True, value_range=(-1,1)))

            # import torchvision as tv
            # p_name = f'{idx}.jpg'
            # tv.utils.save_image(
            #     p_tryon,
            #     Path('runs/test/b') / p_name,
            #     nrow=int(1),
            #     normalize=True,
            #     value_range=(-1,1),
            # )
        
    fid = metrics.pytorch_fid.compute_fid(real_imgs, seen, tryon_imgs, seen)
    
    # FID
    metrics['fid'] = fid

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

    # Model
    from models.afwm_test import AFWM as FSAFWM
    from models.networks import ResUnetGenerator
    warp_model = FSAFWM(3, opt.align_corners)
    warp_model.eval()
    warp_model.to(device)
    smart_pretrained(warp_model, opt.pf_warp_checkpoint)
    # gen_model = MobileNetV2_unet(7, 4)
    gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
    gen_model.eval()
    gen_model.to(device)
    smart_pretrained(gen_model, opt.pf_gen_checkpoint)
    
    # Dataloader
    val_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    log_path = Path(opt.save_dir) / 'log.txt'

    run_val_pf(
        models={'warp': warp_model, 'gen': gen_model},
        val_loader=val_loader,
        align_corners=opt.align_corners,
        device=device,
        log_path=log_path,
    )


if __name__ == "__main__":
    opt = TestOptions().parse_opt()
    main(opt)