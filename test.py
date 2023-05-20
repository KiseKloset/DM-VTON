from pathlib import Path
from tqdm import tqdm

import cupy
import torch
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.viton_dataset import LoadVITONDataset
from models.pfafn.afwm_test import AFWM
from models.mobile_unet_generator import MobileNetV2_unet
from opt.test_opt import TestOptions
from utils.torch_utils import smart_pretrained, select_device
from utils.general import Profile, warm_up


def run_test(models, test_loader, align_corners, device, save_dir):
    warp_model, gen_model = models['warp'], models['gen']

    tryon_dir = Path(save_dir) / 'results' / 'tryon'
    visualize_dir = Path(save_dir) / 'results' / 'visualize'
    tryon_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up gpu
    dummy_input = torch.randn(1, 7, 256, 192, dtype=torch.float).to(device)
    warm_up(gen_model, dummy_input)

    # testidate
    with torch.no_grad():
        seen, dt = 0, (Profile(), Profile(), Profile())

        for idx, data in enumerate(tqdm(test_loader)):
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
            p_name = f'{idx}.jpg'

            tv.utils.save_image(
                p_tryon,
                tryon_dir / p_name,
                nrow=int(1),
                normalize=True,
                value_range=(-1,1),
            )

            combine = torch.cat([real_image.float(), clothes, warped_cloth, p_tryon], -1).squeeze()
            tv.utils.save_image(
                combine,
                visualize_dir / p_name,
                nrow=int(1),
                normalize=True,
                value_range=(-1,1),
            )

    # Speed
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    t = (sum(t), ) + t
    fps = 1000 / sum(t[1:]) # Data loading time is not included
    print(f'FPS: %.1f' % fps)
    print(f'Speed: %.1fms all, %.1fms pre-process, %.1fms warp, %.1fms gen per image at shape {real_image.size()}' % t)


def main(opt):
    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Model
    warp_model = AFWM(3, opt.align_corners)
    warp_model.eval()
    warp_model.to(device)
    smart_pretrained(warp_model, opt.pf_warp_checkpoint)
    gen_model = MobileNetV2_unet(7, 4)
    gen_model.eval()
    gen_model.to(device)
    smart_pretrained(gen_model, opt.pf_gen_checkpoint)
    
    # Dataloader
    test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    run_test(
        models={'warp': warp_model, 'gen': gen_model},
        test_loader=test_loader,
        align_corners=opt.align_corners,
        device=device,
        save_dir=opt.save_dir,
    )


if __name__ == "__main__":
    opt = TestOptions().parse_opt()
    main(opt)
    # python -m pytorch_fid runs/test/a-1/results/tryon ../dataset/Merge-VITON-V1/VITON_test_forward/test_img --device cuda:1