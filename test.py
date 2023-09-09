import shutil
from pathlib import Path

import cupy
import torch
import torchvision as tv
from thop import profile as ops_profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.viton_dataset import LoadVITONDataset
from pipelines import DMVTONPipeline
from opt.test_opt import TestOptions
from utils.general import Profile, print_log, warm_up
from utils.metrics import calculate_fid_given_paths, calculate_lpips_given_paths
from utils.torch_utils import select_device


def run_test_pf(
    pipeline, data_loader, device, img_dir, save_dir, log_path, save_img=True
):
    metrics = {}

    result_dir = Path(save_dir) / 'results'
    tryon_dir = result_dir / 'tryon'
    visualize_dir = result_dir / 'visualize'
    tryon_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up gpu
    dummy_input = {
        'person': torch.randn(1, 3, 256, 192).to(device),
        'clothes': torch.randn(1, 3, 256, 192).to(device),
        'clothes_edge': torch.randn(1, 1, 256, 192).to(device),
    }
    with cupy.cuda.Device(int(device.split(':')[-1])):
        warm_up(pipeline, **dummy_input)

    with torch.no_grad():
        seen, dt = 0, Profile(device=device)

        for idx, data in enumerate(tqdm(data_loader)):
            # Prepare data
            real_image = data['image'].to(device)
            clothes = data['color'].to(device)
            edge = data['edge'].to(device)

            with cupy.cuda.Device(int(device.split(':')[-1])):
                with dt:
                    p_tryon, warped_cloth = pipeline(real_image, clothes, edge, phase="test")

            seen += len(p_tryon)

            # Save images
            for j in range(len(data['p_name'])):
                p_name = data['p_name'][j]

                tv.utils.save_image(
                    p_tryon[j],
                    tryon_dir / p_name,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1, 1),
                )

                combine = torch.cat(
                    [real_image[j].float(), clothes[j], warped_cloth[j], p_tryon[j]], -1
                ).squeeze()
                tv.utils.save_image(
                    combine,
                    visualize_dir / p_name,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1, 1),
                )

    fid = calculate_fid_given_paths(
        paths=[str(img_dir), str(tryon_dir)],
        batch_size=50,
        device=device,
    )
    lpips = calculate_lpips_given_paths(paths=[str(img_dir), str(tryon_dir)], device=device)

    # FID
    metrics['fid'] = fid
    metrics['lpips'] = lpips

    # Speed
    t = dt.t / seen * 1e3  # speeds per image
    metrics['fps'] = 1000 / t
    print_log(
        log_path,
        f'Speed: %.1fms per image {real_image.size()}'
        % t,
    )

    # Memory
    mem_params = sum([param.nelement()*param.element_size() for param in pipeline.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in pipeline.buffers()])
    metrics['mem'] = mem_params + mem_bufs # in bytes

    ops, params = ops_profile(pipeline, (*dummy_input.values(), ), verbose=False)
    metrics['ops'] = ops
    metrics['params'] = params

    # Log
    metrics_str = 'Metric, {}'.format(', '.join([f'{k}: {v}' for k, v in metrics.items()]))
    print_log(log_path, metrics_str)

    # Remove results if not save
    if not save_img:
        shutil.rmtree(result_dir)
    else:
        print_log(log_path, f'Results are saved at {result_dir}')

    return metrics


def main(opt):
    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)
    log_path = Path(opt.save_dir) / 'log.txt'

    # Inference Pipeline
    pipeline = DMVTONPipeline(
        align_corners=opt.align_corners,
        checkpoints={
            'warp': opt.pf_warp_checkpoint,
            'gen': opt.pf_gen_checkpoint,
        },
    ).to(device)
    pipeline.eval()
    print_log(log_path, f'Load pretrained parser-free warp from {opt.pf_warp_checkpoint}')
    print_log(log_path, f'Load pretrained parser-free gen from {opt.pf_gen_checkpoint}')

    # Dataloader
    test_data = LoadVITONDataset(path=opt.dataroot, phase='test', size=(256, 192))
    data_loader = DataLoader(
        test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers
    )

    run_test_pf(
        pipeline=pipeline,
        data_loader=data_loader,
        device=device,
        log_path=log_path,
        save_dir=opt.save_dir,
        img_dir=Path(opt.dataroot) / 'test_img',
        save_img=True,
    )


if __name__ == "__main__":
    opt = TestOptions().parse_opt()
    main(opt)