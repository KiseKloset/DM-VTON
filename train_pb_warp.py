import datetime
import time
from pathlib import Path

import cupy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataloader.viton_dataset import LoadVITONDataset
from losses import TVLoss, VGGLoss
from models.warp_modules.style_afwm import StyleAFWM as PBAFWM
from opt.train_opt import TrainOptions
from utils.general import AverageMeter, print_log
from utils.lr_utils import MyLRScheduler
from utils.torch_utils import get_ckpt, load_ckpt, select_device, smart_optimizer, smart_resume


def train_batch(
    data, models, optimizers, criterions, device, writer, global_step, sample_step, samples_dir
):
    batch_start_time = time.time()

    warp_model = models['warp']
    warp_optimizer = optimizers['warp']
    criterionL1, criterionVGG = criterions['L1'], criterions['VGG']

    t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
    data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
    edge = data['edge']
    pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
    clothes = data['color']
    clothes = clothes * pre_clothes_edge
    person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
    real_image = data['image']
    person_clothes = real_image * person_clothes_edge
    pose = data['pose']
    size = data['label'].size()
    oneHot_size1 = (size[0], 25, size[2], size[3])
    densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1), device=device).zero_()
    densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
    densepose_fore = data['densepose'] / 24.0
    face_mask = torch.FloatTensor(
        (data['label'].cpu().numpy() == 1).astype(np.int64)
    ) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int64))
    other_clothes_mask = (
        torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int64))
        + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int64))
    )
    preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)
    concat = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)

    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out = warp_model(concat.to(device), clothes.to(device), pre_clothes_edge.to(device))
    (
        warped_cloth,
        last_flow,
        cond_fea_all,
        warp_fea_all,
        flow_all,
        delta_list,
        x_all,
        x_edge_all,
        delta_x_all,
        delta_y_all,
    ) = flow_out
    warped_prod_edge = x_edge_all[4]

    epsilon = 0.001
    loss_smooth = sum([TVLoss(x) for x in delta_list])
    # if (global_step % 300 == 0):
    #     print_log(log_path, f'smooth: {str(loss_smooth)}')
    loss_all = 0

    for num in range(5):
        cur_person_clothes = F.interpolate(
            person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear'
        )
        cur_person_clothes_edge = F.interpolate(
            person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear'
        )
        loss_l1 = criterionL1(x_all[num], cur_person_clothes.to(device))
        loss_vgg = criterionVGG(x_all[num], cur_person_clothes.to(device))
        loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.to(device))
        b, c, h, w = delta_x_all[num].shape
        loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
        loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
        loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
        loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
        loss_second_smooth = loss_flow_x + loss_flow_y
        loss_all = (
            loss_all
            + (num + 1) * loss_l1
            + (num + 1) * 0.2 * loss_vgg
            + (num + 1) * 2 * loss_edge
            + (num + 1) * 6 * loss_second_smooth
        )

    loss_all = 0.01 * loss_smooth + loss_all

    warp_optimizer.zero_grad()
    loss_all.backward()
    warp_optimizer.step()

    train_batch_time = time.time() - batch_start_time

    # Visualize
    if global_step % sample_step == 0:
        # Tensorboard
        a = real_image.float().to(device)
        b = person_clothes.to(device)
        c = clothes.to(device)
        d = torch.cat(
            [densepose_fore.to(device), densepose_fore.to(device), densepose_fore.to(device)], 1
        )
        e = warped_cloth
        f = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
        combine = torch.cat([a[0], b[0], c[0], d[0], e[0], f[0]], 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        writer.add_image('combine', (combine.data + 1) / 2.0, global_step)
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(samples_dir / f'{global_step}.jpg'), bgr)

    return loss_all.item(), train_batch_time


def train_pb_warp(opt):
    epoch_num = opt.niter + opt.niter_decay
    writer = SummaryWriter(opt.save_dir)

    # Directories
    log_path = Path(opt.save_dir) / 'log.txt'
    weights_dir = Path(opt.save_dir) / 'weights'  # weights dir
    samples_dir = Path(opt.save_dir) / 'samples'  # samples dir
    weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
    samples_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Model
    warp_model = PBAFWM(45, opt.align_corners).to(device)
    warp_ckpt = get_ckpt(opt.pb_warp_checkpoint)
    load_ckpt(warp_model, warp_ckpt)
    print_log(log_path, f'Load pretrained parser-based warp from {opt.pb_warp_checkpoint}')

    # Optimizer
    warp_optimizer = smart_optimizer(
        model=warp_model, name=opt.optimizer, lr=opt.lr, momentum=opt.momentum
    )

    # Resume
    if opt.resume:
        if warp_ckpt:
            start_epoch = smart_resume(
                warp_ckpt, warp_optimizer, opt.pb_warp_checkpoint, epoch_num=epoch_num
            )
    else:
        start_epoch = 1

    # Scheduler
    last_epoch = start_epoch - 1
    warp_scheduler = MyLRScheduler(warp_optimizer, last_epoch, opt.niter, opt.niter_decay, False)

    # Dataloader
    train_data = LoadVITONDataset(path=opt.dataroot, phase='train', size=(256, 192))
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers
    )

    # Loss
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(device=device)

    # Start training
    nb = len(train_loader)  # number of batches
    total_steps = epoch_num * nb
    eta_meter = AverageMeter()
    global_step = 1
    t0 = time.time()
    train_loss = 0
    steps_loss = 0

    for epoch in range(start_epoch, epoch_num + 1):
        warp_model.train()
        epoch_start_time = time.time()

        for idx, data in enumerate(train_loader):  # batch -----------------------------------------
            loss_all, train_batch_time = train_batch(
                data,
                models={'warp': warp_model},
                optimizers={'warp': warp_optimizer},
                criterions={'L1': criterionL1, 'VGG': criterionVGG},
                device=device,
                writer=writer,
                global_step=global_step,
                sample_step=opt.sample_step,
                samples_dir=samples_dir,
            )
            train_loss += loss_all
            steps_loss += loss_all

            # Logging
            eta_meter.update(train_batch_time)
            now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
            if global_step % opt.print_step == 0:
                eta_sec = ((epoch_num + 1 - epoch) * len(train_loader) - idx - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = '[{}]: [epoch-{}/{}]--[global_step-{}/{}-{:.2%}]--[loss: warp-{:.6f}]--[lr-{}]--[eta-{}]'.format(  # noqa: E501
                    now,
                    epoch,
                    epoch_num,
                    global_step,
                    total_steps,
                    global_step / total_steps,
                    steps_loss / opt.print_step,
                    ['%.6f' % group['lr'] for group in warp_optimizer.param_groups],
                    eta_sec_format,
                )  # noqa: E501
                print_log(log_path, strs)

                steps_loss = 0

            global_step += 1
            # end batch ---------------------------------------------------------------------------

        # Scheduler
        warp_scheduler.step()

        # Visualize train loss
        train_loss /= len(train_loader)
        writer.add_scalar('train_loss', train_loss, epoch)

        # Save model
        warp_ckpt = {
            'epoch': epoch,
            'model': warp_model.state_dict(),
            'optimizer': warp_optimizer.state_dict(),
        }
        torch.save(warp_ckpt, weights_dir / 'pb_warp_last.pt')
        if epoch % opt.save_period == 0:
            torch.save(warp_ckpt, weights_dir / 'pb_warp_epoch_{epoch}.pt')
            print_log(
                log_path, 'Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps)
            )
        del warp_ckpt

        print_log(
            log_path,
            'End of epoch %d / %d: train_loss: %.3f \t time: %d sec'
            % (epoch, opt.niter + opt.niter_decay, train_loss, time.time() - epoch_start_time),
        )

        train_loss = 0
        # end epoch -------------------------------------------------------------------------------
    # end training --------------------------------------------------------------------------------
    print_log(
        log_path,
        (f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.'),
    )
    print_log(log_path, f'Results are saved at {opt.save_dir}')

    with torch.cuda.device(device):
        torch.cuda.empty_cache()


if __name__ == '__main__':
    opt = TrainOptions().parse_opt()
    train_pb_warp(opt)
