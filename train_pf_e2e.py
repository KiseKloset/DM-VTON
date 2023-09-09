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

import val
from dataloader.viton_dataset import LoadVITONDataset
from losses import TVLoss, VGGLoss
from models.generators.mobile_unet import MobileNetV2_unet
from models.generators.res_unet import ResUnetGenerator
from models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from models.warp_modules.style_afwm import StyleAFWM as PBAFWM
from opt.train_opt import TrainOptions
from utils.general import AverageMeter, print_log
from utils.lr_utils import MyLRScheduler
from utils.torch_utils import get_ckpt, load_ckpt, select_device, smart_optimizer, smart_resume


def train_batch(
    data,
    models,
    optimizers,
    criterions,
    device,
    writer,
    global_step,
    sample_step,
    samples_dir,
    loss_lrdecay=False,
):
    batch_start_time = time.time()

    pb_warp_model, pb_gen_model, pf_warp_model, pf_gen_model = (
        models['pb_warp'],
        models['pb_gen'],
        models['pf_warp'],
        models['pf_gen'],
    )
    warp_optimizer, gen_optimizer = optimizers['warp'], optimizers['gen']
    criterionL1, criterionVGG = criterions['L1'], criterions['VGG']

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
    densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1), device=device).zero_()
    densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
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
    face_img = face_mask * real_image
    other_clothes_img = other_clothes_mask * real_image
    preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

    concat_un = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out_un = pb_warp_model(
            concat_un.to(device), clothes_un.to(device), pre_clothes_edge_un.to(device)
        )
    (
        warped_cloth_un,
        last_flow_un,
        cond_fea_un_all,
        warp_fea_un_all,
        flow_un_all,
        delta_list_un,
        x_all_un,
        x_edge_all_un,
        delta_x_all_un,
        delta_y_all_un,
    ) = flow_out_un
    warped_prod_edge_un = F.grid_sample(
        pre_clothes_edge_un.to(device),
        last_flow_un.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=opt.align_corners,
    )

    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out_sup = pb_warp_model(
            concat_un.to(device), clothes.to(device), pre_clothes_edge.to(device)
        )
    (
        warped_cloth_sup,
        last_flow_sup,
        cond_fea_sup_all,
        warp_fea_sup_all,
        flow_sup_all,
        delta_list_sup,
        x_all_sup,
        x_edge_all_sup,
        delta_x_all_sup,
        delta_y_all_sup,
    ) = flow_out_sup

    arm_mask = torch.FloatTensor(
        (data['label'].cpu().numpy() == 11).astype(np.float64)
    ) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
    hand_mask = torch.FloatTensor(
        (data['densepose'].cpu().numpy() == 3).astype(np.int64)
    ) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
    dense_preserve_mask = (
        torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64))
        + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64))
        + torch.FloatTensor(data['densepose'].cpu().numpy() == 22)
    )
    hand_img = (arm_mask * hand_mask) * real_image
    dense_preserve_mask = dense_preserve_mask.to(device) * (1 - warped_prod_edge_un)
    preserve_region = face_img + other_clothes_img + hand_img

    gen_inputs_un = torch.cat(
        [preserve_region.to(device), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1
    )
    gen_outputs_un = pb_gen_model(gen_inputs_un)
    p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
    p_rendered_un = torch.tanh(p_rendered_un)
    m_composite_un = torch.sigmoid(m_composite_un)
    m_composite_un = m_composite_un * warped_prod_edge_un
    p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

    with cupy.cuda.Device(int(device.split(':')[-1])):
        flow_out = pf_warp_model(
            p_tryon_un.detach(), clothes.to(device), pre_clothes_edge.to(device)
        )
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
    loss_warp = 0
    loss_fea_sup_all = 0
    loss_flow_sup_all = 0

    l1_loss_batch = torch.abs(warped_cloth_sup.detach() - person_clothes.to(device))
    l1_loss_batch = l1_loss_batch.reshape(-1, 3 * 256 * 192)  # opt.batchSize
    l1_loss_batch = l1_loss_batch.sum(dim=1) / (3 * 256 * 192)
    l1_loss_batch_pred = torch.abs(warped_cloth.detach() - person_clothes.to(device))
    l1_loss_batch_pred = l1_loss_batch_pred.reshape(-1, 3 * 256 * 192)  # opt.batchSize
    l1_loss_batch_pred = l1_loss_batch_pred.sum(dim=1) / (3 * 256 * 192)
    weight = (l1_loss_batch < l1_loss_batch_pred).float()
    num_all = len(np.where(weight.cpu().numpy() > 0)[0])
    if num_all == 0:
        num_all = 1

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
        b1, c1, h1, w1 = cond_fea_all[num].shape
        weight_all = weight.reshape(-1, 1, 1, 1).repeat(1, 256, h1, w1)
        cond_sup_loss = (
            (cond_fea_sup_all[num].detach() - cond_fea_all[num]) ** 2 * weight_all
        ).sum() / (256 * h1 * w1 * num_all)
        warp_sup_loss = (
            (warp_fea_sup_all[num].detach() - warp_fea_all[num]) ** 2 * weight_all
        ).sum() / (256 * h1 * w1 * num_all)
        # loss_fea_sup_all = loss_fea_sup_all + (5 - num) * 0.04 * cond_sup_loss
        loss_fea_sup_all = (
            loss_fea_sup_all + (5 - num) * 0.04 * cond_sup_loss + (5 - num) * 0.04 * warp_sup_loss
        )
        loss_warp = (
            loss_warp
            + (num + 1) * loss_l1
            + (num + 1) * 0.2 * loss_vgg
            + (num + 1) * 2 * loss_edge
            + (num + 1) * 6 * loss_second_smooth
            + (5 - num) * 0.04 * cond_sup_loss
            + (5 - num) * 0.04 * warp_sup_loss
        )
        if num >= 2:
            b1, c1, h1, w1 = flow_all[num].shape
            weight_all = weight.reshape(-1, 1, 1).repeat(1, h1, w1)
            flow_sup_loss = (
                torch.norm(flow_sup_all[num].detach() - flow_all[num], p=2, dim=1) * weight_all
            ).sum() / (h1 * w1 * num_all)
            loss_flow_sup_all = loss_flow_sup_all + (num + 1) * 1 * flow_sup_loss
            loss_warp = loss_warp + (num + 1) * 1 * flow_sup_loss

    loss_warp = 0.01 * loss_smooth + loss_warp

    skin_mask = warped_prod_edge_un.detach() * (1 - person_clothes_edge.to(device))

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

    loss_mask_l1 = torch.mean(torch.abs(1 - m_composite))
    loss_l1_skin = criterionL1(p_rendered * skin_mask, skin_mask * real_image.to(device))
    loss_vgg_skin = criterionVGG(p_rendered * skin_mask, skin_mask * real_image.to(device))
    loss_l1 = criterionL1(p_tryon, real_image.to(device))
    loss_vgg = criterionVGG(p_tryon, real_image.to(device))
    bg_loss_l1 = criterionL1(p_rendered, real_image.to(device))
    bg_loss_vgg = criterionVGG(p_rendered, real_image.to(device))

    # loss_mask_l1 = criterionL1(person_clothes_edge.to(device), m_composite)
    # loss_l1_skin = criterionL1(p_tryon * skin_mask, skin_mask * real_image.to(device))
    # loss_vgg_skin = criterionVGG(p_tryon * skin_mask, skin_mask * real_image.to(device))
    # loss_l1 = criterionL1(p_tryon, real_image.to(device))
    # loss_vgg = criterionVGG(p_tryon, real_image.to(device))

    if loss_lrdecay:
        loss_gen = (
            loss_l1 * 5
            + loss_l1_skin * 60
            + loss_vgg
            + loss_vgg_skin * 4
            + bg_loss_l1 * 5
            + bg_loss_vgg
            + 1 * loss_mask_l1
        )
    else:
        loss_gen = (
            loss_l1 * 5
            + loss_l1_skin * 30
            + loss_vgg
            + loss_vgg_skin * 2
            + bg_loss_l1 * 5
            + bg_loss_vgg
            + 1 * loss_mask_l1
        )

    loss_all = 0.25 * loss_warp + loss_gen

    warp_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    loss_all.backward()
    warp_optimizer.step()
    gen_optimizer.step()

    train_batch_time = time.time() - batch_start_time

    # Visualize
    if global_step % sample_step == 0:
        a = real_image.float().to(device)
        b = p_tryon_un.detach()
        c = clothes.to(device)
        d = person_clothes.to(device)
        e = torch.cat([skin_mask.to(device), skin_mask.to(device), skin_mask.to(device)], 1)
        f = warped_cloth
        g = p_rendered
        h = torch.cat([m_composite, m_composite, m_composite], 1)
        i = p_tryon
        combine = torch.cat([a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0]], 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        writer.add_image('combine', (combine.data + 1) / 2.0, global_step)
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(samples_dir / f'{global_step}.jpg'), bgr)

    return loss_all.item(), loss_warp.item(), loss_gen.item(), train_batch_time


def train_pf_e2e(opt):
    epoch_num = opt.niter + opt.niter_decay
    writer = SummaryWriter(opt.save_dir)
    validate = True if opt.valroot else False

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Directories
    log_path = Path(opt.save_dir) / 'log.txt'
    weights_dir = Path(opt.save_dir) / 'weights'  # weights dir
    samples_dir = Path(opt.save_dir) / 'samples'  # samples dir
    weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
    samples_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Device
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Model
    pb_warp_model = PBAFWM(45, opt.align_corners).to(device)
    pb_warp_model.eval()
    pb_warp_ckpt = get_ckpt(opt.pb_warp_checkpoint)
    load_ckpt(pb_warp_model, pb_warp_ckpt)
    print_log(log_path, f'Load pretrained parser-based warp from {opt.pb_warp_checkpoint}')
    pb_gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d).to(device)
    pb_gen_model.eval()
    pb_gen_ckpt = get_ckpt(opt.pb_gen_checkpoint)
    load_ckpt(pb_gen_model, pb_gen_ckpt)
    print_log(log_path, f'Load pretrained parser-based gen from {opt.pb_gen_checkpoint}')
    pf_warp_model = AFWM(3, opt.align_corners).to(device)
    pf_warp_ckpt = get_ckpt(opt.pf_warp_checkpoint)
    load_ckpt(pf_warp_model, pf_warp_ckpt)
    print_log(log_path, f'Load pretrained parser-free warp from {opt.pf_warp_checkpoint}')
    pf_gen_model = MobileNetV2_unet(7, 4).to(device)
    pf_gen_ckpt = get_ckpt(opt.pf_gen_checkpoint)
    load_ckpt(pf_gen_model, pf_gen_ckpt)
    print_log(log_path, f'Load pretrained parser-free gen from {opt.pf_gen_checkpoint}')

    # Optimizer
    warp_optimizer = smart_optimizer(
        model=pf_warp_model, name=opt.optimizer, lr=0.2 * opt.lr, momentum=opt.momentum
    )
    gen_optimizer = smart_optimizer(
        model=pf_gen_model, name=opt.optimizer, lr=opt.lr, momentum=opt.momentum
    )

    # Resume
    best_fid, start_epoch = float('inf'), 1
    if opt.resume:
        if pf_warp_ckpt:
            _ = smart_resume(
                pf_warp_ckpt, warp_optimizer, opt.pf_warp_checkpoint, epoch_num=epoch_num
            )
        if pf_gen_ckpt:  # resume with information of gen_model
            start_epoch, best_fid = smart_resume(
                pf_gen_ckpt, gen_optimizer, opt.pf_gen_checkpoint, epoch_num=epoch_num
            )

    # Scheduler
    last_epoch = start_epoch - 1
    warp_scheduler = MyLRScheduler(warp_optimizer, last_epoch, opt.niter, opt.niter_decay, False)
    gen_scheduler = MyLRScheduler(gen_optimizer, last_epoch, opt.niter, opt.niter_decay, False)

    # Dataloader
    train_data = LoadVITONDataset(path=opt.dataroot, phase='train', size=(256, 192))
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers
    )
    if validate:
        val_data = LoadVITONDataset(path=opt.valroot, phase='test', size=(256, 192))
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=opt.workers)

    # Loss
    criterionL1 = nn.L1Loss()
    criterionL2 = nn.MSELoss('sum')
    criterionVGG = VGGLoss(device=device)

    # Start training
    nb = len(train_loader)  # number of batches
    total_steps = epoch_num * nb
    eta_meter = AverageMeter()
    global_step = 1
    t0 = time.time()
    train_warp_loss = 0
    train_gen_loss = 0
    train_loss = 0
    steps_warp_loss = 0
    steps_gen_loss = 0
    steps_loss = 0

    for epoch in range(start_epoch, epoch_num + 1):
        pf_warp_model.train()
        pf_gen_model.train()
        epoch_start_time = time.time()

        loss_lrdecay = epoch > opt.niter
        for idx, data in enumerate(train_loader):  # batch -----------------------------------------
            loss_all, loss_warp, loss_gen, train_batch_time = train_batch(
                data,
                models={
                    'pb_warp': pb_warp_model,
                    'pb_gen': pb_gen_model,
                    'pf_warp': pf_warp_model,
                    'pf_gen': pf_gen_model,
                },
                optimizers={'warp': warp_optimizer, 'gen': gen_optimizer},
                criterions={'L1': criterionL1, 'L2': criterionL2, 'VGG': criterionVGG},
                device=device,
                writer=writer,
                global_step=global_step,
                samples_dir=samples_dir,
                sample_step=opt.sample_step,
                loss_lrdecay=loss_lrdecay,
            )

            train_warp_loss += loss_warp
            train_gen_loss += loss_gen
            train_loss += loss_all
            steps_warp_loss += loss_warp
            steps_gen_loss += loss_gen
            steps_loss += loss_all

            # Logging
            eta_meter.update(train_batch_time)
            now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
            if global_step % opt.print_step == 0:
                eta_sec = ((epoch_num + 1 - epoch) * len(train_loader) - idx - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = '[{}]: [epoch-{}/{}]--[global_step-{}/{}-{:.2%}]--[loss-{:.6f}: warp-{:.6f}, gen-{:.6f}]--[lr: warp-{}, gen-{}]--[eta-{}]'.format(  # noqa: E501
                    now,
                    epoch,
                    epoch_num,
                    global_step,
                    total_steps,
                    global_step / total_steps,
                    steps_loss / opt.print_step,
                    steps_warp_loss / opt.print_step,
                    steps_gen_loss / opt.print_step,
                    ['%.6f' % group['lr'] for group in warp_optimizer.param_groups],
                    ['%.6f' % group['lr'] for group in gen_optimizer.param_groups],
                    eta_sec_format,
                )  # noqa: E501
                print_log(log_path, strs)

                steps_warp_loss = 0
                steps_gen_loss = 0
                steps_loss = 0

            global_step += 1
            # end batch ---------------------------------------------------------------------------

        # Scheduler
        warp_scheduler.step()
        gen_scheduler.step()

        # Visualize train loss
        train_warp_loss /= len(train_loader)
        train_gen_loss /= len(train_loader)
        train_loss /= len(train_loader)
        writer.add_scalar('train_warp_loss', train_warp_loss, epoch)
        writer.add_scalar('train_gen_loss', train_gen_loss, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)

        # Validate
        if validate:
            pf_warp_model.eval()
            pf_gen_model.eval()
            metrics = val.run_val_pf(
                data_loader=val_loader,
                models={'warp': pf_warp_model, 'gen': pf_gen_model},
                align_corners=opt.align_corners,
                device=device,
                log_path=log_path,
                save_dir=opt.save_dir,
                img_dir=Path(opt.valroot) / 'test_img',
                save_img=False,
            )
            fid = metrics['fid']
            if fid < best_fid:
                best_fid = fid

        # Save model
        warp_ckpt = {
            'epoch': epoch,
            'best_fid': best_fid,
            'model': pf_warp_model.state_dict(),
            'optimizer': warp_optimizer.state_dict(),
        }
        gen_ckpt = {
            'epoch': epoch,
            'best_fid': best_fid,
            'model': pf_gen_model.state_dict(),
            'optimizer': gen_optimizer.state_dict(),
        }
        torch.save(warp_ckpt, weights_dir / 'pf_warp_last.pt')
        torch.save(gen_ckpt, weights_dir / 'pf_gen_last.pt')
        if validate and best_fid == fid:
            torch.save(warp_ckpt, weights_dir / 'pf_warp_best.pt')
            torch.save(gen_ckpt, weights_dir / 'pf_gen_best.pt')
            print_log(
                log_path,
                'Save best with fid %.3f at epoch %d, iters %d' % (fid, epoch, global_step - 1),
            )
        if epoch % opt.save_period == 0:
            torch.save(warp_ckpt, weights_dir / 'pf_warp_epoch_{epoch}.pt')
            torch.save(gen_ckpt, weights_dir / 'pf_gen_epoch_{epoch}.pt')
            print_log(
                log_path,
                'Save the model at the end of epoch %d, iters %d' % (epoch, global_step - 1),
            )
        del warp_ckpt, gen_ckpt

        print_log(
            log_path,
            'End of epoch %d / %d: train_loss: %.3f \t time: %d sec'
            % (epoch, opt.niter + opt.niter_decay, train_loss, time.time() - epoch_start_time),
        )

        train_warp_loss = 0
        train_gen_loss = 0
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
    train_pf_e2e(opt)
