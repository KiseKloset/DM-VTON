import datetime
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader.dataloaders import create_dataloader
from models.warp_modules.flow_style.afwm import AFWM
from models.losses.vgg_loss import VGGLoss
from models.losses.tv_loss import TVLoss
from opt.train_opt import TrainOptions
from utils.torch_utils import AverageMeter, select_device, smart_optimizer

opt = TrainOptions().parse()

writer = SummaryWriter(opt.save_dir)

# torch.distributed.init_process_group(
#     'nccl',
#     init_method='env://'
# )
device = select_device(opt.device, batch_size=opt.batch_size)

# Directories
weights_dir = opt.save_dir / 'weights'  # weights dir
samples_dir = opt.save_dir / 'samples' # samples dir
weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
samples_dir.mkdir(parents=True, exist_ok=True)  # make dir
last, best = weights_dir / 'last.pt', weights_dir / 'best.pt'

# Model
warp_model = AFWM(opt, 45).to(device)
# warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(warp_model).to(device)
# if opt.isTrain and len(opt.gpu_ids):
#     model = torch.nn.parallel.DistributedDataParallel(warp_model, device_ids=[opt.gpu_ids[0]])

# Optimizer
warp_optimizer = smart_optimizer(model=warp_model, name=opt.optimizer, lr=opt.lr, momentum=opt.momentum)

# Trainloader
# train_data = CreateDataset(opt) 
# train_sampler = DistributedSampler(train_data)
# train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
#                                                num_workers=16, pin_memory=True, sampler=train_sampler)

# Resume
start_epoch = 0

train_loader = create_dataloader(
    path=opt.data,
    batch_size=opt.batch_size,
    rank=-1,
    workers=opt.workers,
    shuffle=False,
    resize_or_crop=opt.resize_or_crop,
    n_downsample_global=opt.n_downsample_global,
    prefix='train',
)

# Start training
t0 = time.time()
nb = len(train_loader)  # number of batches
dataset_size = len(train_loader)
criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
start_epoch = 0
epochs = opt.niter + opt.niter_decay
total_steps = epochs * dataset_size
global_step = 0
max_iter = len(train_loader) - 1
eta_meter = AverageMeter()

for epoch in range(start_epoch, epochs):
    warp_model.train()
    epoch_start_time = time.time()

    train_loss = 0

    for idx, data in enumerate(train_loader): # batch -------------------------------------------------------------
        batch_start_time = time.time()
        if idx >= max_iter:
            break

        t_mask = torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.float64))
        data['label'] = data['label']*(1-t_mask)+t_mask*4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy()==4).astype(np.int64))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1,data['densepose'].data.long().to(device),1.0)
        densepose_fore = data['densepose']/24.0
        face_mask = torch.FloatTensor((data['label'].cpu().numpy()==1).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy()==12).astype(np.int64))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy()==5).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy()==6).astype(np.int64)) + \
                             torch.FloatTensor((data['label'].cpu().numpy()==8).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy()==9).astype(np.int64)) + \
                             torch.FloatTensor((data['label'].cpu().numpy()==10).astype(np.int64))
        preserve_mask = torch.cat([face_mask,other_clothes_mask],1)
        concat = torch.cat([preserve_mask.to(device),densepose,pose.to(device)],1)

        #import ipdb; ipdb.set_trace()

        flow_out = warp_model(concat.to(device), clothes.to(device), pre_clothes_edge.to(device))
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5**(4-num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5**(4-num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.to(device))
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.to(device))
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.to(device))
            b,c,h,w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x)/(b*c*h*w)
            loss_flow_y = (delta_y_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y)/(b*c*h*w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth

        loss_all = 0.01 * loss_smooth + loss_all
        train_loss += loss_all

        warp_optimizer.zero_grad()
        loss_all.backward()
        warp_optimizer.step()
        # end batch ------------------------------------------------------------------------------------------------

        ############## Display results and errors ##########
        train_batch_time = time.time() - batch_start_time
        eta_meter.update(train_batch_time)
        global_step += 1
        now = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
        if (global_step > 0 and global_step % opt.print_batch_step == 0) or (idx >= len(train_loader) - 1):
            eta_sec = ((epochs + 1 - epoch) * len(train_loader) - idx - 1) * eta_meter.avg
            eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
            strs = '[{}]:[epoch-{}/{}]--[global_step-{}/{}-{:.2%}]--[loss: warp-{:.6f}]--[lr-{:.6f}]--[eta-{}]'.format(
                now, epoch, epochs, global_step, total_steps, global_step/total_steps, 
                loss_all, warp_model.old_lr, eta_sec_format
            )
            print(strs)

        if global_step > 0 and global_step % opt.sample_step == 0:
            a = real_image.float().to(device)
            b = person_clothes.to(device)
            c = clothes.to(device)
            d = torch.cat([densepose_fore.to(device),densepose_fore.to(device),densepose_fore.to(device)],1)
            e = warped_cloth
            f = torch.cat([warped_prod_edge,warped_prod_edge,warped_prod_edge],1)
            combine = torch.cat([a[0],b[0],c[0],d[0],e[0],f[0]], 2).squeeze()
            cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
            writer.add_image('combine', (combine.data + 1) / 2.0, global_step)
            rgb=(cv_img*255).astype(np.uint8)
            bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(samples_dir, f'{global_step}.jpg'), bgr)
    
    # Visualize train loss
    train_loss /= len(train_loader)
    writer.add_scalar('train_loss', train_loss, epoch)

    # end of epoch 
    print('End of epoch %d / %d: train_loss: %.3f \t time: %d sec' %
        (epoch, opt.niter + opt.niter_decay, train_loss, time.time() - epoch_start_time))

    ### save model for this epoch
    ckpt = {
        'epoch': epoch,
        # 'best_fitness': best_fitness,
        'model': deepcopy(warp_model).half(),
        # 'ema': deepcopy(ema.ema).half(),
        # 'updates': ema.updates,
        'optimizer': warp_optimizer.state_dict(),
        'opt': vars(opt),
        'date': datetime.now().isoformat()
    }
    if epoch % opt.save_epoch_freq == 0:
        print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        torch.save(ckpt, w / f'pb_warp_epoch{epoch}.pt')

    if epoch > opt.niter:
        warp_model.update_learning_rate(warp_optimizer)
