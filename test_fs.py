import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from tqdm import tqdm

from data.data_loader_test import CreateDataLoader
from models.afwm_test import AFWM
from models.networks import ResUnetGenerator
from models.rmgn_generator import RMGNGenerator
from options.test_options import TestOptions
from utils.utils import load_checkpoint


opt = TestOptions().parse()

device = torch.device(f'cuda:{opt.gpu_ids[0]}')

start_epoch, epoch_iter = 1, 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)
#import ipdb; ipdb.set_trace()
warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.to(device)
load_checkpoint(warp_model, opt.warp_checkpoint, device)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
# gen_model = RMGNGenerator(multilevel=False, predmask=True)
#print(gen_model)
gen_model.eval()
gen_model.to(device)
load_checkpoint(gen_model, opt.gen_checkpoint, device)

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize


for epoch in range(1,2):

    for i, data in enumerate(tqdm(dataset), start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        real_image = data['image']
        clothes = data['clothes']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = clothes * edge        

        #import ipdb; ipdb.set_trace()

        flow_out = warp_model(real_image.to(device), clothes.to(device))
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.to(device), warped_cloth, warped_edge], 1)
        # gen_inputs_clothes = torch.cat([warped_cloth, warped_edge], 1)
        # gen_inputs_persons = real_image.to(device)
        
        gen_outputs = gen_model(gen_inputs)

        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        tryon_path = os.path.join('results/', opt.name, 'tryon')
        warp_path = os.path.join('results/', opt.name, 'warp')
        os.makedirs(tryon_path, exist_ok=True)
        os.makedirs(warp_path, exist_ok=True)

        #sub_path = path + '/PFAFN'
        #os.makedirs(sub_path,exist_ok=True)

        if step % 1 == 0:
            
            ## save try-on image only

            utils.save_image(
                p_tryon,
                os.path.join(tryon_path, data['p_name'][0]),
                nrow=int(1),
                normalize=True,
                value_range=(-1,1),
            )

            utils.save_image(
                warped_cloth,
                os.path.join(warp_path, data['c_name'][0]),
                nrow=int(1),
                normalize=True,
                value_range=(-1,1),
            )
            
            ## save person image, garment, flow, warped garment, and try-on image
            
            #a = real_image.float().to(device)
            #b = clothes.to(device)
            #flow_offset = de_offset(last_flow)
            #flow_color = f2c(flow_offset).to(device)
            #c= warped_cloth.to(device)
            #d = p_tryon
            #combine = torch.cat([a[0],b[0], flow_color, c[0], d[0]], 2).squeeze()
            #utils.save_image(
            #    combine,
            #    os.path.join('./im_gar_flow_wg', data['p_name'][0]),
            #    nrow=int(1),
            #    normalize=True,
            #    range=(-1,1),
            #)
            

        step += 1
        if epoch_iter >= dataset_size:
            break
