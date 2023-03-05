import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import utils
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from data.base_dataset import get_params, get_transform
from data.data_loader_test import CreateDataLoader
from models.afwm_test import AFWM
from models.rmgn_generator import RMGNGenerator
from options.test_video_options import TestVideoOptions
from utils.utils import load_checkpoint, Profile

TARGET_WIDTH = 192
TARGET_HEIGHT = 256


if __name__ == "__main__":
    opt = TestVideoOptions().parse()

    device = torch.device(f'cuda:{opt.gpu_ids[0]}')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    warp_model = AFWM(opt, 3)
    warp_model.eval()
    warp_model.to(device)
    load_checkpoint(warp_model, opt.warp_checkpoint, device)

    gen_model = RMGNGenerator(multilevel=False, predmask=True)
    gen_model.eval()
    gen_model.to(device)
    load_checkpoint(gen_model, opt.gen_checkpoint, device)

    ### Prepare path
    root_path = Path(__file__).parent.parent
    clothes_path = root_path / "dataset" / "SPLIT-VITON" / "VITON_test" / "test_clothes"
    edge_path = root_path / "dataset" / "SPLIT-VITON" / "VITON_test" / "test_edge"
    clothes_name = opt.target_clothes
    input_path = root_path / "SRMGN-VITON" / "video" / "input"
    input_name = opt.input_video
    output_path = root_path / "SRMGN-VITON" / "video" / "output" / Path(input_name).stem
    output_path.mkdir(parents=True, exist_ok=True)

    ### Create video object
    cap = cv2.VideoCapture(str(input_path / input_name))
    if cap.isOpened() == False:
        print("The video is fucked up")
        sys.exit(0)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    width = TARGET_WIDTH * height / TARGET_HEIGHT

    height = int(height)
    width = int(width)

    ## Parse transform config
    params = get_params(opt, (TARGET_WIDTH, TARGET_HEIGHT))
    transform = get_transform(opt, params)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)

    ### Read edge image
    target_edge = Image.open(edge_path / clothes_name).convert("L")
    target_edge = transform_E(target_edge).unsqueeze(0)
    target_edge = torch.FloatTensor((target_edge.detach().numpy() > 0.5).astype(np.int64))

    ### Read clothes image
    target_clothes = Image.open(clothes_path / clothes_name).convert('RGB')
    target_clothes = transform(target_clothes).unsqueeze(0)
    target_clothes = target_clothes * target_edge

    ### Loop through input video
    with torch.no_grad():
        seen, dt = 0, (Profile(), Profile())

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                ### Resize to target size
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                center = (frame.shape[0] // 2, frame.shape[1] // 2)
                x = center[1] - width // 2
                y = center[0] - height // 2
                frame = frame[y : y + height, x : x + width]
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

                ### The game starts here
                real_image = Image.fromarray(frame)
                real_image = transform(real_image).unsqueeze(0)

                with dt[0]:
                    flow_out = warp_model(real_image.to(device), target_clothes.to(device))
                    warped_clothes, last_flow, = flow_out
                    warped_edge = F.grid_sample(target_edge.to(device), last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros', align_corners=opt.align_corners)
                
                with dt[1]:
                    #gen_inputs = torch.cat([real_image.to(device), warped_clothes, warped_edge], 1)
                    gen_inputs_clothes = torch.cat([warped_clothes, warped_edge], 1)
                    gen_inputs_persons = real_image.to(device)
                    
                    gen_outputs, out_L1, out_L2, M_list = gen_model(gen_inputs_persons, gen_inputs_clothes)

                    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
                    p_rendered = torch.tanh(p_rendered)
                    m_composite = torch.sigmoid(m_composite)
                    m_composite = m_composite * warped_edge
                    p_tryon = warped_clothes * m_composite + p_rendered * (1 - m_composite)
                
                seen += len(p_tryon)

                utils.save_image(
                    p_tryon,
                    output_path / f"{seen}.jpg",
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1,1),
                )

            else: 
                break

    cap.release()

    ############## FPS ##############
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    t = (sum(t), ) + t
    print(f'Speed: %.1fms all, %.1fms warp, %.1fms gen per image at shape {real_image.size()}' % t)


    ##### Magic #####
    # result = cv2.VideoWriter(str(output_path / input_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (TARGET_WIDTH, TARGET_HEIGHT))

    # for i in range(1, seen + 1):
    #     out_frame = cv2.imread(str(output_path / f"{i}.jpg"))
    #     result.write(out_frame)

    # result.release()
    # os.system(f'rm {output_path}/*.jpg')