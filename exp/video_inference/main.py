import os
import torch
import time
import cupy
import numpy as np
import cv2

from PIL import Image, ImageDraw
import torchvision.utils as utils
from tqdm import tqdm
import mediapipe as mp

from RMGN.raw import RMGN
from PFAFN.raw import PFAFN
from FlowStyle.raw import FlowStyle
from SRMGN.raw import SRMGN
from ACGPN.raw import ACGPN
from SDAFN.raw import SDAFN

from utils import get_transform, get_params
from OpenPoseCaffe.predict_pose import PosePredictor
from OpenPoseCaffe.visualize import visualize_pose
from SCHP.extractor import Extractor
from yolov7_pose.estimator import Yolov7PoseEstimation

# WUTON_ID = 1
RMGN_ID = 'rmgn'
PFAFN_ID = 'pfafn'
FLOW_STYLE_ID = 'flow_style'
SRMGN_ID = 'srmgn'
ACGPN_ID = 'acgpn'
SDAFN_ID = 'sdafn'

MODELS = {
    # WUTON_ID: WUTON, 
    SRMGN_ID: SRMGN,
    # FLOW_STYLE_ID: FlowStyle,
    # RMGN_ID: RMGN,
    # PFAFN_ID: PFAFN,
    # SDAFN_ID: SDAFN,
    # ACGPN_ID: ACGPN,
}

TARGET_WIDTH = 192
TARGET_HEIGHT = 256

MP_POSE = None


def crop_upper_body(frame, pose_detector):
    results = pose_detector.process(frame)
    h, w, _ = frame.shape
    
    # TUNGPNT2
    if results.pose_landmarks is None:
        return {'origin_frame': frame, 'cropped_frame': None, 't': 0, 'l': 0, 'b': 0, 'r': 0}
    # return {'origin_frame': frame, 'cropped_frame': frame, 't': 0, 'l': 0, 'b': h, 'r': w}
    
    landmarks = results.pose_landmarks.landmark

    # 2 lower points of upper body
    l1 = {'x': landmarks[23].x, 'y': landmarks[23].y}
    l2 = {'x': landmarks[24].x, 'y': landmarks[24].y}

    # 2 eyes
    e1 = {'x': landmarks[2].x, 'y': landmarks[2].y}
    e2 = {'x': landmarks[5].x, 'y': landmarks[5].y}

    points = [l1, l2, e1, e2]

    # bbox
    bbox = {}
    t = min([i['y'] for i in points])
    b = max([i['y'] for i in points])
    l = min([i['x'] for i in points])
    r = max([i['x'] for i in points])

    # de-normalize
    t = int(t * h)
    b = int(b * h)
    l = int(l * w)
    r = int(r * w)

    # padding
    bh = b - t
    bw = r - l
    b += bh // 5
    t -= bh // 5
    r += bw // 1
    l -= bw // 1

    # crop
    t = max(0, min(t, h))
    b = max(0, min(b, h))
    l = max(0, min(l, w))
    r = max(0, min(r, w))

    return {'origin_frame': frame, 'cropped_frame': frame[t:b, l:r, :], 't': t, 'l': l, 'b': b, 'r': r}


def gen_input(model_id, device, input_paths, pose_detector):
    # if model_id == WUTON_ID:
    #     gan_product_image_batch = torch.rand(1, 3, 224, 224)
    #     model_agnostic_image_batch = torch.rand(1, 6, 224, 224)
    #     return gan_product_image_batch, model_agnostic_image_batch

    ## Path
    clothes_path =  input_paths['cloth_path']
    edge_path = input_paths['edge_path']
    clothes_name = input_paths['cloth_name']
    input_path = input_paths['video_path']

    ### Create video object
    cap = cv2.VideoCapture(input_path)
    if cap.isOpened() == False:
        print("The video is fucked up")
        sys.exit(0)

    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ## Parse transform config
    transform = get_transform()
    transform_E = get_transform(method=Image.NEAREST, normalize=False)

    ### Read edge image
    target_edge = Image.open(os.path.join(edge_path, clothes_name)).convert("L")
    target_edge = transform_E(target_edge).unsqueeze(0)

    ### Read clothes image
    target_clothes = Image.open(os.path.join(clothes_path, clothes_name)).convert('RGB')
    target_clothes = transform(target_clothes).unsqueeze(0)

    ### Pose prediction
    pose_predictor = PosePredictor()

    ### Parser
    extractor = Extractor(device)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = frame[:, :, :]

            # Crop the image using pose
            cropped_result = crop_upper_body(frame, pose_detector)
            frame = cropped_result['cropped_frame']
            if frame is None:
                yield (cropped_result, None)
                continue

            # Crop the image to have same ratio
            if frame.shape[0] * TARGET_WIDTH < frame.shape[1] * TARGET_HEIGHT:
                height = frame.shape[0]
                width = int(TARGET_WIDTH * height / TARGET_HEIGHT)
            else:
                width = frame.shape[1]
                height = int(TARGET_HEIGHT * width / TARGET_WIDTH)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            center = (frame.shape[0] // 2, frame.shape[1] // 2)
            x = center[1] - width // 2
            y = center[0] - height // 2
            frame = frame[y : y + height, x : x + width]
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # Mapping bbox value
            cropped_result['t'] += y
            cropped_result['b'] = cropped_result['t'] + height
            cropped_result['l'] += x
            cropped_result['r'] = cropped_result['l'] + width

            real_image = Image.fromarray(frame)
            real_image = transform(real_image).unsqueeze(0)

            if model_id in (RMGN_ID, PFAFN_ID, FLOW_STYLE_ID, SRMGN_ID):
                yield (cropped_result, real_image, target_clothes, target_edge)
            
            elif model_id == ACGPN_ID:
                # Get pose
                k = pose_predictor.generate_pose_keypoints(frame)
                pose_data = np.array(k['people'][0]['pose_keypoints']).reshape((-1, 3))
                point_num = pose_data.shape[0]
                pose_map = torch.zeros(point_num, TARGET_HEIGHT, TARGET_WIDTH)
                r = 5
                im_pose = Image.new('L', (TARGET_WIDTH, TARGET_HEIGHT))
                pose_draw = ImageDraw.Draw(im_pose)
                for i in range(point_num):
                    one_map = Image.new('L', (TARGET_WIDTH, TARGET_HEIGHT))
                    draw = ImageDraw.Draw(one_map)
                    pointx = pose_data[i, 0]
                    pointy = pose_data[i, 1]
                    if pointx > 1 and pointy > 1:
                        draw.rectangle((pointx-r, pointy-r, pointx +
                                        r, pointy+r), 'white', 'white')
                        pose_draw.rectangle(
                            (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                    one_map = transform(one_map.convert('RGB'))
                    pose_map[i] = one_map[0]
                pose_map = pose_map.unsqueeze(0)

                # Get parse
                parse_map = extractor(frame)
                parse_map = torch.from_numpy(parse_map).float().unsqueeze(0).unsqueeze(0)
                yield (cropped_result, real_image, target_clothes, target_edge, pose_map, parse_map)

            elif model_id == SDAFN_ID:
                k = pose_predictor.generate_pose_keypoints(frame)
                pose = visualize_pose(k)
                pose_img = Image.fromarray(pose)
                pose_img = transform(pose_img).unsqueeze(0)
                ref_input = torch.cat((pose_img, real_image), dim = 1)
                yield (cropped_result, real_image, ref_input, target_clothes, real_image)
            
        else:
            break


def gen_checkpoint_paths(model_id):
    checkpoint_paths = {}
    if model_id==PFAFN_ID:
        checkpoint_paths['warp'] = 'PFAFN/ckp/warp_model_final.pth'
        checkpoint_paths['gen'] = 'PFAFN/ckp/gen_model_final.pth'
    elif model_id==FLOW_STYLE_ID: # aug
        checkpoint_paths['warp'] = 'FlowStyle/ckp/PFAFN_warp_epoch_101.pth'
        checkpoint_paths['gen'] = 'FlowStyle/ckp/PFAFN_gen_epoch_101.pth'
    elif model_id==RMGN_ID:
        checkpoint_paths['warp'] = 'RMGN/ckp/RMGN_warp_epoch_030.pth'
        checkpoint_paths['gen'] = 'RMGN/ckp/RMGN_gen_epoch_030.pth'
    elif model_id==SRMGN_ID:
        # checkpoint_paths['warp'] = 'SRMGN/ckp/PFAFN_warp_epoch_101.pth'
        # checkpoint_paths['gen'] = 'SRMGN/ckp/PFAFN_gen_epoch_101.pth'
        checkpoint_paths['warp'] = '/root/nnknguyen/baseline/SRMGN-VITON/runs/SRMGN_align_merge-viton-v1/PF_e2e_100/weights/pf_warp_epoch_50.pt'
        checkpoint_paths['gen'] = '/root/nnknguyen/baseline/SRMGN-VITON/runs/SRMGN_align_merge-viton-v1/PF_e2e_100/weights/pf_warp_epoch_50.pt'
    elif model_id==ACGPN_ID:
        checkpoint_paths['G'] = 'ACGPN/ckp/latest_net_G.pth'
        checkpoint_paths['G1'] = 'ACGPN/ckp/latest_net_G1.pth'   
        checkpoint_paths['G2'] = 'ACGPN/ckp/latest_net_G2.pth'
        checkpoint_paths['U'] = 'ACGPN/ckp/latest_net_U.pth'        
    elif model_id==SDAFN_ID:
        checkpoint_paths['model'] = 'SDAFN/ckpt/ckpt_viton.pt'
    return checkpoint_paths

def run_once(model_id, pose_model, device, input_paths, output_path):
    checkpoint_paths = gen_checkpoint_paths(model_id)
    model = MODELS[model_id](checkpoint=checkpoint_paths).eval().to(device)

    with pose_model as pose_detector:
        _inputs = gen_input(model_id, device, input_paths, pose_detector)

        for idx, _in in enumerate(_inputs):
            cropped_result  = _in[0]
            original_image = cropped_result['origin_frame']

            if cropped_result['cropped_frame'] is not None:
                _in = [i.to(DEVICE) for i in _in[1:]]

                with cupy.cuda.Device(DEVICE_ID):
                    p_tryon = model(*_in)
                
                cropped_output_path = os.path.join(output_path, f"cropped-{idx}.jpg")
                utils.save_image(
                    p_tryon,
                    cropped_output_path,
                    nrow=int(1),
                    normalize=True,
                    value_range=(-1,1),
                )
                
                # Re-mapping to original image
                t, l, b, r = cropped_result['t'], cropped_result['l'], cropped_result['b'], cropped_result['r']
                cropped_output = cv2.imread(cropped_output_path)
                cropped_output = cv2.resize(cropped_output, (r - l, b - t))            
                original_image[t:b, l:r, :] = cropped_output
                os.remove(cropped_output_path)
            cv2.imwrite(os.path.join(output_path, f"{idx}.jpg"), original_image)


if __name__ == "__main__":
    DEVICE_ID = 2
    DEVICE = f"cuda:{DEVICE_ID}"
    input_paths = {
        'video_path': '../dataset/video/0.mp4',
        'cloth_path': '../dataset/Flow-Style-VTON/VITON_test/test_color',
        'edge_path': '../dataset/Flow-Style-VTON/VITON_test/test_edge',
        'cloth_name': '000074_1.jpg',
    }
    output_path = 'result/demo'

    # pose_model = mp.solutions.pose.Pose(
    #     model_complexity=0,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5)
    pose_model = Yolov7PoseEstimation(
        weight_path="/root/nnknguyen/baseline/video_inference/yolov7_pose/ckpt/yolov7-w6-pose.pt",
        device=DEVICE
    )

    for model_id in tqdm(MODELS):
        with torch.no_grad():
            os.makedirs(os.path.join(output_path, str(model_id)), exist_ok=True)
            run_once(model_id, pose_model, DEVICE, input_paths, os.path.join(output_path, str(model_id)))