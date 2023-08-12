import cv2
import sys
import torch
from pathlib import Path
from torchvision import transforms
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATHß


from mini_utils import letterbox, non_max_suppression_kpt, output_to_keypoints, keypoints_to_mediapipe


class Yolov7PoseEstimation:
    def __init__(self, weight_path="yolov7-w6-pose.pt", device="cpu"):
        self.device = torch.device(device)
        
        weights = torch.load(weight_path, map_location='cpu')
        self.model = weights['model']
        _ = self.model.float().eval()

        self.is_half = torch.cuda.is_available() and self.device.type != 'cpu'
        if self.is_half:
            self.model.half().to(device)
    

    def __enter__(self):
        return self
    

    # image: opencv image (aka numpy array)
    def process(self, image):
        h, w = image.shape[:2]
        _image, new_shape, padding_top, padding_bottom, padding_left, padding_right = letterbox(image, 960, stride=64, auto=True)
        new_h, new_w = new_shape
        image = transforms.ToTensor()(_image).unsqueeze(0)

        if self.is_half:
            image = image.half().to(self.device)

        output, _ = self.model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        keypoints = output_to_keypoints(output)
        result = keypoints_to_mediapipe(keypoints)

        if result.pose_landmarks is not None:
            for landmark in result.pose_landmarks.landmark:
                landmark.x = (landmark.x - padding_left) / new_w
                landmark.y = (landmark.y - padding_top) / new_h

        return result


    def __exit__(self, type, value, traceback):
        pass