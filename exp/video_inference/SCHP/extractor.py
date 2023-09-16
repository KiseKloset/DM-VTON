import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

import torchvision.transforms as transforms

from SCHP.networks import init_model
from SCHP.utils.transforms import transform_logits, get_affine_transform


ROOT = Path(__file__).parent


trans_dict = {
    0:0,
    1:1, 2:1,
    5:4, 6:4, 7:4, 
    18:5,
    19:6,
    9:8, 12:8,
    16:9,
    17:10,
    14:11,
    4:12, 13:12,
    15:13
}


class Extractor:
    def __init__(self, device):
        self.device = device
        self.num_classes = 20
        self.input_size = [473, 473]
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.label = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                    'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                    'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

        self.model = init_model('resnet101', num_classes=self.num_classes, pretrained=None)
        self.__load_checkpoint(self.model)
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        self.upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)


    def __load_checkpoint(self, model):
        state_dict = torch.load(ROOT / 'ckpt/final.pth')['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


    def __box2cs(self, box):
        x, y, w, h = box[:4]
        return self.__xywh2cs(x, y, w, h)


    def __xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale


    def __call__(self, image):
        h, w, _ = image.shape
        c, s = self.__box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(c, s, r, self.input_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        image = self.transform(image).to(self.device).unsqueeze(0)

        output = self.model(image)
        upsample_output = self.upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)
        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
        parsing_result = np.argmax(logits_result, axis=2)
        output_arr = np.asarray(parsing_result, dtype=np.uint8)
        new_arr = np.full(output_arr.shape, 7)
        for old, new in trans_dict.items():
            new_arr = np.where(output_arr == old, new, new_arr)
        output_img = np.asarray(new_arr, dtype=np.uint8)
        return output_img