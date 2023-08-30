import json
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

# https://github.com/levindabhi/Self-Correction-Human-Parsing-for-ACGPN
mapping = {
    0: 0,
    1: 1,
    2: 1,
    3: 12,
    4: 4,
    5: 8,
    6: 8,
    7: 4,
    8: 8,
    9: 5,
    10: 6,
    11: 12,
    12: 9,
    13: 10,
    14: 11,
    15: 13,
    16: 2,  # bag
    17: 11,
}

art2lip = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
    4: 5,
    5: 12,
    6: 9,
    7: 6,
    8: 9,
    9: 18,
    10: 19,
    11: 13,
    12: 16,
    13: 17,
    14: 14,
    15: 15,
    16: -1,
    17: 11,
}

atr_label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

lip_label_map = {
    'background': 0,
    'hat': 1,
    'hair': 2,
    'glove': 3,
    'sunglasses': 4,
    'upper_clothes': 5,
    'dress': 6,
    'coat': 7,
    'socks': 8,
    'pants': 9,
    'jumpsuits': 10,
    'scarf': 11,
    'skirt': 12,
    'face': 13,
    'left_arm': 14,
    'right_arm': 15,
    'left_leg': 16,
    'right_leg': 17,
    'left_shoe': 18,
    'right_shoe': 19,
}


class DressCodeDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: str,
        order: str = 'paired',
        category: List[str] = ['dresses', 'upper_body', 'lower_body'],
        size: Tuple[int, int] = (256, 192),
    ):
        """
        Initialize the PyTroch Dataset Class
        :param args: argparse parameters
        :type args: argparse
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (upper_body | lower_body | dresses)
        :type category: list(str)
        :param size: image size (height, width)
        :type size: tuple(int)
        """
        super(data.Dataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.height, self.width = size[0], size[1]
        self.radius = 5
        self.transform_image = get_transform(train=(self.phase == 'train'))
        self.transform_parse = get_transform(
            train=(self.phase == 'train'), method=Image.NEAREST, normalize=False
        )

        im_names = []
        c_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename) as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        im_name = self.im_names[index]
        c_name = self.c_names[index]
        dataroot = self.dataroot_names[index]

        # Person image
        im = Image.open(os.path.join(dataroot, 'images', im_name)).convert('RGB')
        im = im.resize((self.width, self.height))
        im = self.transform_image(im)  # [-1,1]

        # Clothing image
        cloth = Image.open(os.path.join(dataroot, 'images', c_name)).convert('RGB')
        cloth = cloth.resize((self.width, self.height))
        cloth = self.transform_image(cloth)  # [-1,1]

        # Clothing edge
        cloth_edge = Image.open(os.path.join(dataroot, 'edge', c_name)).convert('L')
        cloth_edge = cloth_edge.resize((self.width, self.height))
        cloth_edge = self.transform_parse(cloth_edge)  # [-1,1]

        # Unpaired clothing image
        un_index = np.random.randint(len(self.c_names))
        un_c_name = self.c_names[un_index]
        un_cloth = Image.open(os.path.join(dataroot, 'images', un_c_name)).convert('RGB')
        un_cloth = un_cloth.resize((self.width, self.height))
        un_cloth = self.transform_image(un_cloth)  # [-1,1]

        # Unpaired Clothing edge
        un_cloth_edge = Image.open(os.path.join(dataroot, 'edge', un_c_name)).convert('L')
        un_cloth_edge = un_cloth_edge.resize((self.width, self.height))
        un_cloth_edge = self.transform_parse(un_cloth_edge)  # [-1,1]

        # Skeleton
        # skeleton = Image.open(os.path.join(dataroot, 'skeletons', im_name.replace("_0", "_5")))
        # skeleton = skeleton.resize((self.width, self.height))
        # skeleton = self.transform(skeleton)

        # Label Map
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(os.path.join(dataroot, 'label_maps', parse_name))
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        im_parse = self.transform_parse(im_parse) * 255.0
        parse = torch.zeros(im_parse.shape)
        # Mapping
        for k, v in mapping.items():
            old_mask = (im_parse == k).float()
            parse = parse * (1 - old_mask) + old_mask * v  # map value k to value v in new mask

        # Load pose points
        pose_name = im_name.replace('_0.jpg', '_2.json')
        with open(os.path.join(dataroot, 'keypoints', pose_name)) as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
            point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
            if point_x > 1 and point_y > 1:
                draw.rectangle(
                    (point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white'
                )
            one_map = self.transform_image(one_map.convert('RGB'))
            pose_map[i] = one_map[0]

        # Dense
        dense_mask = Image.open(
            os.path.join(dataroot, 'dense', im_name.replace('_0.jpg', '_5.png'))
        ).convert('L')
        dense_mask = dense_mask.resize((self.width, self.height), Image.NEAREST)
        dense_mask = self.transform_parse(dense_mask) * 255.0

        if self.phase == 'train':
            return {
                'img_path': os.path.join(dataroot, 'images', im_name),
                'color_path': os.path.join(dataroot, 'images', c_name),
                'color_un_path': os.path.join(dataroot, 'images', un_c_name),
                'path': os.path.join(dataroot, 'label_maps', parse_name),
                'image': im,
                'color': cloth,
                'edge': cloth_edge,
                'color_un': un_cloth,
                'edge_un': un_cloth_edge,
                'label': parse,
                'pose': pose_map,
                'densepose': dense_mask,
            }
        else:
            return {
                'image': im,
                'color': cloth,
                'edge': cloth_edge,
                'p_name': im_name,
                'c_name': c_name,
            }

    def __len__(self):
        return len(self.c_names)


class DressCodeTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: str,
        order: str = 'paired',
        category: List[str] = ['dresses', 'upper_body', 'lower_body'],
        size: Tuple[int, int] = (256, 192),
    ):
        """
        Initialize the PyTroch Dataset Class
        :param args: argparse parameters
        :type args: argparse
        :param dataroot_path: dataset root folder
        :type dataroot_path:  string
        :param phase: phase (train | test)
        :type phase: string
        :param order: setting (paired | unpaired)
        :type order: string
        :param category: clothing category (upper_body | lower_body | dresses)
        :type category: list(str)
        :param size: image size (height, width)
        :type size: tuple(int)
        """
        super(data.Dataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.height, self.width = size[0], size[1]
        self.transform_image = get_transform(train=(self.phase == 'train'))
        self.transform_parse = get_transform(
            train=(self.phase == 'train'), method=Image.NEAREST, normalize=False
        )

        im_names = []
        c_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename) as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        im_name = self.im_names[index]
        c_name = self.c_names[index]
        dataroot = self.dataroot_names[index]

        # Person image
        im = Image.open(os.path.join(dataroot, 'images', im_name)).convert('RGB')
        im = im.resize((self.width, self.height))
        im = self.transform_image(im)  # [-1,1]

        # Clothing image
        cloth = Image.open(os.path.join(dataroot, 'images', c_name)).convert('RGB')
        cloth = cloth.resize((self.width, self.height))
        cloth = self.transform_image(cloth)  # [-1,1]

        # Clothing edge
        cloth_edge = Image.open(os.path.join(dataroot, 'edge', c_name)).convert('L')
        cloth_edge = cloth_edge.resize((self.width, self.height))
        cloth_edge = self.transform_parse(cloth_edge)  # [-1,1]

        result = {
            'image': im,
            'clothes': cloth,
            'edge': cloth_edge,
            'p_name': im_name,
            'c_name': c_name,
        }

        return result

    def __len__(self):
        return len(self.c_names)


def get_transform(train, method=Image.BICUBIC, normalize=True):
    transform_list = []

    # if opt.resize_or_crop == 'none':
    base = float(2**4)
    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if train:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, 0)))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
