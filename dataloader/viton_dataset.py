import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class LoadVITONDataset(Dataset):
    def __init__(
        self,
        path: str,
        phase: str = 'train',
        size: tuple[int, int] = (256, 192),
    ) -> None:
        super().__init__()
        self.dataroot = path
        self.phase = phase
        self.height, self.width = size[0], size[1]
        self.radius = 5
        self.transform_image = get_transform(train=(self.phase == 'train'))
        self.transform_parse = get_transform(
            train=(self.phase == 'train'), method=Image.NEAREST, normalize=False
        )

        self.img_names, self.cloth_names = [], []

        with open(Path(self.dataroot) / f"{phase}_pairs.txt") as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                self.img_names.append(img_name)
                self.cloth_names.append(c_name)

        self.unique_clothes = list(set(self.cloth_names))

    def __getitem__(self, index: int) -> dict:
        im_name, c_name = self.img_names[index], self.cloth_names[index]

        # Person image
        img = Image.open(Path(self.dataroot) / f'{self.phase}_img' / im_name).convert('RGB')
        # img = img.resize((self.width, self.height))
        img_tensor = self.transform_image(img)  # [-1,1]

        # Clothing image
        cloth = Image.open(Path(self.dataroot) / f'{self.phase}_color' / c_name).convert('RGB')
        # cloth = cloth.resize((self.width, self.height))
        cloth_tensor = self.transform_image(cloth)  # [-1,1]

        # Clothing edge
        cloth_edge = Image.open(Path(self.dataroot) / f'{self.phase}_edge' / c_name).convert('L')
        # cloth_edge = cloth_edge.resize((self.width, self.height))
        cloth_edge_tensor = self.transform_parse(cloth_edge)  # [-1,1]

        if self.phase == 'train':
            # Unpaired clothing image
            other_clothes = self.unique_clothes.copy()
            other_clothes.remove(c_name)  # remove the original cloth
            un_c_name = random.choice(other_clothes)
            un_cloth = Image.open(Path(self.dataroot) / f'{self.phase}_color' / un_c_name).convert(
                'RGB'
            )
            # un_cloth = un_cloth.resize((self.width, self.height))
            un_cloth_tensor = self.transform_image(un_cloth)  # [-1,1]

            # Unpaired Clothing edge
            un_cloth_edge = Image.open(
                Path(self.dataroot) / f'{self.phase}_edge' / un_c_name
            ).convert('L')
            # un_cloth_edge = un_cloth_edge.resize((self.width, self.height))
            un_cloth_edge_tensor = self.transform_parse(un_cloth_edge)  # [-1,1]

            # Parse map
            parse_path1 = Path(self.dataroot) / f'{self.phase}_label' / f'{Path(im_name).stem}.jpg'
            parse_path2 = Path(self.dataroot) / f'{self.phase}_label' / f'{Path(im_name).stem}.png'
            parse_path = parse_path1 if parse_path1.is_file() else parse_path2
            parse = Image.open(parse_path).convert('L')
            # parse = parse.resize((self.width, self.height), Image.NEAREST)
            parse_tensor = self.transform_parse(parse) * 255.0

            # Pose: 18 keypoints [x0, y0, z0, x1, y1, z1, ...]
            with open(
                Path(self.dataroot) / f'{self.phase}_pose' / f'{Path(im_name).stem}.json'
            ) as f:
                pose_label = json.load(f)
                try:
                    pose_data = pose_label['people'][0]['pose_keypoints']
                except IndexError:
                    pose_data = [0 for i in range(54)]
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:18]

            point_num = pose_data.shape[0]
            pose_tensor = torch.zeros(point_num, self.height, self.width)
            r = self.radius
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                pointx = pose_data[i, 0]
                pointy = pose_data[i, 1]
                if pointx > 1 and pointy > 1:
                    draw.rectangle(
                        (pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white'
                    )
                    pose_draw.rectangle(
                        (pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white'
                    )
                one_map = self.transform_image(one_map.convert('RGB'))
                pose_tensor[i] = one_map[0]

            # Densepose
            dense_mask = np.load(
                Path(self.dataroot) / f'{self.phase}_densepose' / f'{Path(im_name).stem}.npy'
            ).astype(np.float32)
            dense_tensor = self.transform_parse(dense_mask)

        if self.phase == 'train':
            return {
                'img_name': im_name,
                'color_name': c_name,
                'color_un_name': un_c_name,
                'parse_name': str(parse_path.name),
                'image': img_tensor,
                'color': cloth_tensor,
                'edge': cloth_edge_tensor,
                'color_un': un_cloth_tensor,
                'edge_un': un_cloth_edge_tensor,
                'label': parse_tensor,
                'pose': pose_tensor,
                'densepose': dense_tensor,
            }
        else:
            return {
                'image': img_tensor,
                'color': cloth_tensor,
                'edge': cloth_edge_tensor,
                'p_name': im_name,
                'c_name': c_name,
            }

    def __len__(self) -> int:
        return len(self.img_names)


def get_transform(train, method=Image.BICUBIC, normalize=True):
    transform_list = []

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
    try:
        ow, oh = img.size  # PIL
    except Exception:
        oh, ow = img.shape  # numpy
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
