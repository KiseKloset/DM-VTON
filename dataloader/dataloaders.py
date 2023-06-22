import json
import glob
import os
from PIL import Image
from pathlib import Path

import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset, distributed

from utils.augmentations import get_transform
from utils.general import keypoints2posemaps


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
FOLDER_POSTFIXS = {
    'person': '_img',
    'cloth': '_color',
    'cloth_edge': '_edge',
    'parse': '_label',
    'pose': '_pose',
    'densepose': '_densepose',
}
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders


# TODO: Add Distributed training
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
def create_dataloader(
    path,
    batch_size,
    rank: int = -1,
    workers: int = 4,
    shuffle: bool = False,
    resize_or_crop: str = None,
    n_downsample_global: int = 4,
    prefix: str = 'train',
):
    dataset = LoadVITONTrainDataset(
        path=path,
        resize_or_crop=resize_or_crop,
        n_downsample_global=n_downsample_global,
        prefix=prefix,
    )

    # Sampler for distributed
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
    )


def get_image_paths(dir, prefix):
    try:
        f = []  # image files
        for d in dir if isinstance(dir, list) else [dir]:
            p = Path(d)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
            else:
                raise FileNotFoundError(f'{prefix}{p} does not exist')
        img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert img_files, f'{prefix}No images found'
    except Exception as e:
        raise Exception(f'{prefix}Error loading data from {dir}: {e}') from e


class LoadVITONTrainDataset(Dataset):
    def __init__(
        self,
        path: str,
        resize_or_crop: str = None,
        n_downsample_global: int = 4,
        prefix: str = 'train',
    ) -> None:
        self.fine_size = [256, 192] # height x width
        self.radius = 5
        # params = get_params(self.opt, A.size)
        self.transform = get_transform(resize_or_crop=resize_or_crop, n_downsample_global=n_downsample_global)
        self.edge_transform = get_transform(resize_or_crop=resize_or_crop, n_downsample_global=n_downsample_global, method=Image.NEAREST, normalize=False)  

        # person
        person_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['person'])
        self.person_paths = get_image_paths(person_dir)

        # cloth
        cloth_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['cloth'])
        self.cloth_paths = get_image_paths(cloth_dir)

        # cloth edge
        cloth_edge_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['cloth_edge'])
        self.cloth_edge_paths = get_image_paths(cloth_edge_dir)

        # parse
        label_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['parse'])
        self.label_paths = get_image_paths(label_dir)

        # pose
        pose_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['pose'])
        self.pose_paths = get_image_paths(pose_dir)

        # densepose
        densepose_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['densepose'])
        self.densepose_paths = get_image_paths(densepose_dir)

    def __getitem__(self, index: int) -> dict:        
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        label_tensor = self.edge_transform(label) * 255.0

        person_path = self.person_paths[index]
        person = Image.open(person_path).convert('RGB') 
        person_tensor = self.transform(person)
        
        # Paired cloth
        cloth_path = self.cloth_paths[index]
        cloth = Image.open(cloth_path).convert('RGB')
        cloth_tensor = self.transform(cloth)

        edge_path = self.edge_paths[index]
        edge = Image.open(edge_path).convert('L')
        edge_tensor = self.edge_transform(edge)
        
        # Un-paired cloth
        unpaired_index = np.random.randint(14221)
        cloth_unpaired_path = self.C_paths[unpaired_index]
        cloth_unpaired = Image.open(cloth_unpaired_path).convert('RGB')
        cloth_unpaired_tensor = self.transform(cloth_unpaired)

        edge_unpaired_path = self.edge_paths[unpaired_index]
        edge_unpaired = Image.open(edge_unpaired_path).convert('L')
        edge_unpaired_tensor = self.edge_transform(edge_unpaired)
        
        # Pose: 18 keypoints [x0, y0, z0, x1, y1, z1, ...]
        with open(self.pose_paths[index], 'r') as f:
            pose = json.load(f)
            try:
                keypoints = pose['people'][0]['pose_keypoints']
            except IndexError:
                keypoints = [0 for i in range(54)]
            keypoints = torch.Tensor(keypoints).view(-1, 3)
        pose_maps = keypoints2posemaps(keypoints=keypoints, size=self.fine_size, radius=self.radius)
        pose_tensor = torch.stack([self.transform(p)[0] for p in pose_maps])

        # Densepose
        dense_mask = np.load(self.densepose_paths[index]).astype(np.float32)
        dense_tensor = self.edge_transform(dense_mask)

        return { 
            'label': label_tensor, 
            'image': person_tensor,
            'color': cloth_tensor, 
            'edge': edge_tensor,  
            'color_un': cloth_unpaired_tensor,
            'edge_un': edge_unpaired_tensor,  
            'pose': pose_tensor, 
            'densepose': dense_tensor,
            'path': label_path, 
            'img_path': person_path,
            'color_path': cloth_path,
            'color_un_path': cloth_unpaired_path,
        }

    def __len__(self) -> int:
        return len(self.person_paths)


class LoadVITONTestDataset(Dataset):
    def __init__(
        self,
        path: str,
        test_pairs_path: str,
        resize_or_crop: str = None,
        n_downsample_global: int = 4,
        prefix: str = 'test',
    ) -> None:
        self.transform = get_transform(resize_or_crop=resize_or_crop, n_downsample_global=n_downsample_global)  
        self.edge_transform = get_transform(resize_or_crop=resize_or_crop, n_downsample_global=n_downsample_global, method=Image.NEAREST, normalize=False)

        # person
        person_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['person'])
        cloth_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['cloth'])
        cloth_edge_dir = os.path.join(path, prefix + FOLDER_POSTFIXS['cloth_edge'])
        self.person_paths, self.cloth_paths, self.cloth_edge_paths = [], [], []

        with open(test_pairs_path, 'r') as f:
            for line in f.readlines():
                person_name, cloth_name = line.strip().split()
                self.person_paths.append(os.path.join(person_dir, person_name))
                self.cloth_paths.append(os.path.join(cloth_dir, cloth_name))
                self.cloth_edge_paths.append(os.path.join(cloth_edge_dir, cloth_name))
    
    def __getitem__(self, index: int) -> dict:
        # Person
        person = Image.open(self.person_paths[index]).convert('RGB')
        person_tensor = self.transform(person)
        
        # Cloth
        cloth = Image.open(self.cloth_paths[index]).convert('RGB')
        cloth_tensor = self.transform(cloth)

        edge = Image.open(self.edge_paths[index]).convert('L')
        edge_tensor = self.edge_transform(edge)

        return { 
            'image': person_tensor,
            'clothes': cloth_tensor, 
            'edge': edge_tensor, 
            'p_name': os.path.basename(self.person_name[index]), 
            'c_name': os.path.basename(self.cloth_name[index]),
        }

    def __len__(self):
        return len(self.person_paths)




