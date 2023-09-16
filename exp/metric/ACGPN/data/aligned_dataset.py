import json
import os.path
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_test
from PIL import Image, ImageDraw


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.diction = {}

        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

        # load data list from pairs file
        human_names = []
        cloth_names = []
        i = 0
        with open(os.path.join(opt.dataroot, opt.datapairs)) as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                human_names.append(h_name)
                cloth_names.append(c_name)
        self.human_names = human_names
        self.cloth_names = cloth_names
        self.dataset_size = len(human_names)

        # input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

        # input A test (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        # self.A_paths = sorted(make_dataset_test(self.dir_A))

        # input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        # self.B_paths = sorted(make_dataset(self.dir_B))

        # self.dataset_size = len(self.A_paths)
        # self.build_index(self.B_paths)

        dir_E = '_edge'
        self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
        # self.E_paths = sorted(make_dataset(self.dir_E))
        # self.ER_paths = make_dataset(self.dir_E)

        dir_M = '_mask'
        self.dir_M = os.path.join(opt.dataroot, opt.phase + dir_M)
        # self.M_paths = sorted(make_dataset(self.dir_M))
        # self.MR_paths = make_dataset(self.dir_M)

        dir_MC = '_colormask'
        self.dir_MC = os.path.join(opt.dataroot, opt.phase + dir_MC)
        # self.MC_paths = sorted(make_dataset(self.dir_MC))
        # self.MCR_paths = make_dataset(self.dir_MC)

        dir_C = '_color'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
        # self.C_paths = sorted(make_dataset(self.dir_C))
        # self.CR_paths = make_dataset(self.dir_C)
        # self.build_index(self.C_paths)

        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        # self.A_paths = sorted(make_dataset_test(self.dir_A))

    def random_sample(self, item):
        name = item.split('/')[-1]
        name = name.split('-')[0]
        lst = self.diction[name]
        new_lst = []
        for dir in lst:
            if dir != item:
                new_lst.append(dir)
        return new_lst[np.random.randint(len(new_lst))]

    def build_index(self, dirs):
        for k, dir in enumerate(dirs):
            name = dir.split('/')[-1]
            name = name.split('-')[0]

            # print(name)
            for k, d in enumerate(dirs[max(k - 20, 0) : k + 20]):
                if name in d:
                    if name not in self.diction.keys():
                        self.diction[name] = []
                        self.diction[name].append(d)
                    else:
                        self.diction[name].append(d)

    def __getitem__(self, index):
        train_mask = 9600
        # input A (label maps)
        box = []
        # for k,x in enumerate(self.A_paths):
        #     if '000386' in x :
        #         index=k
        #         break
        test = np.random.randint(2032)
        # for k, s in enumerate(self.B_paths):
        #    if '006581' in s:
        #        test = k
        #        break

        # get names from the pairs file
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]

        # A_path = self.A_paths[index]
        A_path = osp.join(self.dir_A, h_name.replace(".jpg", ".png"))
        A = Image.open(A_path).convert('L')

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        # input B (real images)

        # B_path = self.B_paths[index]
        B_path = osp.join(self.dir_B, h_name)
        name = B_path.split('/')[-1]

        B = Image.open(B_path).convert('RGB')
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)

        # input M (masks)
        M_path = B_path  # self.M_paths[np.random.randint(1)]
        MR_path = B_path  # self.MR_paths[np.random.randint(1)]
        M = Image.open(M_path).convert('L')
        MR = Image.open(MR_path).convert('L')
        M_tensor = transform_A(MR)

        ### input_MC (colorMasks)
        MC_path = B_path  # self.MC_paths[1]
        MCR_path = B_path  # self.MCR_paths[1]
        MCR = Image.open(MCR_path).convert('L')
        MC_tensor = transform_A(MCR)

        ### input_C (color)
        # print(self.C_paths)
        # C_path = self.C_paths[test]
        C_path = osp.join(self.dir_C, c_name)
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        # Edge
        # E_path = self.E_paths[test]
        E_path = osp.join(self.dir_E, c_name)
        # print(E_path)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)

        # Pose
        pose_name = B_path.replace('.jpg', '_keypoints.json').replace('test_img', 'test_pose')
        with open(osp.join(pose_name)) as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white'
                )
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor = pose_map

        input_dict = {
            'label': A_tensor,
            'image': B_tensor,
            'path': A_path,
            'name': h_name,
            'edge': E_tensor,
            'color': C_tensor,
            'mask': M_tensor,
            'colormask': MC_tensor,
            'pose': P_tensor,
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'AlignedDataset'
