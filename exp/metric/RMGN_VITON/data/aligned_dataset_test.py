import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import linecache
import os.path as osp
import json
import numpy as np
import torch
from PIL import ImageDraw

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        if opt.hr:
            self.fine_height=512
            self.fine_width=384           
        else:
            self.fine_height=256
            self.fine_width=192
        
        self.dataset_size = len(open(os.path.join(self.opt.dataroot, 'test_pairs.txt')).readlines())
        

        self.dir_I = os.path.join(opt.dataroot, 'test_img')
        self.dir_C = os.path.join(opt.dataroot, 'test_clothes')
        self.dir_E = os.path.join(opt.dataroot, 'test_edge')       
        

    def __getitem__(self, index):        

        file_path = os.path.join(self.opt.dataroot, 'test_pairs.txt')
        im_name, c_name = linecache.getline(file_path, index+1).strip().split()
        
        I_path = os.path.join(self.dir_I,im_name)
        I = Image.open(I_path).convert('RGB')

        params = get_params(self.opt, I.size)
        transform = get_transform(self.opt, params)
        transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)

        C_path = os.path.join(self.dir_C,c_name)
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E_path = os.path.join(self.dir_E, c_name.split("/")[-1])
        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)
            
        input_dict = { 'image': I_tensor,'clothes': C_tensor, 'edge': E_tensor, 'p_name': im_name, 'c_name': c_name}
        
        return input_dict

    def __len__(self):
        return self.dataset_size 

    def name(self):
        return 'AlignedDataset'