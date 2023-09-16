import glob
import json
from os import path as osp
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F
# import pandas as pd
# import h5py
import numpy as np

class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.opt = opt
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_imgpath)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_names = []
        c_names = []

        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                if opt.mode == "train":
                   img_name= line.rstrip().replace("png","jpg")
                   c_name = img_name
                else:
                   img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)
        self.img_names = img_names
        self.c_names = dict()
        self.c_names['paired'] = c_names
        self.c_names['unpaired'] = c_names

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            if key == "unpaired":
                continue
            else:
                c_name[key] = self.c_names[key][index].replace("_0.jpg","_1.jpg")
            c[key] = Image.open(osp.join(self.data_path, 'clothes', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.load_width, interpolation=2)(c[key])
            c[key] = self.transform(c[key])  

        pose_name = img_name.replace('.jpg', '_keypoints.jpg')
        pose_rgb = Image.open(osp.join(self.data_path, 'vis_pose', pose_name))
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  

        # load person image
        img = Image.open(osp.join(self.data_path, self.opt.mode+'_img', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        img_agnostic = Image.open(osp.join(self.data_path, 'img_agnostic', img_name))
        img_agnostic = transforms.Resize(self.load_width, interpolation=2)(img_agnostic)
        try:
           img = self.transform(img)
        except:
           print(img_name)
           #raise erro
        img_agnostic = self.transform(img_agnostic)  # [-1,1]
        if self.opt.mode =='train' and random.random()>0.5:
            c['paired'] = torch.flip(c['paired'],[2])
            result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': torch.flip(img,[2]),
            'img_agnostic': torch.flip(img_agnostic,[2]),
           'pose': torch.flip(pose_rgb,[2]),
           'cloth': c,
            }
        else:
            result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_agnostic': img_agnostic,
            'pose': pose_rgb,
            'cloth': c,
            }
        return result
    def __len__(self):
        return len(self.img_names)


class ShapeDataset_T(data.Dataset):
    def __init__(self, opt):
        super(ShapeDataset_T, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = opt.dataset_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.hdf5_data =  h5py.File(opt.dataset_list, 'r')
        self.image_ids = np.genfromtxt(osp.join(opt.dataset_dir, opt.dataset_list), dtype=np.str)
        self.angle_list = range(0, 360, 20)
        self.source_name_id = 0
        self.view_id = 0

    def get_random_target_id(self, source_id):
        target_angle = int(np.random.choice(self.angle_list)/10)
        id_base = source_id.split('_')[0]
        h = source_id.split('_')[-1]
        target_id = '_'.join([id_base, str(target_angle), str(h)])
        return target_id

    def obtain_camera_pose(self, pose):
        inputs_h = pose[0,:,:].unsqueeze(1)/2
        inputs_v = pose[1,:,:].unsqueeze(1)/10
        input_label1 = torch.FloatTensor(18, 1,1).zero_()
        input_label2 = torch.FloatTensor(3, 1,1).zero_()
        #print(inputs_h,inputs_v)
        semantics1 = input_label1.scatter_(0, inputs_h.long(), 1.0).repeat(1,self.load_height, self.load_width)
        semantics2 = input_label2.scatter_(0, inputs_v.long(), 1.0).repeat(1,self.load_height, self.load_width)

        input_label = torch.cat((semantics1, semantics2), 0)
        return input_label

    def __getitem__(self, index):
        source_id = self.image_ids[self.source_name_id]
        source_id = source_id.decode("utf-8") if isinstance(source_id, bytes) else source_id
        target_id = self.get_random_target_id(source_id)
        h = self.angle_list[self.view_id]
        self.view_id = self.view_id+2

        img_source = self.hdf5_data[source_id]['image'][()]
        img_target = self.hdf5_data[target_id]['image'][()]

        img_source = Image.fromarray(np.uint8(img_source))
        img_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_source)

        img_target =  Image.fromarray(np.uint8(img_target))
        img_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_target)

        pose_source = torch.tensor(self.hdf5_data[source_id]['pose'][()]).view(-1, 1, 1)
        pose_source = self.obtain_camera_pose(pose_source)
        pose_target = torch.tensor(self.hdf5_data[target_id]['pose'][()]).view(-1, 1, 1)
        #pose_target = self.obtain_camera_pose(pose_target)
        pose_target[0,:,:]=h/10
        pose_target[1,:,:]=0
        pose_target = self.obtain_camera_pose(pose_target)
        pose_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_source)
        pose_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_target)

        img_source = self.transform(img_source)  # [-1,1]
        img_target = self.transform(img_target)

        result = {
        'img_name': source_id,
        'img_source': img_source,
        'img_target': img_target,
        'pose_source': pose_source,
        'pose_target': pose_target
        }
        return result

    def __len__(self):
        return len(self.image_ids)

# class ShapeDataset(data.Dataset):
#     def __init__(self, opt):
#         super(ShapeDataset, self).__init__()
#         self.load_height = opt.load_height
#         self.load_width = opt.load_width
#         self.semantic_nc = opt.semantic_nc
#         self.data_path = opt.dataset_dir
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#         self.hdf5_data =  h5py.File(opt.dataset_list, 'r')
#         self.image_ids = np.genfromtxt(osp.join(opt.dataset_dir, opt.dataset_list), dtype=np.str)

#         self.angle_list = range(0, 360, 20)



#     def get_random_target_id(self, source_id):
#         target_angle = int(np.random.choice(self.angle_list)/10)
#         id_base = source_id.split('_')[0]
#         h = source_id.split('_')[-1]
#         target_id = '_'.join([id_base, str(target_angle), str(h)])
#         return target_id

#     def obtain_camera_pose(self, pose):
#         inputs_h = pose[0,:,:].unsqueeze(1)/2
#         inputs_v = pose[1,:,:].unsqueeze(1)/10
#         input_label1 = torch.FloatTensor(18, 1,1).zero_()
#         input_label2 = torch.FloatTensor(3, 1,1).zero_()
#         semantics1 = input_label1.scatter_(0, inputs_h.long(), 1.0).repeat(1,self.load_height, self.load_width)
#         semantics2 = input_label2.scatter_(0, inputs_v.long(), 1.0).repeat(1,self.load_height, self.load_width)
#         input_label = torch.cat((semantics1, semantics2), 0)
#         return input_label





#     def __getitem__(self, index):


#         source_id = self.image_ids[index]
#         source_id = source_id.decode("utf-8") if isinstance(source_id, bytes) else source_id
#         target_id = self.get_random_target_id(source_id)


#         img_source = self.hdf5_data[source_id]['image'][()]
#         img_target = self.hdf5_data[target_id]['image'][()]

#         img_source = Image.fromarray(np.uint8(img_source))
#         img_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_source)

#         img_target =  Image.fromarray(np.uint8(img_target))
#         img_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_target)

#         pose_source = torch.tensor(self.hdf5_data[source_id]['pose'][()]).view(-1, 1, 1)
#         pose_source = self.obtain_camera_pose(pose_source)
#         pose_target = torch.tensor(self.hdf5_data[target_id]['pose'][()]).view(-1, 1, 1)
#         pose_target = self.obtain_camera_pose(pose_target)
#         pose_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_source)
#         pose_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_target)
#         img_source = self.transform(img_source)  # [-1,1]
#         img_target = self.transform(img_target)

#         result = {
#         'img_name': source_id,
#         'img_source': img_source,
#         'img_target': img_target,
#         'pose_source': pose_source,
#         'pose_target': pose_target
#         }
#         return result
#     def __len__(self):
#         return len(self.image_ids)
class MPVDataset(data.Dataset):
    def __init__(self, opt):
        super(MPVDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_imgpath)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        print(opt.dataset_list)
        # load data list
        img_names = []
        c_names = []
        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.rstrip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names = dict()
        import random
        random.seed(666)
        self.c_names['paired'] = c_names
        self.c_names['unpaired'] = c_names


    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:
            if key == "unpaired":
               index_un =  random.randint(0,len(self.c_names[key])-1)
               c_name[key] = self.c_names[key][index_un]
            else:
               c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'images', c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.load_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, 'images', c_name[key].replace('.jpg', '_mask.jpg')))
            cm[key] = transforms.Resize(self.load_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            if len(np.array(cm[key]).shape)==3:
              cm_array = np.array(cm[key])[:,:,0]
            else:
                cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)


        pose_name = img_name.replace('.jpg', '_keypoints.jpg')
        pose_rgb = Image.open(osp.join(self.data_path, 'vis_pose', pose_name))
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]


        # load person image
        img = Image.open(osp.join(self.data_path, 'images', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        if random.random()>0.2:
           img_agnostic = Image.open(osp.join(self.data_path, 'img_agnostic', img_name))
        else:
           img_agnostic = Image.open(osp.join(self.data_path, 'img_agnostic_maskmore', img_name))
        img_agnostic = transforms.Resize(self.load_width, interpolation=2)(img_agnostic)
        try:
           img = self.transform(img)
        except:
           print(img_name)
           #raise erro
        img_agnostic = self.transform(img_agnostic)  # [-1,1]
        if random.random()>100:
            c['paired'] = torch.flip(c['paired'],[2])
            cm['paired'] = torch.flip(cm['paired'],[2])
            c['unpaired'] = torch.flip(c['unpaired'],[2])
            cm['unpaired'] = torch.flip(cm['unpaired'],[2])
            result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': torch.flip(img,[2]),
            'img_agnostic': torch.flip(img_agnostic,[2]),
           'pose': torch.flip(pose_rgb,[2]),
           'cloth': c,
           'cloth_mask': cm,
            }
        else:
            result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_agnostic': img_agnostic,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
            }
        return result
    def __len__(self):
        return len(self.img_names)
class FASHIONVDataset_Test(data.Dataset):
    def __init__(self, opt,index=2):
        super(FASHIONVDataset_Test, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = opt.dataset_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.img_names = [x.rstrip() for x in open(opt.dataset_list)]
        self.frame_id = 0
        self.video_id = 99#index
        self.source = 0

    def __getitem__(self, index):
        video_name = self.img_names[self.source]
        img_names = sorted(glob.glob(osp.join(self.data_path,video_name,"*png")))



        img_source_name = img_names[0]
        img_names = sorted(glob.glob(osp.join(self.data_path,self.img_names[self.video_id],"*png")))

        # load person image
        img_source_name1 = img_names[0]
        img_target_name1 = img_names[self.frame_id]
        self.frame_id = self.frame_id + 8
        img_source = Image.open(img_source_name)
        img_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_source)
        img_target = Image.open(img_target_name1)
        img_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_target)

        pose_source = Image.open(img_source_name.replace("test_frames","test_keypoints_noise").replace("png","jpg"))
        pose_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_source)
        pose_target = Image.open(img_target_name1.replace("test_frames","test_keypoints").replace("png","jpg"))
        pose_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_target)

        img_source = self.transform(img_source)  # [-1,1]
        img_target = self.transform(img_target)
        pose_source = self.transform(pose_source)
        pose_target = self.transform(pose_target)

        result = {
        'img_name': "%s_%04d.jpg"%(video_name,self.frame_id),
        'img_source': img_source,
        'img_target': img_target,
        'pose_source': pose_source,
        'pose_target': pose_target
        }
        return result
    def __len__(self):
        return 30#len(self.img_names)
class FASHIONVDataset(data.Dataset):
    def __init__(self, opt):
        super(FASHIONVDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = opt.dataset_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.img_names = [x.rstrip() for x in open(opt.dataset_list)]


    def __getitem__(self, index):
        video_name = self.img_names[index]
        img_names = sorted(glob.glob(osp.join(self.data_path,video_name,"*png")))



        # load person image
        img_source_name = random.choice(img_names[:20])
        img_target_name = random.choice(img_names)
        img_source = Image.open(img_source_name)
        img_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_source)
        img_target = Image.open(img_target_name)
        img_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_target)

        pose_source = Image.open(img_source_name.replace("train_frames","train_keypoints_noise").replace("png","jpg"))
        pose_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_source)
        pose_target = Image.open(img_target_name.replace("train_frames","train_keypoints").replace("png","jpg"))
        pose_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_target)

        img_source = self.transform(img_source)  # [-1,1]
        img_target = self.transform(img_target)
        pose_source = self.transform(pose_source)
        pose_target = self.transform(pose_target)

        result = {
        'img_name': img_source_name,
        'img_source': img_source,
        'img_target': img_target,
        'pose_source': pose_source,
        'pose_target': pose_target
        }
        return result
    def __len__(self):
        return len(self.img_names)
# class FASHIONDataset(data.Dataset):
#     def __init__(self, opt):
#         super(FASHIONDataset, self).__init__()
#         self.load_height = opt.load_height
#         self.load_width = opt.load_width
#         self.semantic_nc = opt.semantic_nc
#         self.data_path = opt.dataset_dir
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#         pairs_file_train = pd.read_csv(opt.dataset_list)
#         size = len(pairs_file_train)
#         pairs = []
#         print('Loading data pairs ...')
#         for i in range(size):
#             pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
#             pairs.append(pair)
#         self.img_names = pairs


#     def __getitem__(self, index):
#         img_source_name, img_target_name = self.img_names[index]


#         # load person image
#         img_source = Image.open(osp.join(self.data_path, 'test', img_source_name))
#         img_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_source)
#         img_target = Image.open(osp.join(self.data_path, 'test', img_target_name))
#         img_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(img_target)

#         pose_source = Image.open(osp.join(self.data_path, 'vis_pose', img_source_name))
#         pose_source = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_source)
#         pose_target = Image.open(osp.join(self.data_path, 'vis_pose', img_target_name))
#         pose_target = transforms.Resize((self.load_height,self.load_width), interpolation=2)(pose_target)

#         img_source = self.transform(img_source)  # [-1,1]
#         img_target = self.transform(img_target)
#         pose_source = self.transform(pose_source)
#         pose_target = self.transform(pose_target)

#         result = {
#         'img_name': img_source_name,
#         'img_source': img_source,
#         'img_target': img_target,
#         'pose_source': pose_source,
#         'pose_target': pose_target
#         }
#         return result
#     def __len__(self):
#         return len(self.img_names)



class DAFDataLoader:
    def __init__(self, opt, dataset):
        super(DAFDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
