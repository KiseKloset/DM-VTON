
import torch
from data.aligned_dataset import AlignedDataset
from options.train_options import TrainOptions
from torchvision import utils
from data.dresscode_dataset import DressCodeDataset

name = 'aligned'
opt = TrainOptions().parse()
opt.dataroot = '../dataset/Flow-Style-VTON/VITON_traindata'
opt.batchSize = 1
opt.resize_or_crop = 'None'

dataset = AlignedDataset()
dataset.initialize(opt)


# name = 'dress_code'
# dataset = DressCodeDataset(dataroot_path='../dataset/DressCode', phase='train', 
#                             category=['upper_body'], size=(256, 192))

'''
Person: torch.Size([3, 256, 192]) tensor(-1.) tensor(1.)
Cloth: torch.Size([3, 256, 192]) tensor(-1.) tensor(1.)
Edge: torch.Size([1, 256, 192]) tensor(0.) tensor(1.)
Cloth un: torch.Size([3, 256, 192]) tensor(-0.9294) tensor(1.)
Edge un: torch.Size([1, 256, 192]) tensor(0.) tensor(1.)
Parse: torch.Size([1, 256, 192]) tensor(0.) tensor(13.)
Pose: torch.Size([18, 256, 192]) tensor(-1.) tensor(1.)
Densepose: torch.Size([1, 256, 192]) tensor(0.) tensor(24.)
'''

'''
Person: torch.Size([3, 256, 192]) tensor(-0.8980) tensor(1.)
Cloth: torch.Size([3, 256, 192]) tensor(-0.8510) tensor(1.)
Edge: torch.Size([1, 256, 192]) tensor(0.) tensor(1.)
Cloth un: torch.Size([3, 256, 192]) tensor(-0.9922) tensor(1.)
Edge un: torch.Size([1, 256, 192]) tensor(0.) tensor(1.)
Parse: torch.Size([1, 256, 192]) tensor(0.) tensor(13.)
Pose: torch.Size([18, 256, 192]) tensor(-1.) tensor(1.)
Densepose: torch.Size([1, 256, 192]) tensor(0.) tensor(24.)
'''
i=0
for data in dataset:
    image = data['image']
    print('Person:', image.shape, image.min(), image.max())
    color = data['color']
    print('Cloth:', color.shape, color.min(), color.max())
    edge = data['edge']
    print('Edge:', edge.shape, edge.min(), edge.max())
    color_un = data['color_un']
    print('Cloth un:', color_un.shape, color_un.min(), color_un.max())
    edge_un = data['edge_un']
    print('Edge un:', edge_un.shape, edge_un.min(), edge_un.max())
    label = data['label']
    print('Parse:', label.shape, label.min(), label.max())

    pose = data['pose']
    print('Pose:', pose.shape, pose.min(), pose.max())
    densepose = data['densepose']
    print('Densepose:', densepose.shape, densepose.min(), densepose.max())
    combine3 = torch.cat([image, color, color_un], -1).squeeze() 
    combine1 = torch.cat([edge, edge_un, densepose], -1).squeeze() 
    parse = torch.cat([(label == i).float() for i in range(14)], -1).squeeze()
    pose_map = torch.cat([pose[i] for i in range(len(pose))], -1).squeeze()
    
    t_mask = (data['label'] == 7).float()
    a = data['label'] * (1 - t_mask) + t_mask * 4
    person_clothes_edge = (a == 4).float()

    # utils.save_image(
    #     combine1,
    #     f'{name}_dataset_1.png',
    #     nrow=1,
    #     normalize=True,
    #     value_range=(-1,1),
    # )

    utils.save_image(
        combine3,
        f'{name}_dataset_3.png',
        nrow=1,
        normalize=True,
        value_range=(-1,1),
    )

    utils.save_image(
        parse,
        f'{name}_dataset_parse.png',
        nrow=1,
        normalize=True,
        value_range=(-1,1),
    )

    utils.save_image(
        person_clothes_edge,
        f'{name}_dataset_cloth.png',
        nrow=1,
        normalize=True,
        value_range=(-1,1),
    )

    # utils.save_image(
    #     pose_map,
    #     f'{name}_dataset_pose.png',
    #     nrow=1,
    #     normalize=True,
    #     value_range=(-1,1),
    # )

    if i== 5:
        break
    i+=1