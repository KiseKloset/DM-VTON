import argparse
import os

import torch

from utils import utils


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument(
            '--name',
            type=str,
            default='flow',
            help='name of the experiment. It decides where to store samples and models',
        )
        self.parser.add_argument(
            '--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
        )
        self.parser.add_argument('--num_gpus', type=int, default=1, help='the number of gpus')
        self.parser.add_argument(
            '--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here'
        )
        self.parser.add_argument(
            '--norm',
            type=str,
            default='instance',
            help='instance normalization or batch normalization',
        )
        self.parser.add_argument(
            '--use_dropout', action='store_true', help='use dropout for the generator'
        )
        self.parser.add_argument(
            '--data_type',
            default=32,
            type=int,
            choices=[8, 16, 32],
            help="Supported data type i.e. 8, 16, 32 bit",
        )
        self.parser.add_argument(
            '--verbose', action='store_true', default=False, help='toggles verbose'
        )

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.parser.add_argument(
            '--loadSize', type=int, default=512, help='scale images to this size'
        )
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument(
            '--label_nc', type=int, default=20, help='# of input label channels'
        )
        self.parser.add_argument(
            '--input_nc', type=int, default=3, help='# of input image channels'
        )
        self.parser.add_argument(
            '--output_nc', type=int, default=3, help='# of output image channels'
        )

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='dataset/VITON_traindata/')
        self.parser.add_argument(
            '--resize_or_crop',
            type=str,
            default='scale_width',
            help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]',
        )
        self.parser.add_argument(
            '--serial_batches',
            action='store_true',
            help='if true, takes images in order to make batches, otherwise takes them randomly',
        )
        self.parser.add_argument(
            '--no_flip',
            action='store_true',
            help='if specified, do not flip the images for data argumentation',
        )
        self.parser.add_argument(
            '--nThreads', default=1, type=int, help='# threads for loading data'
        )
        self.parser.add_argument(
            '--max_dataset_size',
            type=int,
            default=float("inf"),
            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.',
        )

        # for displays
        self.parser.add_argument(
            '--display_winsize', type=int, default=512, help='display window size'
        )
        self.parser.add_argument(
            '--tf_log',
            action='store_true',
            help='if specified, use tensorboard logging. Requires tensorflow installed',
        )

        # for model
        self.parser.add_argument(
            '--netG', type=str, default='global', help='selects model to use for netG'
        )
        self.parser.add_argument(
            '--ngf', type=int, default=64, help='# of gen filters in first conv layer'
        )
        self.parser.add_argument(
            '--n_downsample_global',
            type=int,
            default=4,
            help='number of downsampling layers in netG',
        )
        self.parser.add_argument(
            '--n_blocks_global',
            type=int,
            default=4,
            help='number of residual blocks in the global generator network',
        )
        self.parser.add_argument(
            '--n_blocks_local',
            type=int,
            default=3,
            help='number of residual blocks in the local enhancer network',
        )
        self.parser.add_argument(
            '--n_local_enhancers', type=int, default=1, help='number of local enhancers to use'
        )
        self.parser.add_argument(
            '--niter_fix_global',
            type=int,
            default=0,
            help='number of epochs that we only train the outmost local enhancer',
        )
        self.parser.add_argument('--tv_weight', type=float, default=0.1, help='weight for TV loss')
        self.parser.add_argument(
            '--align_corners', action='store_true', help='align corners for grid_sample'
        )

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'w') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('{}: {}\n'.format(str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
