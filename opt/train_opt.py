import argparse
import os
from pathlib import Path

from utils.general import increment_path, yaml_save

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def _parse_opt(self, known=False):    
        # experiment specifics
        self.parser.add_argument('--name', default='exp', help='save to project/name')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--project', default='runs/train', help='save to project/name')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes       
        self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        self.parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=20, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--data', type=str,default='dataset/VITON_traindata/')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=4, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        
        self.parser.add_argument('--tv_weight', type=float, default=0.1, help='weight for TV loss')
        self.parser.add_argument('--align_corners', action='store_true', help='align corners for grid_sample')

        # for displays
        self.parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
        self.parser.add_argument('--local_rank', type=int, default=-1)
        self.parser.add_argument('--print_batch_step',  type=int, default=100, help='frequency of print training results on screen')
        self.parser.add_argument('--sample_step',  type=int, default=100, help='frequency of sample training results')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--momentum', type=float, default=0.5, help='momentum term of optimizer')
        self.parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument('--PFAFN_warp_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_gen_checkpoint', type=str,  help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_warp_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_gen_checkpoint', type=str,  help='load the pretrained model from the specified location')
        
        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        return self.parser.parse_known_args()[0] if known else self.parser.parse_args()

    def parse_opt(self, save=True):
        opt = self._parse_opt()

        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False)) #  exist_ok=opt.exist_ok
        opt.save_dir = Path(opt.save_dir)
        
        # save to the disk        
        if save:
            yaml_save(yaml_save(self.opt.save_dir / 'opt.yaml', vars(opt)))

        return opt
