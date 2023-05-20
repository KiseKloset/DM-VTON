from .base_opt import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def _add_args(self) -> None:
        super()._add_args()            

        # For logging
        self.parser.add_argument('--print_step',  type=int, default=100, help='frequency of print training results on screen')
        self.parser.add_argument('--sample_step',  type=int, default=100, help='frequency of sample training results')
        self.parser.add_argument('--save_period', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')      

        # For training
        self.parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume training')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--align_corners', action='store_true', help='align corners for grid_sample')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--local_rank', type=int, default=-1)
        self.parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--momentum', type=float, default=0.5, help='momentum term of optimizer')
        self.parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')

        # Checkpoints
        self.parser.add_argument('--pb_warp_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--pb_gen_checkpoint', type=str,  help='load the pretrained model from the specified location')
        self.parser.add_argument('--pf_warp_checkpoint', type=str, help='load the pretrained model from the specified location')
        self.parser.add_argument('--pf_gen_checkpoint', type=str,  help='load the pretrained model from the specified location')

        # self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')  
        
        # input/output sizes       
        # self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        # self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        # self.parser.add_argument('--label_nc', type=int, default=20, help='# of input label channels')
        # self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        # self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        # self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

        # for displays
        # self.parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
        # self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        # self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        # self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        # self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        # self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        # self.parser.add_argument('--n_blocks_global', type=int, default=4, help='number of residual blocks in the global generator network')
        # self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        # self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        # self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        
        # self.parser.add_argument('--tv_weight', type=float, default=0.1, help='weight for TV loss')

        # self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        # self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')