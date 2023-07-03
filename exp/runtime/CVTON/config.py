import argparse
import math

semantic_cloth_labels = [
    [128, 0, 128],
    [128, 128, 64],
    [128, 128, 192],
    [0, 255, 0],
    [0, 128, 128], # dress
    [128, 128, 128], # something upper?
    
    [0, 0, 0], # bg
    
    [0, 128, 0], # hair
    [0, 64, 0], # left leg?
    [128, 128, 0], # right hand
    [0, 192, 0], # left foot
    [128, 0, 192], # head
    [0, 0, 192], # legs / skirt?
    [0, 64, 128], # skirt?
    [128, 0, 64], # left hand
    [0, 192, 128], # right foot
    [0, 0, 128],
    [0, 128, 64],
    [0, 0, 64],
    [0, 128, 192]
]

semantic_densepose_labels = [
    [0, 0, 0],
	[105, 105, 105],
	[85, 107, 47],
	[139, 69, 19],
	[72, 61, 139],
	[0, 128, 0],
	[154, 205, 50],
	[0, 0, 139],
	[255, 69, 0],
	[255, 165, 0],
	[255, 255, 0],
	[0, 255, 0],
	[186, 85, 211],
	[0, 255, 127],
	[220, 20, 60],
	[0, 191, 255],
	[0, 0, 255],
	[216, 191, 216],
	[255, 0, 255],
	[30, 144, 255],
	[219, 112, 147],
	[240, 230, 140],
	[255, 20, 147],
	[255, 160, 122],
	[127, 255, 212]
]

semantic_body_labels = [
    [127, 127, 127],
    [0, 255, 255],
    [255, 255, 0],
    [127, 127, 0],
    [255, 127, 127],
    [0, 255, 0],
    [0, 0, 0],
    [255, 127, 0],
    [0, 0, 255],
    [127, 255, 127],
    [0, 127, 255],
    [127, 0, 255],
    [255, 255, 127],
    [255, 0, 0],
    [255, 0, 255]
]


def get_test_arguments():
    opt = read_arguments(train=False)
    opt.n = "C-VTON-VITON"
    opt.dataset = "viton"
    opt.batch_size = 16 
    opt.which_iter = "best"
    opt.segmentation = ["densepose"]
    opt.transform_cloth = True
    opt.bpgm_id = "256_26_3_viton"
    opt.img_size = [256, 192]
    opt.label_nc = [len(semantic_body_labels), len(semantic_cloth_labels), len(semantic_densepose_labels)]
    opt.semantic_nc = [label_nc + 1 for label_nc in opt.label_nc]
    return opt


def read_arguments(train=False, whole=False):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')
    
    opt = parser.parse_args()
    set_dataset_default_lm(opt, parser)
    
    opt.phase = 'train' if train else 'test'
    opt.phase += "_whole" if whole else ""    
    return opt


def add_all_arguments(parser, train):
    #--- general options ---
    parser.add_argument('--name', "-n", type=str, help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--no_spectral_norm', action='store_true', help='this option deactivates spectral norm in all layers')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--dataset', type=str, default="mpv", help="Dataset to use.")
    parser.add_argument('--dataroot', type=str, default='./data/mpv/', help='path to dataset root')
    parser.add_argument('--img_size', type=int, default=256, help='image size.')
    parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')
    parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help="GPUs to use for training / inference")

    # for generator
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=32, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--param_free_norm', type=str, default='batch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true', help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--z_dim', type=int, default=9, help="dimension of the latent z vector")
    
    parser.add_argument('--val_size', type=float, default=0.05, help="Validation set size (fraction - not int).")
    parser.add_argument('--train_size', type=float, default=0.95, help="Train set size (fraction - not int).")
    parser.add_argument('--transform_cloth', action='store_true', help="Whether to feed a transformed cloth to the OASIS architecture.")
    parser.add_argument('--bpgm_id', type=str, default="256_3_5", help="BPGM identification for pretrained weights loading.")
    parser.add_argument('--seg_edit_id', type=str, default="256", help="SEG_EDIT model identification for pretrained weights loading.")
    parser.add_argument('--segmentation', nargs='+', default=["body"], help="Which segmentations to use for conditioning. {body, cloth, densepose}")

    parser.add_argument('--no_seg', action='store_true', default=False, help='whether to train the model without masking clothing')
    parser.add_argument("--no_bg", action='store_true', default=False, help="whether to remove the background in I_m")

    if train:
        parser.add_argument('--freq_print', type=int, default=1000, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=10000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=10000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2000, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=5000, help='frequency of saving the fid score (in training iterations)')
        parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0001')
        parser.add_argument('--lr_d', type=float, default=0.0004, help='D learning rate, default=0.0004')

        parser.add_argument('--channels_D', type=int, default=64, help='# of discrim filters in first conv layer in discriminator')
        parser.add_argument('--add_vgg_loss', action='store_true', help='if specified, add VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for VGG loss')
        parser.add_argument('--add_lpips_loss', action='store_true', help='if specified, add LPIPS feature matching loss')
        parser.add_argument('--lambda_lpips', type=float, default=1.0, help='weight for LPIPS loss')
        parser.add_argument('--add_l1_loss', action='store_true', help='if specified, add L1 loss')
        parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--add_d_loss', action="store_true", help="if specified, add segmentation discriminator loss")
        parser.add_argument('--add_cd_loss', action="store_true", help="if specified, add conditional discriminator loss")
        parser.add_argument('--add_pd_loss', action="store_true", help="if specified, add patch discriminator loss")
        parser.add_argument('--patch_size', type=int, default=0, help="patch size for patch discriminator")
        parser.add_argument('--no_balancing_inloss', action='store_true', default=False, help='if specified, do *not* use class balancing in the loss function')
        parser.add_argument('--no_labelmix', action='store_true', default=False, help='if specified, do *not* use LabelMix')
        parser.add_argument('--lambda_labelmix', type=float, default=10.0, help='weight for LabelMix regularization')
        
    else:
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves testing results here.')
        
    return parser


def set_dataset_default_lm(opt, parser):
    if opt.dataset == "mpv":
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(num_res_blocks=int(math.log(opt.img_size, 2)) - 2)
        parser.set_defaults(dataroot="./data/mpv")
        parser.set_defaults(patch_size=opt.img_size // 4)
    elif opt.dataset == "viton":
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(num_res_blocks=int(math.log(opt.img_size, 2)) - 2)
        parser.set_defaults(dataroot="./data/viton")
        parser.set_defaults(patch_size=opt.img_size // 4)
    else:
        raise NotImplementedError