from .test_base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='checkpoints_fs/PFAFN_e2e/PFAFN_warp_epoch_007.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='checkpoints_fs/PFAFN_e2e/PFAFN_gen_epoch_007.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--lr', type=float, default=0, help='initial learning rate for adam')

        self.isTrain = False
