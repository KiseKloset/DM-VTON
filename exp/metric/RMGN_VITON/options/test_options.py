from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--warp_checkpoint', type=str, default='checkpoints/RMGN_warp_epoch_030.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='checkpoints/RMGN_gen_epoch_030.pth', help='load the pretrained model from the specified location')
        self.isTrain = False