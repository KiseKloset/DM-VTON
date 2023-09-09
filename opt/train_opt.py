from .base_opt import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def _add_args(self) -> None:
        super()._add_args()

        # For logging
        self.parser.add_argument(
            '--print_step',
            type=int,
            default=100,
            help='frequency of print training results on screen',
        )
        self.parser.add_argument(
            '--sample_step', type=int, default=100, help='frequency of sample training results'
        )
        self.parser.add_argument(
            '--save_period',
            type=int,
            default=20,
            help='frequency of saving checkpoints at the end of epochs',
        )

        # For training
        self.parser.add_argument(
            '--resume', nargs='?', const=True, default=False, help='resume training'
        )
        self.parser.add_argument(
            '--use_dropout', action='store_true', help='use dropout for the generator'
        )
        self.parser.add_argument(
            '--align_corners', action='store_true', help='align corners for grid_sample'
        )
        self.parser.add_argument(
            '--verbose', action='store_true', default=False, help='toggles verbose'
        )
        self.parser.add_argument('--local_rank', type=int, default=-1)
        self.parser.add_argument(
            '--optimizer',
            type=str,
            choices=['SGD', 'Adam', 'AdamW'],
            default='Adam',
            help='optimizer',
        )
        self.parser.add_argument(
            '--niter', type=int, default=50, help='number of epochs at starting learning rate'
        )
        self.parser.add_argument(
            '--niter_decay',
            type=int,
            default=50,
            help='number of epochs to linearly decay learning rate to zero',
        )
        self.parser.add_argument(
            '--momentum', type=float, default=0.5, help='momentum term of optimizer'
        )
        self.parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')

        # Checkpoints
        self.parser.add_argument(
            '--pb_warp_checkpoint',
            type=str,
            help='load the pretrained model from the specified location',
        )
        self.parser.add_argument(
            '--pb_gen_checkpoint',
            type=str,
            help='load the pretrained model from the specified location',
        )
        self.parser.add_argument(
            '--pf_warp_checkpoint',
            type=str,
            help='load the pretrained model from the specified location',
        )
        self.parser.add_argument(
            '--pf_gen_checkpoint',
            type=str,
            help='load the pretrained model from the specified location',
        )
