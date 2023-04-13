from .test_options import TestOptions

class TestVideoOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--is_video',  action='store_true', default=False)
        self.parser.add_argument('--input_video', type=str, default='', help='co lam thi moi co an')
        self.parser.add_argument('--target_clothes', type=str, default='', help='co lam thi moi co an')