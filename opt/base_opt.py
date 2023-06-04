import argparse
from pathlib import Path

from utils.general import increment_path, yaml_save


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def parse_opt(self, save: bool = True):
        opt = self._parse_opt()

        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False, sep='-')

        # save to the disk        
        if save:
            Path(opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            yaml_save(Path(opt.save_dir) / 'opt.yaml', vars(opt))

        return opt

    def _parse_opt(self, known: bool = False) -> None:
        self._add_args()

        return self.parser.parse_known_args()[0] if known else self.parser.parse_args()
    
    def _add_args(self):
        # For experiment
        self.parser.add_argument('--project', default='runs/train', help='save to project/name')  
        self.parser.add_argument('--name', default='exp', help='save to project/name')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    

        # For data
        self.parser.add_argument('--dataroot', type=str, help='train dataset path')
        self.parser.add_argument('--valroot', type=str, help='val/test dataset path')
        self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        self.parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')    
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
