import sys
import os.path as osp

from ViTPose.utils.util import load_checkpoint, resize, constant_init, normal_init
from ViTPose.utils.top_down_eval import keypoints_from_heatmaps, pose_pck_accuracy
from ViTPose.utils.post_processing import *