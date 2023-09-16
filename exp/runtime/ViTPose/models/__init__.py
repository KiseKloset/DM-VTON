import os.path as osp
import sys

from ViTPose.utils.post_processing import *
from ViTPose.utils.top_down_eval import keypoints_from_heatmaps, pose_pck_accuracy
from ViTPose.utils.util import constant_init, load_checkpoint, normal_init, resize
