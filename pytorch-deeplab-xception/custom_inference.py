# -*- coding: utf-8 -*-
"""
@author: LiuXin
@contact: xinliu1996@163.com
@Created on: DATE{TIME}
"""
import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


# class Inferner(object):
#     def __init__(self,args):
#
#         #define the path of inference result
#         self.save_path=
#         # Define network
#         model = DeepLab(num_classes=self.nclass,
#                         backbone=args.backbone,
#                         output_stride=args.out_stride,
#                         sync_bn=args.sync_bn,
#                         freeze_bn=args.freeze_bn)
#
