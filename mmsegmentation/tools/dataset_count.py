# -*- coding: utf-8 -*-
"""
@author: LiuXin
@contact: xinliu1996@163.com
@Created on: DATE{TIME}
"""

from pycocotools.coco import COCO ,maskUtils
from mmseg.datasets.skmt import SkmtDataset
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pylab
import cv2


dataset_root = '/home/liuxin/Documents/CV/dataset/VOCdevkit/Seg/skmt5'
dataType = 'default'
annFile = '{}/annotations/instances_{}.json'.format(dataset_root, dataType)
img_path=os.path.join(dataset_root,'JPEGImages')
seg_img_path=os.path.join(dataset_root,'seg_map')
#读取coco文件
coco = COCO(annFile)


def compute(path):
    '''

    Args:
        path:

    Returns:

    '''
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    stdR = np.std(per_image_Rmean)
    stdG = np.std(per_image_Gmean)
    stdB = np.std(per_image_Bmean)
    return R_mean, G_mean, B_mean, stdR, stdG, stdB

def count_anns_category(coco):
    '''count the distribution of annotations

    Args:
        coco:the obj of COCO

    Returns:
        countor_dict: the dict of
    '''
    anns = coco.loadAnns(coco.getAnnIds())
    countor_values=[0]*len(SkmtDataset.CLASSES)
    for ann in anns:
        countor_values[ann['category_id']]+=1
    countor_dict = dict(zip(SkmtDataset.CLASSES, countor_values))
    return countor_dict


if __name__ == '__main__':
    Rmean, Gmean , Bmean,stdR, stdG, stdB= compute(path)
    print(Rmean, Gmean , Bmean,stdR, stdG, stdB)