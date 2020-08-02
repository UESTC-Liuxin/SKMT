from pycocotools.coco import COCO ,maskUtils
from mmseg.datasets.skmt import SkmtDataset
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pylab
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataset_root = '/home/liuxin/Documents/CV/dataset/VOCdevkit/Seg/skmt5'
dataType = 'default'
annFile = '{}/annotations/instances_{}.json'.format(dataset_root, dataType)
img_path=os.path.join(dataset_root,'JPEGImages')
seg_img_path=os.path.join(dataset_root,'seg_map')
#读取coco文件
coco = COCO(annFile)

# get all images containing given categories, select one at random


# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))



def get_mask(anns,seg_mask):
    for ann in anns:
        mask = coco.annToMask(ann)
        seg_mask[np.where(mask)]=SkmtDataset.PALETTE[ann['category_id']]
        # return mask
    img = Image.fromarray(seg_mask.astype('uint8')).convert('P')
    return img



        # return mask


def ann_to_segmap(coco):
    imgIds = coco.getImgIds()
    img_infos = coco.loadImgs(imgIds)
    #加载所有图片
    for index,img_info in enumerate(img_infos):
        file_name=os.path.join(img_path,img_info['file_name'])
        img=Image.open(file_name)
        seg_mask=np.zeros((img.size[1],img.size[0],3),dtype=np.int)
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        seg_map=get_mask(anns,seg_mask)
        # seg_map.save(os.path.join(seg_map_path,img_info['file_name'].replace('jpg','png')))


def check_imgs():
    img_files=os.listdir(img_path)
    for file in img_files:
        im=cv2.imread(os.path.join(img_path,file))
        print(im.shape)

def count_anns_category(coco):
    anns = coco.loadAnns(coco.getAnnIds())
    countor_values=[0]*len(SkmtDataset.CLASSES)
    for ann in anns:
        mask = coco.annToMask(ann)
        countor_values[ann['category_id']]+=1
    countor_dict = dict(zip(SkmtDataset.CLASSES, countor_values))
    print(countor_dict)


if __name__ =='__main__':
    # ann_to_segmap(img_infos,seg_img_path)
    # check_imgs()
    # ann_to_segmap(img_infos,seg_img_path)
    count_anns_category(coco)