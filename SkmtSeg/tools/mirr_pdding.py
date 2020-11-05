# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/4 下午5:56
"""

import os
import sys
import random
import math
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt  # plt 用于显示图片



def mirr_padding(image,padding_size):
    #
    w,h=image.size
    mode = image.mode
    # print(image.size)
    img = Image.new(mode, (3 * image.size[0], 3 * image.size[1]))
    image_T_B = image.transpose(Image.FLIP_TOP_BOTTOM)
    image_L_R = image.transpose(Image.FLIP_LEFT_RIGHT)
    image_LR_BT = image_L_R.transpose(Image.FLIP_TOP_BOTTOM)

    # 第一行
    img.paste(image_LR_BT, (0, 0))
    img.paste(image_T_B, (image.size[0], 0))
    img.paste(image_LR_BT, (image.size[0] * 2, 0))

    # 第二行
    img.paste(image_L_R, (0, image.size[1]))
    img.paste(image, (image.size[0], image.size[1]))
    img.paste(image_L_R, (image.size[0] * 2, image.size[1]))

    # 第三行
    img.paste(image_LR_BT, (0, image.size[1] * 2))
    img.paste(image_T_B, (image.size[0], image.size[1] * 2))
    img.paste(image_LR_BT, (image.size[0] * 2, image.size[1] * 2))

    x = image.size[0] - padding_size / 2
    y = image.size[1] - padding_size / 2
    img = img.crop((x, y, x + w+ padding_size,y +h+padding_size))
    return img


def read_padding_save(root,slipt):
    set_path = os.path.join(root, slipt)
    train_img = os.path.join(set_path, 'image')
    train_label = os.path.join(set_path, 'label')
    image_savedir= os.path.join(set_path, 'image_mirred')
    label_savedir   = os.path.join(set_path, 'label_mirred')

    if not os.path.exists(image_savedir):
        os.makedirs(image_savedir)
        print(image_savedir)
    if not os.path.exists(label_savedir):
        os.makedirs(label_savedir)

    for img_filename in os.listdir(train_img):
        label_filename=img_filename.replace(".tif","_noBoundary.tif")
        img_path=os.path.join(train_img,img_filename)
        label_path=os.path.join(train_label,label_filename)

        image=Image.open(img_path)
        label=Image.open(label_path)

        image=mirr_padding(image,512)
        label=mirr_padding(label,512)
        visualize(image,'mirr')
        image.save(os.path.join(image_savedir,img_filename))
        label.save(os.path.join(label_savedir,label_filename))





#TODO:用于调试的visualize代码，观察取的图片和裁剪的图片是否有问题
def visualize(img,tag):

    if(isinstance(img,Image.Image)):
        img=np.array(img).astype(np.uint8)

    plt.title(tag)
    plt.imshow(img)
    plt.show()


def main(root):
    #切训练集
    read_padding_save(root,'trainl')
    read_padding_save(root, 'testl')



if __name__ =="__main__":
    main("/media/Program/CV/dataset/VAI")








