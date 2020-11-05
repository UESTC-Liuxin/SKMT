# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/1 下午9:05
"""

import os
import sys
import random
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.multiprocessing
from PIL import Image, ImageOps, ImageFilter


class Base(Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__()


    def Crop(self,image,label,crop_size:int,coordinate:tuple):
        """
        裁剪方法，根据坐标和宽度进行裁剪
        """
        w=image.size[-2]
        h=image.size[-1]

        if(isinstance(coordinate,tuple)):
            x=coordinate[0]
            y=coordinate[1]
        else:
            raise TypeError

        if(isinstance(image,Image.Image)):
            image=image.crop((x, y, x + crop_size, y + crop_size))
            label=label.crop((x, y, x + crop_size, y + crop_size))
        elif(isinstance(image,np.ndarray)):
            image=image[y: y + crop_size][x:x+crop_size]
            label = label[y: y + crop_size][x:x + crop_size]
        else:
            TypeError

        return (image,label)


    def Normalize(self, image, label, div_std=False):
        image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.float32)
        image /= 255
        image -= self.image_mean

        if div_std == True:
            image /= self.std

        return image, label

    def Resize(self,image, label,size):
        if(isinstance(size,tuple)):
            w,h=size
        else:
            w=h=size
        if(isinstance(image,Image.Image)):
            image = image.resize((w, h), Image.BILINEAR)
            label = label.resize((w, h), Image.NEAREST)
        else:
            TypeError

        return image,label

    def toTensor(self, image, label):
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return image, label


    def get_ISPRS(self):
        return np.asarray([])


    def RandomHorizontalFlip(self, image, label):
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label


    def RandomRotate(self, image, label,degree):
        rotate_degree = random.random() * 2 * degree - degree
        image = image.rotate(rotate_degree, Image.BILINEAR,0)
        label = label.rotate(rotate_degree, Image.NEAREST,0)
        return image, label

    def RandomGaussianBlur(self, image, label):
        radius = random.random()
        if radius < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, label

    def RandomScaleCrop(self, image, label,crop_size):
        short_size = random.randint(int(crop_size * 0.5), int(crop_size * 2.0))
        w, h = image.size

        oh = short_size
        ow = int(1.0 * w * oh / h)

        image = image.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size <= crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = image.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)

        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return image, label


        return image, label








