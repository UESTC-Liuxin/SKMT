# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/9/27 下午3:32
"""

from models.backbone import atrous_resnet,xception

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return atrous_resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError