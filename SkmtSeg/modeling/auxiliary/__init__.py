# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/10/27 下午8:12
"""

from modeling.auxiliary import fcn


def build_auxiliary(auxiliary, num_classes,BatchNorm,backbone='mobilenet'):
    if auxiliary == 'fcn':
        return fcn.FCN(backbone=backbone,BatchNorm=BatchNorm,n_class=num_classes)
    else:
        raise NotImplementedError

