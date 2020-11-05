# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/10/27 下午5:31
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.auxiliary import build_auxiliary
from modeling.deeplabv3 import DeepLab




class ERFTSN(nn.Module):
    def __init__(self, auxiliary,trunk,img_size=256,num_classes=21,freeze_bn=False):
        super(ERFTSN, self).__init__()
        self.auxiliary=auxiliary
        self.trunk=trunk
        self.img_size=img_size
        self.num_classes=num_classes

        #创建两个相加的权重
        self.weights=nn.Parameter(torch.rand(num_classes,1,1),requires_grad=True)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, inputs:dict):
        centre = self.trunk(inputs['centre_imgs'])
        if(self.auxiliary is not None):
            outer=self.auxiliary(inputs['outer_imgs'])
            out=self.add_centre_outer(outer,centre)
        else:
            outer=None
            out=centre

        return {'outer_out':outer,'centre_out':centre,'out':out}

    def add_centre_outer(self,outer,centre):
        y=x=self.img_size
        auxiliary_centre=outer[...,int(x/4):int(3*x/4),int(y/4):int(3*y/4)]
        auxiliary_centre=F.interpolate(auxiliary_centre, scale_factor=2, mode='bilinear', align_corners=True)
        weights=self.weights.expand(centre.size()[0],self.num_classes,1,1)
        weights=weights.view(centre.size()[0],-1,1,1)

        out=auxiliary_centre*weights+centre*(1-weights)
        return out






    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p



def build_erftsn(auxiliary,trunk, img_size=512,freeze_bn=False,output_stride=16, num_classes=6,
                 auxiliary_backbone='resnet',trunk_backbone='resnet',sync_bn=False,):
    if sync_bn == True:
        BatchNorm = SynchronizedBatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if auxiliary ==None:
        auxiliary=None
    elif auxiliary == 'fcn':
        auxiliary=build_auxiliary(auxiliary='fcn',backbone=auxiliary_backbone,num_classes=num_classes,BatchNorm=BatchNorm)
    elif auxiliary =='deeplab':
        auxiliary =DeepLab(num_classes=num_classes,backbone=auxiliary_backbone,BatchNorm=BatchNorm)
    else:
        raise NotImplementedError

    if(trunk=='deeplab'):
        trunk=DeepLab(num_classes=num_classes,backbone=trunk_backbone,BatchNorm=BatchNorm)
    else:
        raise NotImplementedError

    return ERFTSN(auxiliary,trunk,num_classes=num_classes,img_size=img_size)




if __name__ == "__main__":
    model=build_erftsn(auxiliary='fcn',trunk='deeplab')
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    inputs={'outer':input,'centre':input}
    output = model(inputs)
    print(output.size())


