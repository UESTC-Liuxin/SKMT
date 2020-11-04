# -*- coding: utf-8 -*-
"""
@description:空洞resnet

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/9/24 下午5:29
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,dilation=1,BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,dilation=dilation,
                               stride=stride, padding=dilation, bias=False)#padding=dilation
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)#
        self.bn3 = BatchNorm(self.expansion * planes)

        self.shortcut = nn.Sequential()#
        if stride != 1 or in_planes != self.expansion * planes: #处理通道和空间尺寸的不同相加
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,output_stride=8,num_classes=10,BatchNorm=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.BatchNorm=BatchNorm
        if output_stride ==16:
            strides=[1,2,2,1] #这里设置的残差组的stride，只应用在每组的第一个残差模块中，剩余的全为1。因此经过4个残差组最后会空间尺度
            dilations=[1,1,1,2] #空洞率
        else:
            strides=[1,1,2,1]  #
            dilations=[1,1,2,4]

        # module
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #前三组，每个残差模块都应用相同的空洞率，第四组，每个残差模块都应用不同的空洞率
        self.layer1 = self._make_layer(block, 64, num_blocks[0], dilation=dilations[0],stride=strides[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], dilation=dilations[1],stride=strides[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2],dilation=dilations[2], stride=strides[2])
        self.layer4 = self._make_MG_unit(block, 512, num_blocks[3], dilation=dilations[3],stride=strides[3])


    def _make_layer(self, block, planes, num_blocks, stride,dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for index,stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride,dilation=dilation,BatchNorm=self.BatchNorm))
            # 这里解释一下为什么要有个expansion，因为resnet的残差结构当中最后一个1x1的卷积的filters=前两个的4倍
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, num_blocks, stride,dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for index,stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride,dilation=int(dilation*math.pow(2,index)),BatchNorm=self.BatchNorm))
            # 这里解释一下为什么要有个expansion，因为resnet的残差结构当中最后一个1x1的卷积的filters=前两个的4倍
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out =self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.maxpool(out)
        out = self.layer1(out)
        low_level_feature =out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out,low_level_feature



def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101(output_stride, BatchNorm):
    return ResNet(Bottleneck, num_blocks=[3, 4, 23, 3],output_stride=output_stride,BatchNorm=BatchNorm)


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])





if __name__ == '__main__':
    from tensorboardX import SummaryWriter

    # TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    # log_dir = 'logs/resnet18/' + TIMESTAMP
    # writer = SummaryWriter(log_dir=log_dir)

    net = ResNet50()
    x=torch.rand(1,3,512, 512)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    x=x.to(device)
    out=net(x)
    print(out.size())
    # with writer:
    #     writer.add_graph(net, (torch.rand(1,3,32, 32),))