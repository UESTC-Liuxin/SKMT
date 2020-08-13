# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

ratio_train1 = 0.8 #训练集比例
ratio_val1 = 0.2 #测试集比例

ratio_train2 = 0.8 #训练集比例
ratio_val2 = 0.2 #测试集比例

ratio_train3 = 0.8 #训练集比例
ratio_val3 = 0.2 #测试集比例

ratio_train4 = 0.8 #训练集比例
ratio_val4 = 0.2 #测试集比例

ratio_train5 = 0.8 #训练集比例
ratio_val5 = 0.2 #测试集比例
assert (ratio_train1  + ratio_val1) == 1.0
assert (ratio_train2  + ratio_val2) == 1.0
assert (ratio_train3  + ratio_val3) == 1.0
assert (ratio_train4  + ratio_val4) == 1.0
assert (ratio_train5  + ratio_val5) == 1.0

dataset_root='/home/sunny/code/python/'
img_path = os.path.join(dataset_root,'JPEGImages')
f_tra=open(os.path.join(dataset_root,'train.txt'),'wb')
f_val=open(os.path.join(dataset_root,'val.txt'),'wb')

files_list1 = []    
files_list2 = [] 
files_list3 = [] 
files_list4 = [] 
files_list5 = [] 
for file in os.listdir(img_path):
    if file[8]=='1':
        files_list1.append(file)
    elif file[8]=='2':
        files_list2.append(file)
    elif file[8]=='3':
        files_list3.append(file)
    elif file[8]=='4':
        files_list4.append(file)
    elif file[8]=='5':
        files_list5.append(file)
print(len(files_list1))
print(len(files_list2))
print(len(files_list3))
print(len(files_list4))
print(len(files_list5))

train_list = []
val_list = []
if len(files_list1)!=0:
    np.random.shuffle(files_list1) ##打乱文件列表
    cnt_val = round(len(files_list1) * ratio_val1 ,0)
    cnt_train = len(files_list1) - cnt_val
    for i in range(int(cnt_train)):
        train_list.append(files_list1[i])
    for i in range(int(cnt_train) ,int(cnt_train + cnt_val)):
        val_list.append(files_list1[i])

if len(files_list2)!=0:
    np.random.shuffle(files_list2) ##打乱文件列表
    cnt_val = round(len(files_list2) * ratio_val2 ,0)
    cnt_train = len(files_list2) - cnt_val
    for i in range(int(cnt_train)):
        train_list.append(files_list2[i])
    for i in range(int(cnt_train) ,int(cnt_train + cnt_val)):
        val_list.append(files_list2[i])

if len(files_list3)!=0:
    np.random.shuffle(files_list3) ##打乱文件列表
    cnt_val = round(len(files_list3) * ratio_val3 ,0)
    cnt_train = len(files_list3) - cnt_val
    for i in range(int(cnt_train)):
        train_list.append(files_list3[i])
    for i in range(int(cnt_train) ,int(cnt_train + cnt_val)):
        val_list.append(files_list3[i])

if len(files_list4)!=0:
    np.random.shuffle(files_list4) ##打乱文件列表
    cnt_val = round(len(files_list4) * ratio_val4 ,0)
    cnt_train = len(files_list4) - cnt_val
    for i in range(int(cnt_train)):
        train_list.append(files_list4[i])
    for i in range(int(cnt_train) ,int(cnt_train + cnt_val)):
        val_list.append(files_list4[i])

if len(files_list5)!=0:
    np.random.shuffle(files_list5) ##打乱文件列表
    cnt_val = round(len(files_list5) * ratio_val5 ,0)
    cnt_train = len(files_list5) - cnt_val
    for i in range(int(cnt_train)):
        train_list.append(files_list5[i])
    for i in range(int(cnt_train) ,int(cnt_train + cnt_val)):
        val_list.append(files_list5[i])

for i in range(len(train_list)):   
    name = str(train_list[i])
    index = name.rfind('.')
    name = name[:index]+'\n'
    f_tra.write(name.encode())
f_tra.close()

for i in range(len(val_list)):   
    name = str(val_list[i])
    index = name.rfind('.')
    name = name[:index]+'\n'
    
    f_val.write(name.encode())
f_val.close()