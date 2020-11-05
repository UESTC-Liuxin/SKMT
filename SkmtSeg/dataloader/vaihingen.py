# -*- coding: utf-8 -*-
"""
@description:

@author: LiuXin
@contact: xinliu1996@163.com
@Created on: 2020/11/1 下午8:54
"""

from dataloader.base import *
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

class VaiHinGen(Base):

    def __init__(self,root,split,logger=None,outer_size=1024,centre_size=512):
        super(VaiHinGen,self).__init__()
        self.root = root
        self.split = split
        self.outer_size = outer_size
        self.centre_size= centre_size

        self.image_list = []
        self.label_list = []

        # mean, std
        self.image_mean = [0.38083647, 0.33612353, 0.35943412]
        self.image_std = [0.10240941, 0.10278902, 0.10292039]

        self.ignore_label = 255

        #read files
        for image_fp in os.listdir(os.path.join(self.root, self.split, 'image')):
            # IRRG img path
            image_path = os.path.join(self.root, self.split, 'image', image_fp)
            # Label path
            label_fp = image_fp.replace(".tif","_noBoundary.tif")
            label_path = os.path.join(self.root, self.split, 'label', label_fp)

            self.image_list.append(image_path)
            self.label_list.append(label_path)

        print('Vaihingen dataset have {} images.'.format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if(self.split=="trainl"):
            return self.get_train(index)
        else:
            return self.get_test(index)


    def get_train(self,index):
        """

        :param index:
        :return:
        """
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        name = os.path.basename(image_path)
        img = Image.open(image_path)
        label = Image.open(label_path)

        # 进行预处理
        image = self.MirroPadding(img,
                                        (img.size[0]+self.centre_size,img.size[1]+self.centre_size))
        label = self.MirroPadding(label,
                                        (label.size[0]+self.centre_size,label.size[1]+self.centre_size))

        # 先随机切1024
        image, label = self.RandomCrop(image, label)
        image, label = self.Transoform(image, label)
        # 随机取一块做伸缩
        image, label = self.RandomScaleCrop(image, label,image.size[0])


        # 从1024中切出中心512
        centre_img, centre_label = self.Crop(image, label, crop_size=self.centre_size,
                                             coordinate=(int(self.centre_size / 4), int(self.centre_size / 4)))
        # 将1024尺寸变为512
        outer_img = image.resize((self.centre_size, self.centre_size), Image.BILINEAR)
        outer_label = label.resize((self.centre_size, self.centre_size), Image.NEAREST)

        outer_img, outer_label = self.Normalize(outer_img, outer_label)
        centre_img, centre_label=self.Normalize(centre_img, centre_label)


        #TODO:显示大图
        visualize(outer_img, 'outer_img')
        # TODO:显示大图标签
        visualize(outer_label, 'outer_label')
        #TODO:显示小图
        visualize(centre_img, 'centre_img')
        # TODO:显示小图标签
        visualize(centre_label, 'centre_label')


        # 进行编解码
        outer_label = np.asarray(outer_label, dtype=np.uint8)
        outer_label = self.encode_segmap(outer_label).astype(np.uint8)  # numpy

        # 进行编解码
        centre_label = np.asarray(centre_label, dtype=np.uint8)
        centre_label = self.encode_segmap(centre_label).astype(np.uint8)  # numpy

        # 变为tensor
        outer_img, outer_label = self.toTensor(outer_img, outer_label)
        centre_img, centre_label = self.toTensor(centre_img, centre_label)

        #返回字典
        return {'outer_imgs':outer_img,'centre_imgs':centre_img,
                'outer_labels':outer_label,'centre_labels':centre_label,'names':name}


    def get_test(self,index):
        """

        :param index:
        :return:
        """
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        name = os.path.basename(image_path)
        source_img = Image.open(image_path)
        source_label = Image.open(label_path)

        outer_imgs=[]
        outer_labels=[]
        centre_imgs=[]
        centre_labels=[]

        #m表示w能切能切centre_size的个数,向上取整
        m=int(math.ceil(source_img.size[0]/self.centre_size))
        n=int(math.ceil(source_img.size[1]/self.centre_size))

        # TODO:在测试时对图片进行镜像扩充
        # 实际的尺寸对四周进行了扩充，保证每个部位能够被测试到
        # 为了保证信息的完整性，可以
        img = self.MirroPadding(source_img,((m+1)*self.centre_size,(n+1)*self.centre_size))
        label=self.MirroPadding(source_label,((m+1)*self.centre_size,(n+1)*self.centre_size))

        #一行一行的切
        for h_index in range(n):
            for w_index in range(m):
                x= w_index * self.centre_size
                y = h_index * self.centre_size
                outer_img, outer_label = self.Crop(img, label, self.outer_size, coordinate=(x, y))
                centre_img, centre_label = self.Crop(outer_img, outer_label, crop_size=self.centre_size,
                                                     coordinate=(int(self.centre_size /2), int(self.centre_size /2)))
                outer_img, outer_label = self.Normalize(outer_img, outer_label)
                centre_img, centre_label = self.Normalize(centre_img, centre_label)

                # 进行编解码
                outer_label = np.asarray(outer_label, dtype=np.uint8)
                outer_label = self.encode_segmap(outer_label).astype(np.uint8)  # numpy

                # 进行编解码
                centre_label = np.asarray(centre_label, dtype=np.uint8)
                centre_label = self.encode_segmap(centre_label).astype(np.uint8)  # numpy

                # 变为tensor

                outer_img, outer_label = self.toTensor(outer_img, outer_label)
                centre_img, centre_label = self.toTensor(centre_img, centre_label)

                outer_imgs.append(outer_img)
                outer_labels.append(outer_label)
                centre_imgs.append(centre_img)
                centre_labels.append(centre_label)


        # 进行编解码
        source_label = np.asarray(source_label, dtype=np.uint8)
        source_label = self.encode_segmap(source_label).astype(np.uint8)  # numpy
        source_img, source_label = self.toTensor(source_img,source_label)
        return {'outer_imgs':outer_imgs,'centre_imgs':centre_imgs,
                'outer_labels':outer_labels,'centre_labels':centre_labels,
                'source_img':source_img,'source_label':source_label,
                'names':[name]}


    def ConvergeLable(self,preds,label):
        #n表示h ，m表示w
        n=int(math.ceil(label.shape[0]/self.centre_size))
        m=int(math.ceil(label.shape[1]/self.centre_size))

        pred_label = np.zeros((self.centre_size * n, self.centre_size * m)).astype(np.uint8)
        k=0
        for h_index in range(n):
            for w_index in range(m):
                x= w_index * self.centre_size
                y = h_index * self.centre_size
                pred_label[y:y + self.centre_size, x:x + self.centre_size] = preds[k]
                k+=1
        pred_label=pred_label[:label.shape[0],:label.shape[1]]

        # pred_label=self.decode_segmap(pred_label)
        # visualize(pred_label,"converge")
        return pred_label

    # TODO:用于调试
    # def Converge(self,imgs,m,n):
    #     mode=imgs[0].mode
    #     img = Image.new(mode,(self.centre_size*m,self.centre_size*n))
    #     k=0
    #     for h_index in range(n):
    #         for w_index in range(m):
    #             x= w_index * self.centre_size
    #             y = h_index * self.centre_size
    #             img.paste(imgs[k],(x,y))
    #             k+=1
    #     return img



    #TODO:镜像padding，并取出中心部分
    def MirroPadding(self,image,size):
        #
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

        x = image.size[0] - self.centre_size/2
        y = image.size[1] - self.centre_size/2
        img = img.crop((x, y, x + size[0], y + size[1]))
        return img


    def Transoform(self,img,label):
        """

        :param img:
        :param label:
        :param train:
        :return:
        """
        image, label = self.RandomHorizontalFlip(img, label)
        image, label = self.RandomGaussianBlur(image, label)
        image, label = self.RandomRotate(image, label,10)
        visualize(image,"rotate")
        return image, label

    def RandomCrop(self,image,label):
        w, h = image.size
        x = random.randint(0, int(w-self.outer_size))
        y = random.randint(0, int(h-self.outer_size))
        image = image.crop((x, y, x + self.outer_size, y + self.outer_size))
        label = label.crop((x, y, x + self.outer_size, y + self.outer_size))

        return image, label


   # get_ISPRS and encode_segmap generate label map
    @classmethod
    def get_ISPRS(cls):
        return np.asarray(
            [
              [255, 255, 255],  # 不透水面
              [  0,   0, 255],  # 建筑物
              [  0, 255, 255],  # 低植被
              [  0, 255,   0],  # 树
              [255, 255,   0],  # 车
              [255,   0,   0],  # Clutter/background
              [  0,   0,   0]   # ignore
            ]
        )

    @classmethod
    def encode_segmap(cls,mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

        for ii, label in enumerate(cls.get_ISPRS()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        label_mask[label_mask == 6] = 255
        # plt.imshow(label_mask)
        # plt.title('Remote Sense')
        # pylab.show()
        return label_mask


    @classmethod
    def decode_segmap(cls,label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = cls.get_ISPRS()
        n_classes = len(label_colours)

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

#TODO:用于调试的visualize代码，观察取的图片和裁剪的图片是否有问题
def visualize(img,tag):

    if(isinstance(img,Image.Image)):
        img=np.array(img).astype(np.uint8)

    plt.title(tag)
    plt.imshow(img)
    plt.show()





if __name__ =="__main__":
    root="/home/liuxin/Documents/CV/Project/RemoteSensor/RemoteSensorSeg/data/VAI"
    train_set = VaiHinGen(root=root, split='trainl', outer_size=1024, centre_size=512)
    train_loader = DataLoader(train_set, batch_size=1, drop_last=True, shuffle=True)
    for iter,batch in enumerate(train_loader):
        pass
