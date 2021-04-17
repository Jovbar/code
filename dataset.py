# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:00:05 2021

@author: Administrator
"""
import torch
import random
import glob
import torchvision.transforms as transforms 
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class My_datasets(Dataset):
    def __init__(self,txt_path,train_flag=True,):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        self.valid_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])

    def get_images(self,txt_path):
        #txt_path = 'D:\\TASK\\YB\\wjr\\data\\train.txt'此处只是设置了参数txt_path,并不赋值
        #self.imgs_info = []
        with open(txt_path, 'r', encoding='utf-8') as f:
#            for x in f.readlines():
#                self.imgs_info.append(x.strip("\n"))
#            random.shuffle(self.imgs_info)
            imgs_info = f.readlines()#读取file
            imgs_info = list(map(lambda x:x.strip('\t').split(), imgs_info))
            #将每一行的元素变为list，strip()删除的字符,按照split()中的符号进行每行元素分割为list的元素
        return imgs_info

    def padding_black(self, img):

        w, h  = img.size

        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = 224

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self,index):
        img_path = self.imgs_info[index][0]
        label = self.imgs_info[index][1]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.valid_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)

'''
当模块被直接运行时，__name__ =__main__,以下代码块将被运行，
当模块是被导入时，__name__ = dataset,代码块不被运行。
'''
if __name__ == "__main__":
    train_dataset = My_datasets("train.txt", True)
#    valid_dataset = My_datasets("valid.txt", False)
    print("训练数据个数：", len(train_dataset))
#    print('验证数据个数：',len(valid_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label)

