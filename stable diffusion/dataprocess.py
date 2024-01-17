import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import torchvision.transforms as transforms
import cv2
import parser
from PIL import Image
from torch.autograd import Variable
import torchvision.utils
import difflib
import torch.utils.data as data
import scipy.io as scio


from nltk.tokenize import RegexpTokenizer



class Bird_Dataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize(int(256*76/64))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])

        import pickle
        file = open('C:/Users/TanTian/pythonproject/image_generation/captions_DAMSM.pickle', 'rb')
        data = pickle.load(file)
        self.ixtoword, self.wordtoix=data[2],data[3]
        self.filenames,self.all_captions,self.bboxs,self.attrs= self.load_bird_dataset(self.split)
        self.all_txt=self.all_captions
        self.class_id=np.arange(self.filenames.__len__())

    def __getitem__(self, index):
        img_path = self.filenames[index]
        bbox=self.bboxs[index]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        img = img.crop([x1, y1, x2, y2])


        img= self.transform(img)
        img=self.norm(img)
        img=np.array(img)
        img=torch.tensor(img,dtype=torch.float32)

        cap_idx=random.randint(5*index,5*index+4)
        cap=self.all_captions[cap_idx]
        attr=self.attrs[cap_idx]
        cap_len=(cap.__len__()-cap.count(0))

        attr=np.array(attr)
        if np.count_nonzero(attr[1])==0:
            attr[1]=attr[0]
        if np.count_nonzero(attr[2])==0:
            attr[2]=attr[0]

        attr=torch.tensor(attr,dtype=torch.int32)
        img= Variable(img)
        attr = Variable(attr)

        cap=np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]

        return img,attr,cap,cap_len,class_id

    def __len__(self):
        return len(self.filenames)

    def load_bird_dataset(self,split):
        filenames=[]
        all_captions=[]
        all_attrs=[]
        attrs_exists=False
        if os.path.exists('C:/Users/TanTian/pythonproject/image_generation/birds_data/train_attrs.npy') and os.path.exists('C:/Users/TanTian/pythonproject/image_generation/birds_data/test_attrs.npy'):
            attrs_exists=True
            if self.split=='train':
                all_attrs=np.load('C:/Users/TanTian/pythonproject/image_generation/birds_data/train_attrs.npy',allow_pickle=True)
            else:
                all_attrs=np.load('C:/Users/TanTian/pythonproject/image_generation/birds_data/test_attrs.npy',allow_pickle=True)
        if split=='test':
            f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bird_images_test.txt", "r")  # 设置文件对象
        else:
            f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bird_images_train.txt", "r")  # 设置文件对象
        line = f.readline()
        while line:  # 直到读取完文件
            if not line:
                break
            line = line.replace('\n','') # 去掉换行符，也可以不去
            filenames.append(line)
            line=line.replace('CUB_200_2011/images','text')
            line=line.replace('.jpg','.txt')
            captions_path=line
            with open(captions_path, "r") as cap_f:
                captions = cap_f.read().encode('utf-8').decode('utf8').split('\n')
                cnt_captions=0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    while cap.count('��'):
                        cap = cap.replace('��', ' ')
                    while cap.count('.'):
                        cap = cap.replace('.', '')
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    if tokens_new.__len__() > 24:
                        continue
                    if cnt_captions < 5:
                        all_captions.append(tokens_new)
                        if not attrs_exists:
                            attrs = self.get_attrs(cap)
                            all_attrs.append(attrs)
                    else:
                        break
                    cnt_captions += 1
                if cnt_captions != 5:
                    print('the count of captions is not enough')
                    return 0
            line = f.readline()  # 读取一行文件，包括换行符
        if split=='test':
            bbox_f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bboxs_test.txt", "r")  # 设置文件对象
        else:
            bbox_f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bboxs_train.txt", "r")  # 设置文件对象
        line = bbox_f.readline()
        bboxs=[]
        while line:  # 直到读取完文件
            if not line:
                break
            line = line.replace('\n', '')  # 去掉换行符，也可以不去
            x1,width,x2,hight=line.split(' ')
            x1,width,x2,hight=float(x1),float(width),float(x2),float(hight)
            bboxs.append([x1,width,x2,hight])
            line = bbox_f.readline()  # 读取一行文件，包括换行符
        if attrs_exists==False:
            if self.split == 'train':
                np.save('C:/Users/TanTian/pythonproject/image_generation/birds_data/train_attrs.npy', all_attrs)
            else:
                np.save('C:/Users/TanTian/pythonproject/image_generation/birds_data/test_attrs.npy', all_attrs)
        return filenames,all_captions,bboxs,all_attrs


class Bird_Dataset_AFM(data.Dataset):
    def __init__(self,split):
        self.filenames=[]
        self.all_captions=[]
        self.bboxs=[]
        self.attrs=[]
        self.all_txt=[]
        import pickle
        if split=='train':
            load_file = open("C:/Users/TanTian/pythonproject/image_generation/filenames_train.pickle", "rb")
        elif split=='test':
            load_file = open("C:/Users/TanTian/pythonproject/image_generation/filenames_test.pickle", "rb")
        data = pickle.load(load_file)
        train_dataset = Bird_Dataset('train')
        test_dataset = Bird_Dataset('test')

        import pickle
        file = open('C:/Users/TanTian/pythonproject/image_generation/captions_DAMSM.pickle', 'rb')
        data2 = pickle.load(file)
        self.ixtoword, self.wordtoix=data2[2],data2[3]

        for i in range(train_dataset.__len__()):
            filename=train_dataset.filenames[i]
            filename=filename.replace('./data/birds/CUB_200_2011/images/','')
            filename=filename.replace('.jpg','')
            if filename in data:
                self.filenames.append(train_dataset.filenames[i])
                self.all_captions.append(train_dataset.all_captions[i*5])
                self.all_captions.append(train_dataset.all_captions[i*5+1])
                self.all_captions.append(train_dataset.all_captions[i*5+2])
                self.all_captions.append(train_dataset.all_captions[i*5+3])
                self.all_captions.append(train_dataset.all_captions[i*5+4])
                self.bboxs.append(train_dataset.bboxs[i])
                self.attrs.append(train_dataset.attrs[i*5])
                self.attrs.append(train_dataset.attrs[i*5+1])
                self.attrs.append(train_dataset.attrs[i*5+2])
                self.attrs.append(train_dataset.attrs[i*5+3])
                self.attrs.append(train_dataset.attrs[i*5+4])

                self.all_txt.append(train_dataset.all_txt[i*5])
                self.all_txt.append(train_dataset.all_txt[i*5+1])
                self.all_txt.append(train_dataset.all_txt[i*5+2])
                self.all_txt.append(train_dataset.all_txt[i*5+3])
                self.all_txt.append(train_dataset.all_txt[i*5+4])

        for i in range(test_dataset.__len__()):
            filename = test_dataset.filenames[i]
            filename = filename.replace('C:/Users/TanTian/pythonproject/image_generation/data/birds/CUB_200_2011/images/', '')
            filename = filename.replace('.jpg', '')
            if filename in data:
                self.filenames.append(test_dataset.filenames[i])
                self.all_captions.append(test_dataset.all_captions[i * 5])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 1])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 2])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 3])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 4])
                self.bboxs.append(test_dataset.bboxs[i])
                self.attrs.append(test_dataset.attrs[i * 5])
                self.attrs.append(test_dataset.attrs[i * 5 + 1])
                self.attrs.append(test_dataset.attrs[i * 5 + 2])
                self.attrs.append(test_dataset.attrs[i * 5 + 3])
                self.attrs.append(test_dataset.attrs[i * 5 + 4])

                self.all_txt.append(test_dataset.all_txt[i* 5])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 1])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 2])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 3])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 4])

        self.transform = transforms.Compose([
            transforms.Resize(int(256*76/64))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])
        self.class_id=np.arange(self.filenames.__len__())

    def __getitem__(self, index):
        img_path = self.filenames[index]
        bbox=self.bboxs[index]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        img = img.crop([x1, y1, x2, y2])


        img= self.transform(img)
        img=self.norm(img)
        img=np.array(img)
        img=torch.tensor(img,dtype=torch.float32)

        cap_idx=random.randint(5*index,5*index+4)
        cap=self.all_txt[cap_idx]
        attr = self.attrs[cap_idx]

        attrs=[]
        for a in attr:
            attr_i=''
            for word in a:
                attr_i+=word+' '
            attrs.append(attr_i[:-1])
            if attrs.__len__()==3:
                break
        while attrs.__len__()<3:
            attrs.append('')

        img= Variable(img)

        tot_txt = ''
        for c in cap:
            tot_txt += c + ' '
        return img,attrs,tot_txt

    def __len__(self):
        return len(self.filenames)


