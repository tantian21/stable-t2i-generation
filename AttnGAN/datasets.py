from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import nltk
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Bird_Dataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize(int(256*76/64))])
        self.transform64 = transforms.Compose([
            transforms.Resize(64)])
        self.transform128 = transforms.Compose([
            transforms.Resize(128)])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])
        import pickle
        x = pickle.load(open('data/captions.pickle', 'rb'))
        self.ixtoword = x[2]
        self.wordtoix = x[3]
        self.filenames,self.all_captions,self.bboxs,self.attrs= self.load_bird_dataset(self.split)
        self.all_txt=self.all_captions
        self.all_captions,self.attrs=self.build_dictionary(self.all_captions,self.attrs)
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
        img64=self.transform64(img)
        img128=self.transform128(img)
        img,img64,img128=np.array(img),np.array(img64),np.array(img128)
        img,img64,img128=torch.tensor(img,dtype=torch.float32),torch.tensor(img64,dtype=torch.float32),torch.tensor(img128,dtype=torch.float32),

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
        img,img64,img128= Variable(img),Variable(img64),Variable(img128)
        attr = Variable(attr)

        cap=np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]

        return [img64,img128,img],attr,cap,cap_len,class_id

    def __len__(self):
        return len(self.filenames)

    def load_bird_dataset(self,split):
        filenames=[]
        all_captions=[]
        all_attrs=[]
        attrs_exists=False
        if os.path.exists('./birds_data/train_attrs.npy') and os.path.exists('./birds_data/test_attrs.npy'):
            attrs_exists=True
            if self.split=='train':
                all_attrs=np.load('./birds_data/train_attrs.npy',allow_pickle=True)
            else:
                all_attrs=np.load('./birds_data/test_attrs.npy',allow_pickle=True)
        if split=='test':
            f = open("./birds_data/bird_images_test.txt", "r") 
        else:
            f = open("./birds_data/bird_images_train.txt", "r")  
        line = f.readline()
        while line: 
            if not line:
                break
            line = '.'+line.replace('\n','') 
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
            line = f.readline()  
        if split=='test':
            bbox_f = open("./birds_data/bboxs_test.txt", "r") 
        else:
            bbox_f = open("./birds_data/bboxs_train.txt", "r")  
        line = bbox_f.readline()
        bboxs=[]
        while line: 
            if not line:
                break
            line = line.replace('\n', '')  
            x1,width,x2,hight=line.split(' ')
            x1,width,x2,hight=float(x1),float(width),float(x2),float(hight)
            bboxs.append([x1,width,x2,hight])
            line = bbox_f.readline()  
        if attrs_exists==False:
            if self.split == 'train':
                np.save('./birds_data/train_attrs.npy', all_attrs)
            else:
                np.save('./birds_data/test_attrs.npy', all_attrs)
        return filenames,all_captions,bboxs,all_attrs

    def build_dictionary(self, captions,  all_attrs):
        captions_new = []
        all_attrs_new=[]
        for t in captions:
            rev = []
            for w in t:
                if w not in self.wordtoix.keys():
                    print('word not in wordtoix')
                    continue
                rev.append(self.wordtoix[w])
            while rev.__len__()<25:
                rev.append(0)
            captions_new.append(rev)
        for attrs in all_attrs:
            new_attrs=[]
            for attr in attrs:
                new_attr=[]
                for w in attr:
                    if w not in self.wordtoix.keys():
                        print('word not in wordtoix')
                        continue
                    new_attr.append(self.wordtoix[w])
                if new_attr.__len__()>5:
                    ix = list(np.arange(new_attr.__len__()))  
                    np.random.shuffle(ix)
                    ix = ix[:5]
                    ix = np.sort(ix)
                    new_attr = np.array(new_attr)[ix]
                    new_attr=list(new_attr)
                while new_attr.__len__()<5:
                    new_attr.append(0)
                new_attrs.append(new_attr)
            if new_attrs.__len__()>3:
                ix = list(np.arange(new_attrs.__len__()))  
                np.random.shuffle(ix)
                ix = ix[:3]
                ix = np.sort(ix)
                new_attrs = np.array(new_attrs)[ix]
                new_attrs=list(new_attrs)
            while new_attrs.__len__()<3:
                new_attrs.append([0,0,0,0,0])
            all_attrs_new.append(new_attrs)
        return captions_new, all_attrs_new
class Bird_Dataset_AFM(data.Dataset):
    def __init__(self,split):
        self.filenames=[]
        self.all_captions=[]
        self.bboxs=[]
        self.attrs=[]
        import pickle
        if split=='train':
            load_file = open("./filenames_train.pickle", "rb")
        elif split=='test':
            load_file = open("./filenames_test.pickle", "rb")
        data = pickle.load(load_file)
        train_dataset = Bird_Dataset('train')
        test_dataset = Bird_Dataset('test')

        import pickle
        x = pickle.load(open('data/captions.pickle', 'rb'))
        self.ixtoword = x[2]
        self.wordtoix = x[3]

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

        for i in range(test_dataset.__len__()):
            filename=test_dataset.filenames[i]
            filename=filename.replace('./data/birds/CUB_200_2011/images/','')
            filename=filename.replace('.jpg','')
            if filename in data:
                self.filenames.append(test_dataset.filenames[i])
                self.all_captions.append(test_dataset.all_captions[i*5])
                self.all_captions.append(test_dataset.all_captions[i*5+1])
                self.all_captions.append(test_dataset.all_captions[i*5+2])
                self.all_captions.append(test_dataset.all_captions[i*5+3])
                self.all_captions.append(test_dataset.all_captions[i*5+4])
                self.bboxs.append(test_dataset.bboxs[i])
                self.attrs.append(test_dataset.attrs[i*5])
                self.attrs.append(test_dataset.attrs[i*5+1])
                self.attrs.append(test_dataset.attrs[i*5+2])
                self.attrs.append(test_dataset.attrs[i*5+3])
                self.attrs.append(test_dataset.attrs[i*5+4])

        self.transform = transforms.Compose([
            transforms.Resize(int(256*76/64))])
        self.transform64 = transforms.Compose([
            transforms.Resize(64)])
        self.transform128 = transforms.Compose([
            transforms.Resize(128)])
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
        img64=self.transform64(img)
        img128=self.transform128(img)
        img,img64,img128=np.array(img),np.array(img64),np.array(img128)
        img,img64,img128=torch.tensor(img,dtype=torch.float32),torch.tensor(img64,dtype=torch.float32),torch.tensor(img128,dtype=torch.float32),

        cap_idx=random.randint(5*index,5*index+4)
        cap=self.all_captions[cap_idx]
        attr=self.attrs[cap_idx]
        cap_len=(cap.__len__()-cap.count(0))

        while cap_len==0:
            cap_idx = random.randint(5 * index, 5 * index + 4)
            cap = self.all_captions[cap_idx]
            attr = self.attrs[cap_idx]
            cap_len = (cap.__len__() - cap.count(0))

        attr=np.array(attr)
        if np.count_nonzero(attr[1])==0:
            attr[1]=attr[0]
        if np.count_nonzero(attr[2])==0:
            attr[2]=attr[0]

        attr=torch.tensor(attr,dtype=torch.int32)
        img,img64,img128= Variable(img),Variable(img64),Variable(img128)
        attr = Variable(attr)

        cap=np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]
        return [img64,img128,img],attr,cap,cap_len,class_id

    def __len__(self):
        return len(self.filenames)
