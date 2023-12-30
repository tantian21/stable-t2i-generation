import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import random
from scipy.io import loadmat

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
import cv2
import spacy
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def Bird_collate_fn(data,split):
    if split=='train':
        imgs,attribute= data
        imgs=Variable(imgs).to(device)
        attribute=Variable(attribute).to(device)
        return imgs, attribute
    else:
        attribute= data
        attribute=Variable(attribute).to(device)
        return attribute


class Bird_SceneGraphDataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize([256, 256])])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.data = []

        self.bird_parts={'bird':0,'color':0,'body':0,'back':1,'nape':1,'beak':2,'bill':2,'belly':3,'breast':4,'crown':5,'forehead':6,'eye':7,'eyes':7,'leg':8,'legs':8,'wing':9,'wings':9,'wingspan':9,'tail':10,'throat':11,'feather':12,'feathers':12,'plumage':12,'head':13,'neck':14}
        self.adj={'black':0,'dark':0, 'white':1,'bright':1,'light':1,'brown':2,'yellow':3,'gray':4,'grey':4,'red':5,'blue':6,'orange':7,'green':8}
        if self.split=='train':
            self.filenames,self.attributes= self.load_cubbird_102flowers()
        else:
            self.attributes = self.load_test()

    def __getitem__(self, index):
        if self.split=='train':
            img_path = self.filenames[index]
            attribute=self.attributes[index]
            img = Image.open(img_path).convert('RGB')
            img=np.array(img)

            H, W = img.shape[0], img.shape[1]
            img=Image.fromarray(img)
            img= self.transform(img)
            img=self.norm(img)
            img=np.array(img)
            img=torch.tensor(img,dtype=torch.float32)
            attribute=np.array(attribute)
            attribute=torch.tensor(attribute,dtype=torch.int32)

            return img,attribute
        else:
            attribute = self.attributes[index]
            attribute = np.array(attribute)
            attribute = torch.tensor(attribute, dtype=torch.int32)

            return attribute

    def __len__(self):
        if self.split=='train':
            return len(self.filenames)
        else:
            return len(self.attributes)

    def load_cubbird_102flowers(self):
        nlp = spacy.load("en_core_web_sm")
        filenames=[]
        attributes=[]

        f = open("../../data/birds/CUB_200_2011/images.txt", "r")  
        line = f.readline()
        line = line.replace('\n','')
        while line:  
            if not line:
                break
            line = line.replace('\n','')
            line=line.split(' ')[1]
            filenames.append('../../data/birds/CUB_200_2011/images/'+line)
            line = f.readline() 

        attributes = np.loadtxt('../attributes.txt', dtype=int)
        return filenames,attributes

        f = open("../../data/birds/CUB_200_2011/images.txt", "r") 
        line = f.readline()
        line = line.replace('\n','')

        cnt=0
        while line: 
            if cnt%100==0:
                print(cnt)
            cnt+=1
            if not line:
                break
            line = line.replace('\n','') 
            line = line.split(' ')[1]
            line = line.replace('.jpg', '.txt')
            caption_path = '../../data/birds/text/' + line

            attribute=[]
            with open(caption_path, 'r') as caption_f:
                caption_line = caption_f.readline()
                while caption_line:
                    if not caption_line:
                        break
                    caption_line = caption_line.replace('\n','')
                    caption_line = caption_line.replace('��', ' ')
                    caption_line = caption_line.replace('.', '')
                    doc = nlp(caption_line)
                    for token in doc:

                        if token.tag_ == 'JJ':
                            temp = token.text
                            while token.tag_ != 'NN' and token.tag_ != 'NNS':
                                token = token.head
                                if token == token.head:
                                    break
                            if token.text in self.bird_parts.keys() and temp in self.adj.keys():
                                temp=self.bird_parts[token.text]*9+self.adj[temp]+1
                                if temp not in attribute:
                                    attribute.append(temp)
                    caption_line = caption_f.readline()
            while attribute.__len__()<23:
                attribute.append(0)
            attributes.append(attribute)
            line = f.readline()  
        np.savetxt('../attributes.txt', attributes, fmt='%d', delimiter=' ')
        return filenames,attributes

    def load_test(self):
        nlp = spacy.load("en_core_web_sm")
        attributes=[]

        attributes = np.loadtxt('../attributes.txt', dtype=int)
        return attributes
        f = open("../../data/birds/CUB_200_2011/images.txt", "r") 
        line = f.readline()
        line = line.replace('\n', '')

        cnt = 0
        while line: 
            if cnt % 100 == 0:
                print(cnt)
            cnt += 1
            if not line:
                break
            line = line.replace('\n', '')  
            line = line.split(' ')[1]
            line = line.replace('.jpg', '.txt')
            caption_path = '../../data/birds/text/' + line

            attribute = []
            with open(caption_path, 'r') as caption_f:
                caption_line = caption_f.readline()
                while caption_line:
                    if not caption_line:
                        break
                    caption_line = caption_line.replace('\n', '')
                    caption_line = caption_line.replace('��', ' ')
                    caption_line = caption_line.replace('.', '')
                    doc = nlp(caption_line)
                    # print(caption_line)
                    for token in doc:
                        if token.tag_ == 'JJ':
                            temp = token.text
                            while token.tag_ != 'NN' and token.tag_ != 'NNS':
                                token = token.head
                                if token == token.head:
                                    break
                            if token.text in self.bird_parts.keys() and temp in self.adj.keys():
                                temp = self.bird_parts[token.text] * 9 + self.adj[temp] + 1
                                if temp not in attribute:
                                    attribute.append(temp)
                    caption_line = caption_f.readline()
            while attribute.__len__() < 23:
                attribute.append(0)
            attributes.append(attribute)
            line = f.readline()  

        return attributes

def Flower_collate_fn(data,split):
    if split=='train':
        imgs,attribute= data
        imgs=Variable(imgs).to(device)
        attribute=Variable(attribute).to(device)
        return imgs, attribute
    else:
        attribute= data
        attribute=Variable(attribute).to(device)
        return attribute
class Flower_SceneGraphDataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize([256, 256])])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.flower_parts={'petals':0,'petal':0, 'stamen':1, 'stamens':1,'pistil':1,'flower':2,'flowers':2, 'color':2, 'stigma':3,  'center':4, 'pedicel':5,  'leaves':6, 'leaf':6,'edges':7, 'anthers':8,  'spots':9,  'dots':9, 'sepals':10,'filaments':11, 'stem':12}
        self.adj={'yellow':0, 'white':1,  'bright':1, 'pink':2, 'purple':3, 'red':4, 'green':5, 'large':6, 'long':7, 'orange':8, 'dark':9,'black':9, 'small':10, 'thin':11, 'blue':12, 'brown':13}
        #('yellow', 33371), ('white', 28438), ('pink', 21935), ('purple', 17177), ('red', 13013), ('green', 11895), ('large', 7195), ('long', 7125), ('orange', 5485), ('bright', 5063), ('dark', 4278), ('small', 4248), ('light', 3814), ('thin', 3769), ('blue', 3132), ('many', 2954), ('smooth', 2380), ('pale', 1997), ('rounded', 1857), ('wide', 1840), ('brown', 1791), ('petal', 1661), ('pointed', 1647), ('black', 1608), ('short', 1578), ('thick', 1483), ('soft', 1454), ('round', 1402), ('several', 1191),
        #('petals', 86459), ('is', 21946), ('stamen', 17696), ('flower', 12410), ('are', 10726), ('stigma', 4973), ('pistil', 4300), ('center', 4274), ('pedicel', 3933), ('has', 3852), ('petal', 3043), ('leaves', 2952), ('edges', 2306), ('anthers', 2305), ('stamens', 2011), ('flowers', 1860), ('spots', 1663), ('color', 1615), ('sepals', 1524), ('filaments', 1518), ('tips', 1331), ('layers', 1324), ('lines', 1320), ('pedals', 1300), ('purple', 1169), ('stem', 1133), ('pink', 1046), ('dots', 1044),
        self.attributes,self.filenames=self.load_102flowers()
    def load_102flowers(self):
        texts_path='./data/cvpr2016_flowers/text_c10/'
        attributes=[]
        filenames=[]
        cnt=0
        for class_path in os.listdir(texts_path):
            sub_path=texts_path+class_path
            if sub_path.endswith('.t7'):
                continue
            print(sub_path)
            for text_path in os.listdir(sub_path):
                if not text_path.endswith('.txt'):
                    continue
                if cnt%10==0:
                    print(cnt)
                cnt+=1
                path=sub_path+'/'+text_path
                if text_path.endswith('.txt'):
                    attribute,img_name=self.read_captions(path)
                    filenames.append(img_name)

                    while attribute.__len__() < 19:
                        attribute.append(0)
                    attributes.append(attribute)

        #attributes = np.loadtxt('../../flower_attributes.txt', dtype=int)

        np.savetxt('./flower_attributes.txt', attributes, fmt='%d', delimiter=' ')
        return attributes,filenames

    def __getitem__(self, index):
        if self.split=='train':
            img_path = self.filenames[index]
            attribute=self.attributes[index]
            img = Image.open(img_path).convert('RGB')
            img=np.array(img)

            img=Image.fromarray(img)
            img= self.transform(img)
            img=self.norm(img)
            img=np.array(img)
            img=torch.tensor(img,dtype=torch.float32)
            attribute=np.array(attribute)
            attribute=torch.tensor(attribute,dtype=torch.int32)

            return img,attribute
        else:
            attribute = self.attributes[index]
            attribute = np.array(attribute)
            attribute = torch.tensor(attribute, dtype=torch.int32)

            return attribute

    def __len__(self):
        if self.split=='train':
            return len(self.filenames)
        else:
            return len(self.attributes)
    def read_captions(self,caption_path):
        img_name='./102flowers/jpg/'+caption_path.split('/')[-1].replace('.txt','.jpg')
        #return None,img_name
        nlp = spacy.load("en_core_web_sm")
        f = open(caption_path, "r")  
        caption_line = f.readline()
        caption_line = caption_line.replace('\n', '')
        attribute=[]
        while caption_line: 
            if not caption_line:
                break

            doc = nlp(caption_line)
            # print(caption_line)
            for token in doc:
                if token.tag_ == 'JJ':
                    temp = token.text
                    while token.tag_ != 'NN' and token.tag_ != 'NNS':
                        token = token.head
                        if token == token.head:
                            break
                    if token.text in self.flower_parts.keys() and temp in self.adj.keys():
                        temp = self.flower_parts[token.text] * 14 + self.adj[temp] + 1
                        if temp not in attribute:
                            attribute.append(temp)

            caption_line = f.readline()
        return attribute,img_name


if __name__ == '__main__':
    bboxs=[]
    f = open("./data/birds/CUB_200_2011/bounding_boxes.txt", "r")  
    line = f.readline()
    while line:  
        line = line.replace('\n', '')  
        line = line.split(' ')[1:]
        bbox=line[0]+' '+line[1]+' '+line[2]+' '+line[3]
        bboxs.append(bbox)
        line = f.readline() 

    index = np.loadtxt('./data/birds/CUB_200_2011/train_test_split.txt', dtype=int)
    print(index.__len__())
    train_index,test_index=[],[]
    for i in range(index.__len__()):
        if index[i][1]==1:
            train_index.append(i)
        elif index[i][1]==0:
            test_index.append(i)
    train_index.sort()
    test_index.sort()
    bbox_train=[]
    bbox_test=[]
    for index in train_index:
        bbox_train.append(bboxs[index])
    for index in test_index:
        bbox_test.append(bboxs[index])
    with open('./bboxs_train.txt','w') as f:
        for i in range(bbox_train.__len__()):
            f.write(bbox_train[i]+'\n')
    with open('./bboxs_test.txt','w') as f:
        for i in range(bbox_test.__len__()):
            f.write(bbox_test[i]+'\n')
