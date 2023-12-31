import pickle

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
import matplotlib.pyplot as plt
import parser
from PIL import Image
from torch.autograd import Variable
import torchvision.utils
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import nltk
import nltk.stem as ns
import difflib
from nltk.corpus import brown
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('nps_chat')

import torch.utils.data as data


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

        #fd = nltk.FreqDist(brown.tagged_words(categories='news') + nltk.corpus.nps_chat.tagged_words())
        #words = [word for (word, num) in fd.keys()]
        #cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news') + nltk.corpus.nps_chat.tagged_words())
        #likely_tags = dict((word, cfd[word].max()) for word in words)
        #likely_tags['skinny'] = 'JJ'
        #likely_tags['back'] = 'JJ'
        #likely_tags['pointy'] = 'JJ'
        #self.btr = nltk.UnigramTagger(model=likely_tags)  # 查询标注器和正则标注器组合

        import pickle
        file = open('./captions_DAMSM.pickle', 'rb')
        data = pickle.load(file)
        self.ixtoword, self.wordtoix=data[2],data[3]
        self.filenames,self.all_captions,self.bboxs,self.attrs= self.load_bird_dataset(self.split)
        self.all_txt=self.all_captions
        #_,self.all_captions2,_,self.attrs2=self.load_bird_dataset('test' if self.split=='train' else 'train')
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
        if os.path.exists('./birds_data/train_attrs.npy') and os.path.exists('./birds_data/test_attrs.npy'):
            attrs_exists=True
            if self.split=='train':
                all_attrs=np.load('./birds_data/train_attrs.npy',allow_pickle=True)
            else:
                all_attrs=np.load('./birds_data/test_attrs.npy',allow_pickle=True)
        if split=='test':
            f = open("./birds_data/bird_images_test.txt", "r")  # 设置文件对象
        else:
            f = open("./birds_data/bird_images_train.txt", "r")  # 设置文件对象
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
            bbox_f = open("./birds_data/bboxs_test.txt", "r")  # 设置文件对象
        else:
            bbox_f = open("./birds_data/bboxs_train.txt", "r")  # 设置文件对象
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
                np.save('./birds_data/train_attrs.npy', all_attrs)
            else:
                np.save('./birds_data/test_attrs.npy', all_attrs)
        return filenames,all_captions,bboxs,all_attrs

    def build_dictionary(self, captions,  all_attrs):
        '''if os.path.exists('./birds_data/wordtoix.npy') and os.path.exists('./birds_data/ixtoword.npy'):
            wordtoix = np.load('./birds_data/wordtoix.npy', allow_pickle=True).item()
            ixtoword = np.load('./birds_data/ixtoword.npy', allow_pickle=True).item()
        else:
            word_counts = dict()
            captions = captions1 + captions2
            for sent in captions:
                for word in sent:
                    if word in word_counts.keys():
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
            word_counts=sorted(word_counts.items(), key=lambda d: d[1],reverse=True)
            word_counts = dict(word_counts)
            vocab = [w for w in word_counts.keys() if word_counts[w] > 0]
            ixtoword = {}
            ixtoword[0] = '<end>'
            wordtoix = {}
            wordtoix['<end>'] = 0
            ix = 1
            for w in vocab:
                wordtoix[w] = ix
                ixtoword[ix] = w
                ix += 1
            np.save('./birds_data/wordtoix.npy',wordtoix)
            np.save('./birds_data/ixtoword.npy', ixtoword)'''

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
                if new_attr.__len__()>5:#一个attr的单词数
                    ix = list(np.arange(new_attr.__len__()))  # 1, 2, 3,..., maxNum
                    np.random.shuffle(ix)
                    ix = ix[:5]
                    ix = np.sort(ix)
                    new_attr = np.array(new_attr)[ix]
                    new_attr=list(new_attr)
                while new_attr.__len__()<5:
                    new_attr.append(0)
                new_attrs.append(new_attr)
            if new_attrs.__len__()>3:#attr数量
                ix = list(np.arange(new_attrs.__len__()))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:3]
                ix = np.sort(ix)
                new_attrs = np.array(new_attrs)[ix]
                new_attrs=list(new_attrs)
            while new_attrs.__len__()<3:
                new_attrs.append([0,0,0,0,0])
            all_attrs_new.append(new_attrs)
        return captions_new, all_attrs_new
    def get_attrs(self,cap):
        while cap.count(', '):
            cap = cap.replace(', ', ' ')
        while cap.count(','):
            cap = cap.replace(',', ' ')
        while cap.count('.'):
            cap = cap.replace('.', '')
        while cap.count('that is'):
            cap = cap.replace('that is', 'is')
        while cap.count('that are'):
            cap = cap.replace('that are', 'are')
        while cap.count('which is'):
            cap = cap.replace('which is', 'is')
        while cap.count('which are'):
            cap = cap.replace('which are', 'are')
        while cap.count('that\'s'):
            cap = cap.replace('that\'s', 'is')
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())
        tokens_new=[]
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
        tokens=tokens_new
        sentence_tag = self.btr.tag(tokens)
        default_tag = nltk.pos_tag(tokens)
        for i in range(sentence_tag.__len__()):
            if sentence_tag[i][1] == None:
                sentence_tag[i] = default_tag[i]
        new_sentence_tag = []
        for i in range(sentence_tag.__len__()):
            if sentence_tag[i][1] == 'QL' or sentence_tag[i][1] == 'RB' or sentence_tag[i][1] == 'DTI':
                continue
            else:
                new_sentence_tag.append(sentence_tag[i])
        sentence_tag = new_sentence_tag
        grammar = r'''
                ADJS: {<CC><JJ|JJR|JJS|VBD|VBN|VBG>}
                ADJ: {<JJ|JJR|JJS|VBD|VBN|VBG>+<ADJS>*}
                VERB: {<VB|VBG|VBN|VBP|VBZ|BEZ|HVZ>}
                NOUNS: {<CC><NN|NNS>}
                NOUN: {<NN|NNS>+<NOUNS>*}
                Sentence_pattern1: {<AT|DT>?<ADJ>+<NOUN>+}
                Sentence_pattern2: {<AT|DT>?<NOUN>+<IN><DT>*<NN|NNS>+<VERB><ADJ>}
                Sentence_pattern3: {<AT|DT>?<NOUN>+<VERB><ADJ>}
                Sentence_pattern4: {<AT|DT>?<NOUN>+<VERB><RB><ADJS>}
                Sentence_pattern5: {<ADJS><NOUN>}
                Sentence_pattern6: {<AT|DT>?<NOUN>+<BER><ADJ>}
                '''
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(sentence_tag)

        attrs = []
        for i in range(len(tree)):
            if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                       'Sentence_pattern3', 'Sentence_pattern4',
                                                                       'Sentence_pattern5', 'Sentence_pattern6']:
                is_negative = False
                for subtree in tree[i].subtrees():
                    for leaf in subtree.leaves():
                        if leaf[0] in ['no', 'not']:
                            is_negative = True
                for j in range(i - 1, -1, -1):
                    if type(tree[j]) == nltk.tree.Tree and tree[j].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                               'Sentence_pattern3', 'Sentence_pattern4',
                                                                               'Sentence_pattern5',
                                                                               'Sentence_pattern6']:
                        break
                    if type(tree[j]) != nltk.tree.Tree:
                        if tree[j][0] in ['no', 'not']:
                            is_negative = True
                            break
                        continue
                    for leaf in tree[j].leaves():
                        if leaf[0] in ['no', 'not']:
                            is_negative = True
                if is_negative:
                    continue
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                           'Sentence_pattern3', 'Sentence_pattern6']:
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJ':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)

                if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern4':
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)

                if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern5':
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                            for leaf in subtree.leaves():
                                if subtree.index(leaf) == 0:
                                    continue
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
        return attrs

class Flower_Dataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize((256,256))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])

        '''fd = nltk.FreqDist(brown.tagged_words(categories='news') + nltk.corpus.nps_chat.tagged_words())
        words = [word for (word, num) in fd.keys()]
        cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news') + nltk.corpus.nps_chat.tagged_words())
        likely_tags = dict((word, cfd[word].max()) for word in words)
        self.btr = nltk.UnigramTagger(model=likely_tags)'''

        self.filenames, self.all_captions,self.wordtoix,self.ixtoword,self.attrs=self.load_flower_dataset()
        self.class_id=np.arange(self.filenames.__len__())

    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        img = self.norm(img)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)

        cap_idx = random.randint(5 * index, 5 * index + 4)
        cap = self.all_captions[cap_idx]

        attr = self.attrs[cap_idx]
        cap_len = (cap.__len__() - cap.count(0))
        attr = np.array(attr)
        if np.count_nonzero(attr[1]) == 0:
            attr[1] = attr[0]
        if np.count_nonzero(attr[2]) == 0:
            attr[2] = attr[0]
        attr = attr.astype(np.int32)
        attr = torch.tensor(attr, dtype=torch.int32)
        img = Variable(img)
        attr = Variable(attr)

        img = Variable(img)
        cap = np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]

        return img, attr, cap,cap_len,class_id
    def __len__(self):
        return len(self.filenames)

    def load_flower_dataset(self):
        word2idx,idx2word,caption2image=self.get_dictionary()
        #train_paths = './data/flowers/test.txt'
        #test_paths = './data/flowers/train.txt'
        #valid_paths = './data/flowers/valid.txt'
        all_paths='./data/flowers/all.txt'

        import pickle
        file = open('./data/flowers/filenames_train.pickle', 'rb')
        train_data = pickle.load(file)
        file = open('./data/flowers/filenames_test.pickle', 'rb')
        test_data = pickle.load(file)
        img_paths = []
        all_captions=[]
        all_attrs=[]

        attrs_exists = False
        if os.path.exists('./birds_data/flower_train_attrs.npy') and os.path.exists('./birds_data/flower_test_attrs.npy'):
            attrs_exists = True
            if self.split == 'train':
                all_attrs = np.load('./birds_data/flower_train_attrs.npy', allow_pickle=True)
            else:
                all_attrs = np.load('./birds_data/flower_test_attrs.npy', allow_pickle=True)

        with open(all_paths, "r") as f:
            img_path = f.readline().replace('\n', '').split(' ')[0]
            while img_path:
                if self.split=='train' and img_path.replace('.jpg','') in test_data:
                    img_path = f.readline().replace('\n', '').split(' ')[0]
                    continue

                if self.split=='test' and img_path.replace('.jpg','') in train_data:
                    img_path = f.readline().replace('\n', '').split(' ')[0]
                    continue
                img_paths.append('./data/flowers/' + img_path)
                captions_path=caption2image[img_path.split('/')[-1]]
                with open(captions_path, "r") as cap_f:
                    captions = cap_f.read().encode('utf-8').decode('utf8').split('\n')
                    cnt_captions = 0
                    for cap in captions:
                        if len(cap) == 0:
                            continue
                        while cap.count('.'):
                            cap = cap.replace('.', '')
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(cap.lower())
                        tokens_new = []
                        for t in tokens:
                            t = t.encode('ascii', 'ignore').decode('ascii')
                            if len(t) > 0:
                                tokens_new.append(t)
                        if tokens_new.__len__() > 20 or tokens_new.__len__()==0:
                            continue
                        if cnt_captions < 5:
                            all_captions.append(tokens_new)
                            cnt_captions+=1
                            if attrs_exists==False:
                                attrs = self.get_attrs(cap)
                                all_attrs.append(attrs)
                        else:
                            break
                img_path = f.readline().replace('\n', '').split(' ')[0]
        captions_new=[]
        for t in all_captions:
            rev = []
            for w in t:
                if w not in word2idx.keys():
                    print('word not in wordtoix')
                    continue
                rev.append(word2idx[w])
            while rev.__len__()<20:
                rev.append(0)
            captions_new.append(rev)

        if attrs_exists==False:
            if self.split == 'train':
                np.save('./birds_data/flower_train_attrs.npy', all_attrs)
            else:
                np.save('./birds_data/flower_test_attrs.npy', all_attrs)

        new_all_attrs=[]
        for attrs in all_attrs:
            new_attrs=[]
            for attr in attrs:
                new_attr = []
                for w in attr:
                    if w not in word2idx.keys():
                        print('word not in wordtoix')
                        continue
                    new_attr.append(word2idx[w])
                if new_attr.__len__()>5:
                    continue
                while new_attr.__len__()<5:
                    new_attr.append(0)
                new_attrs.append(new_attr)
            while new_attrs.__len__()<3:
                new_attrs.append([0,0,0,0,0])
            if new_attrs.__len__()>3:
                ix = list(np.arange(new_attrs.__len__()))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:3]
                ix = np.sort(ix)
                new_attrs = np.array(new_attrs)[ix]
                new_attrs = list(new_attrs)
            new_all_attrs.append(new_attrs)
        all_attrs=new_all_attrs
        return img_paths,captions_new,word2idx,idx2word,all_attrs

    def get_dictionary(self):
        captiontoimage={}
        texts_path = './data/flowers/text_c10/'
        all_captions=[]
        for class_path in os.listdir(texts_path):
            sub_path = texts_path + class_path
            if sub_path.endswith('.t7'):
                continue
            for text_path in os.listdir(sub_path):
                if not text_path.endswith('.txt'):
                    continue
                captiontoimage[text_path.replace('.txt','.jpg')]=sub_path + '/' + text_path
                path = sub_path + '/' + text_path
                if text_path.endswith('.txt'):
                    with open(path) as f:
                        caption=f.readline()
                        while caption:
                            caption=caption.replace('\n','').replace('.','')
                            tokenizer = RegexpTokenizer(r'\w+')
                            tokens = tokenizer.tokenize(caption.lower())
                            tokens_new = []
                            for t in tokens:
                                t = t.encode('ascii', 'ignore').decode('ascii')
                                if len(t) > 0:
                                    tokens_new.append(t)
                            all_captions.append(tokens_new)
                            caption=f.readline()
        word_counts = dict()
        for sent in all_captions:
            for word in sent:
                if word in word_counts.keys():
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        word_counts = sorted(word_counts.items(), key=lambda d: d[1], reverse=True)
        word_counts = dict(word_counts)
        vocab = [w for w in word_counts.keys() if word_counts[w] > 0]
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        if os.path.exists('./birds_data/flower_wordtoix.npy'):
            print('wordtoix exists')
            wordtoix = np.load('./birds_data/flower_wordtoix.npy', allow_pickle=True).item()
        if os.path.exists('./birds_data/flower_ixtoword.npy'):
            print('ixtoword exists')
            ixtoword = np.load('./birds_data/flower_ixtoword.npy', allow_pickle=True).item()
        return wordtoix,ixtoword,captiontoimage

    def get_attrs(self,cap):
        while cap.count(', '):
            cap = cap.replace(', ', ' ')
        while cap.count(','):
            cap = cap.replace(',', ' ')
        while cap.count('.'):
            cap = cap.replace('.', '')
        while cap.count('that\'s'):
            cap = cap.replace('that\'s', 'that is')
        while cap.count('that is'):
            cap = cap.replace('that is', 'is')
        while cap.count('that are'):
            cap = cap.replace('that are', 'are')
        while cap.count('which is'):
            cap = cap.replace('which is', 'is')
        while cap.count('which are'):
            cap = cap.replace('which are', 'are')
        while cap.count('as well as'):
            cap = cap.replace('as well as', 'and')
        while cap.count('don\'t'):
            cap = cap.replace('don\'t', 'do not')
        while cap.count('doesn\'t'):
            cap = cap.replace('doesn\'t', 'does not')
        while cap.count('and are'):
            cap = cap.replace('and are', 'and')
        while cap.count('and is'):
            cap = cap.replace('and is', 'and')
        tokens = word_tokenize(cap.lower())
        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
        tokens = tokens_new
        sentence_tag = self.btr.tag(tokens)
        default_tag = nltk.pos_tag(tokens)
        for i in range(sentence_tag.__len__()):
            if sentence_tag[i][1] == None:
                sentence_tag[i] = default_tag[i]
        new_sentence_tag = []
        for i in range(sentence_tag.__len__()):
            if sentence_tag[i][1] == 'QL' or sentence_tag[i][1] == 'RB' or sentence_tag[i][1] == 'DTI':
                continue
            else:
                new_sentence_tag.append(sentence_tag[i])
        sentence_tag = new_sentence_tag
        grammar = r'''
                    ADJS: {<CC><JJ|JJR|JJS|VBD|VBN|VBG>}
                    ADJ: {<JJ|JJR|JJS|VBD|VBN|VBG>+<ADJS>*}
                    VERB: {<VB|VBG|VBN|VBP|VBZ|BEZ|HVZ>}
                    NOUNS: {<CC><NN|NNS>}
                    NOUN: {<NN|NNS>+<NOUNS>*}
                    Sentence_pattern1: {<AT|DT>?<ADJ>+<NOUN>+}
                    Sentence_pattern2: {<AT|DT>?<NOUN>+<IN><DT>*<NN|NNS>+<VERB><ADJ>}
                    Sentence_pattern3: {<AT|DT>?<NOUN>+<VERB><ADJ>}
                    Sentence_pattern4: {<AT|DT>?<NOUN>+<VERB><RB><ADJS>}
                    Sentence_pattern5: {<ADJS><NOUN>}
                    Sentence_pattern6: {<AT|DT>?<NOUN>+<BER><ADJ>}
                    '''
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(sentence_tag)
        attrs = []
        for i in range(len(tree)):
            if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                       'Sentence_pattern3', 'Sentence_pattern4',
                                                                       'Sentence_pattern5', 'Sentence_pattern6']:
                is_negative = False
                for subtree in tree[i].subtrees():
                    for leaf in subtree.leaves():
                        if leaf[0] in ['no', 'not']:
                            is_negative = True
                for j in range(i - 1, -1, -1):
                    if type(tree[j]) == nltk.tree.Tree and tree[j].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                               'Sentence_pattern3', 'Sentence_pattern4',
                                                                               'Sentence_pattern5',
                                                                               'Sentence_pattern6']:
                        break
                    if type(tree[j]) != nltk.tree.Tree:
                        if tree[j][0] in ['no', 'not']:
                            is_negative = True
                            break
                        continue
                    for leaf in tree[j].leaves():
                        if leaf[0] in ['no', 'not']:
                            is_negative = True
                if is_negative:
                    continue
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                           'Sentence_pattern3', 'Sentence_pattern6']:
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJ':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern4':
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern5':
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                            for leaf in subtree.leaves():
                                if subtree.index(leaf) == 0:
                                    continue
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
        return attrs

class Celeba_Dataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize((256,256))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])

        '''fd = nltk.FreqDist(brown.tagged_words(categories='news') + nltk.corpus.nps_chat.tagged_words())
        words = [word for (word, num) in fd.keys()]
        cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news') + nltk.corpus.nps_chat.tagged_words())
        likely_tags = dict((word, cfd[word].max()) for word in words)
        likely_tags['thirties'] = 'AGE'
        likely_tags['forties'] = 'AGE'
        likely_tags['fifties'] = 'AGE'
        likely_tags['sixties'] = 'AGE'
        likely_tags['seventies'] = 'AGE'
        likely_tags['eighties'] = 'AGE'
        likely_tags['ninties'] = 'AGE'
        likely_tags['no'] = 'NEG'
        likely_tags['bit'] = 'JJ'
        self.btr = nltk.UnigramTagger(model=likely_tags)  # 查询标注器和正则标注器组合'''

        self.filenames, self.all_captions,self.wordtoix,self.ixtoword,self.attrs=self.load_celeba_dataset()
        self.class_id=np.arange(self.filenames.__len__())

    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        img = self.norm(img)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)

        cap = self.all_captions[index]

        attr = self.attrs[index]
        cap_len = (cap.__len__() - cap.count(0))

        attr = np.array(attr)
        if np.count_nonzero(attr[1]) == 0:
            attr[1] = attr[0]
        if np.count_nonzero(attr[2]) == 0:
            attr[2] = attr[0]

        attr = attr.astype(np.int32)
        attr = torch.tensor(attr, dtype=torch.int32)
        img = Variable(img)
        attr = Variable(attr)

        img = Variable(img)
        cap = np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)
        class_id = self.class_id[index]
        return img, attr,cap,cap_len,class_id
    def __len__(self):
        return len(self.filenames)

    def load_celeba_dataset(self):
        word2idx,idx2word=self.get_dictionary()
        img_paths = []
        all_captions=[]
        all_attrs=[]

        attrs_exists=False
        if os.path.exists('./birds_data/celeba_train_attrs.npy') and os.path.exists('./birds_data/celeba_test_attrs.npy'):
            attrs_exists=True
            if self.split=='train':
                all_attrs=np.load('./birds_data/celeba_train_attrs.npy',allow_pickle=True)
            else:
                all_attrs=np.load('./birds_data/celeba_test_attrs.npy',allow_pickle=True)
        if self.split=='train':
            paths,_,_=self.get_split()
        elif self.split=='test':
            _,paths,_=self.get_split()
        texts_path = './data/celeba/captions_hq.json'
        with open(texts_path, 'r') as f:
            captions = json.load(f)
        for path in paths:
            if path not in captions.keys():
                print(path)
                continue
            img_paths.append('./data/celeba/image/'+path)
            caption = captions[path]['overall_caption'].replace('\n','')
            if len(caption) == 0:
                continue
            while caption.count('.'):
                caption = caption.replace('.', '')
            if attrs_exists==False:
                attrs = self.get_attrs(caption)
                all_attrs.append(attrs)
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(caption.lower())
            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            all_captions.append(tokens_new)
        captions_new=[]
        for t in all_captions:
            rev = []
            for w in t:
                if w not in word2idx.keys():
                    print('word not in wordtoix')
                    continue
                rev.append(word2idx[w])
            while rev.__len__()<45:
                rev.append(0)
            captions_new.append(rev)

        if attrs_exists==False:
            if self.split == 'train':
                np.save('./birds_data/celeba_train_attrs.npy', all_attrs)
            else:
                np.save('./birds_data/celeba_test_attrs.npy', all_attrs)

        new_all_attrs=[]
        for attrs in all_attrs:
            new_attrs=[]
            for attr in attrs:
                new_attr = []
                for w in attr:
                    if w not in word2idx.keys():
                        print('word not in wordtoix')
                        continue
                    new_attr.append(word2idx[w])
                if new_attr.__len__()>5:
                    continue
                while new_attr.__len__()<5:
                    new_attr.append(0)
                new_attrs.append(new_attr)
            while new_attrs.__len__()<3:
                new_attrs.append([0,0,0,0,0])
            if new_attrs.__len__()>3:
                ix = list(np.arange(new_attrs.__len__()))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:3]
                ix = np.sort(ix)
                new_attrs = np.array(new_attrs)[ix]
                new_attrs = list(new_attrs)
            new_all_attrs.append(new_attrs)
        all_attrs=new_all_attrs
        return img_paths,captions_new,word2idx,idx2word,all_attrs

    def get_dictionary(self):
        if os.path.exists('./birds_data/celeba_wordtoix.npy') and os.path.exists('./birds_data/celeba_ixtoword.npy'):
            wordtoix = np.load('./birds_data/celeba_wordtoix.npy', allow_pickle=True).item()
            ixtoword = np.load('./birds_data/celeba_ixtoword.npy', allow_pickle=True).item()
            return wordtoix, ixtoword
        all_captions=[]
        texts_path = './data/celeba/captions_hq.json'
        with open(texts_path, 'r') as f:
            captions = json.load(f)
        for key in captions.keys():
            caption=captions[key]['overall_caption']
            caption = caption.replace('\n', '').replace('.', '')
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(caption.lower())
            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
                    all_captions.append(tokens_new)

        word_counts = dict()
        for sent in all_captions:
            for word in sent:
                if word in word_counts.keys():
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        word_counts = sorted(word_counts.items(), key=lambda d: d[1], reverse=True)
        word_counts = dict(word_counts)
        vocab = [w for w in word_counts.keys() if word_counts[w] > 0]
        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        return wordtoix,ixtoword

    def get_split(self):
        import pandas as pd
        image_list = pd.read_csv('./data/celeba/CelebA-HQ-to-CelebA-mapping.txt', delim_whitespace=True, header=None)
        train_paths=[]
        test_paths=[]
        val_paths=[]

        for idx, x in enumerate(image_list.loc[:, 1]):
            if x >= 162771 and x < 182638:
                val_paths.append(str(idx) + '.jpg')
            elif x >= 182638:
                test_paths.append(str(idx) + '.jpg')
            else:
                train_paths.append(str(idx) + '.jpg')
        print(train_paths.__len__(),test_paths.__len__(),val_paths.__len__())
        return train_paths,test_paths,val_paths

    def get_attrs(self,cap):
        while cap.count(', '):
            cap = cap.replace(', ', ' ')
        while cap.count(','):
            cap = cap.replace(',', ' ')
        while cap.count('.'):
            cap = cap.replace('.', '')
        while cap.count('that\'s'):
            cap = cap.replace('that\'s', 'that is')
        while cap.count('that is'):
            cap = cap.replace('that is', 'is')
        while cap.count('that are'):
            cap = cap.replace('that are', 'are')
        while cap.count('which is'):
            cap = cap.replace('which is', 'is')
        while cap.count('which are'):
            cap = cap.replace('which are', 'are')
        while cap.count('as well as'):
            cap = cap.replace('as well as', 'and')
        while cap.count('don\'t have'):
            cap = cap.replace('don\'t have', 'have no')
        while cap.count('doesn\'t have'):
            cap = cap.replace('doesn\'t have', 'have no')
        while cap.count('don\'t'):
            cap = cap.replace('don\'t', 'do not')
        while cap.count('doesn\'t'):
            cap = cap.replace('doesn\'t', 'does not')
        while cap.count('and are'):
            cap = cap.replace('and are', 'and')
        while cap.count('and is'):
            cap = cap.replace('and is', 'and')
        tokens = word_tokenize(cap.lower())
        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
        tokens = tokens_new
        sentence_tag = self.btr.tag(tokens)
        default_tag = nltk.pos_tag(tokens)
        for i in range(sentence_tag.__len__()):
            if sentence_tag[i][1] == None:
                sentence_tag[i] = default_tag[i]
        new_sentence_tag = []
        for i in range(sentence_tag.__len__()):
            if sentence_tag[i][1] == 'QL' or sentence_tag[i][1] == 'RB' or sentence_tag[i][1] == 'DTI' or \
                    sentence_tag[i][1] == 'DT':
                continue
            else:
                new_sentence_tag.append(sentence_tag[i])
        sentence_tag = new_sentence_tag
        grammar = r'''
                            ADJS: {<CC><JJ|JJR|JJS|VBD|VBN|VBG>}
                            ADJ: {<JJ|JJR|JJS|VBD|VBN|VBG>+<ADJS>*}
                            VERB: {<VB|VBG|VBN|VBP|VBZ|BEZ|HVZ>}
                            NOUNS: {<CC><NN|NNS>}
                            NOUN: {<NN|NNS>+<NOUNS>*}
                            NEGATIVE: {<NEG><ADJ>*<NOUN>+}
                            Sentence_pattern1: {<AT|DT>?<ADJ>+<NOUN>+}
                            Sentence_pattern2: {<AT|DT>?<NOUN>+<IN><DT>*<NN|NNS>+<VERB><ADJ>}
                            Sentence_pattern3: {<AT|DT>?<NOUN>+<VERB><ADJ>}
                            Sentence_pattern4: {<AT|DT>?<NOUN>+<VERB><RB><ADJS>}
                            Sentence_pattern5: {<ADJS><NOUN>}
                            Sentence_pattern6: {<AT|DT>?<NOUN>+<BER><ADJ>}
                            '''
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(sentence_tag)
        attrs = []
        for i in range(len(tree)):
            if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2',
                                                                       'Sentence_pattern3', 'Sentence_pattern6']:
                attr = []
                for subtree in tree[i].subtrees():
                    if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJ':
                        for leaf in subtree.leaves():
                            attr.append(leaf[0])
                for subtree in tree[i].subtrees():
                    if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                        for leaf in subtree.leaves():
                            attr.append(leaf[0])
                attrs.append(attr)
            if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern4':
                attr = []
                for subtree in tree[i].subtrees():
                    if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                        for leaf in subtree.leaves():
                            attr.append(leaf[0])
                for subtree in tree[i].subtrees():
                    if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                        for leaf in subtree.leaves():
                            attr.append(leaf[0])
                attrs.append(attr)
            if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern5':
                attr = []
                for subtree in tree[i].subtrees():
                    if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                        for leaf in subtree.leaves():
                            if subtree.index(leaf) == 0:
                                continue
                            attr.append(leaf[0])
                for subtree in tree[i].subtrees():
                    if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                        for leaf in subtree.leaves():
                            attr.append(leaf[0])
                attrs.append(attr)
            if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'NEGATIVE':
                attr = []
                for leaf in tree[i].leaves():
                    attr.append(leaf[0])
                attrs.append(attr)
            if type(tree[i]) != nltk.tree.Tree and tree[i][1] == 'AGE':
                attr = []
                attr.append(tree[i][0])
                attrs.append(attr)
        return attrs


def get_attributes(captions,split):
    all_attr = []
    fd = nltk.FreqDist(brown.tagged_words(categories='news')+nltk.corpus.nps_chat.tagged_words())
    words = [word for (word, num) in fd.keys()]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news')+nltk.corpus.nps_chat.tagged_words())
    likely_tags = dict((word, cfd[word].max()) for word in words)

    if split=='bird':
        likely_tags['skinny']='JJ'
        likely_tags['back']='JJ'
        likely_tags['pointy']='JJ'
        likely_tags['features']='NN'
        btr = nltk.UnigramTagger(model=likely_tags)  # 查询标注器和正则标注器组合

        for cap in captions:
            while cap.count(', '):
                cap = cap.replace(', ', ' ')
            while cap.count(','):
                cap = cap.replace(',', ' ')
            while cap.count('.'):
                cap=cap.replace('.','')
            while cap.count('that is'):
                cap=cap.replace('that is','is')
            while cap.count('that are'):
                cap=cap.replace('that are','are')
            while cap.count('which is'):
                cap=cap.replace('which is','is')
            while cap.count('which are'):
                cap=cap.replace('which are','are')
            while cap.count('that\'s'):
                cap=cap.replace('that\'s','is')
            while cap.count('as well as'):
                cap=cap.replace('as well as','and')
            while cap.count('don\'t'):
                cap=cap.replace('don\'t','do not')
            while cap.count('doesn\'t'):
                cap=cap.replace('doesn\'t','does not')
            #tokenizer = RegexpTokenizer(r'\w+')
            #tokens = tokenizer.tokenize(cap.lower())
            tokens = word_tokenize(cap.lower())
            tokens_new=[]
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            tokens=tokens_new
            sentence_tag=btr.tag(tokens)
            default_tag = nltk.pos_tag(tokens)
            for i in range(sentence_tag.__len__()):
                if sentence_tag[i][1]==None:
                    sentence_tag[i]=default_tag[i]
            new_sentence_tag=[]
            for i in range(sentence_tag.__len__()):
                #QL: very completely
                #RB: occasionally unabatingly maddeningly also
                #DTI: some
                if sentence_tag[i][1]=='QL' or sentence_tag[i][1]=='RB' or sentence_tag[i][1]=='DTI' or sentence_tag[i][1]=='AP':
                    continue
                else:
                    new_sentence_tag.append(sentence_tag[i])
            sentence_tag=new_sentence_tag
            #Sentence_pattern1: a blue and white belly
            #Sentence_pattern2: the belly of the bird is blue
            #Sentence_pattern3: the bird has blue/the belly is blue
            #Sentence_pattern4: the belly is as blue as
            #Sentence_pattern5: the bird has ... and yellow black belly
            #Sentence_pattern6: the bird is white
            #NP: {<Sentence_pattern1>|<Sentence_pattern2>|<Sentence_pattern3>|<Sentence_pattern4>}
            grammar=r'''
                ADJS: {<CC><JJ|JJR|JJS|VBD|VBN|VBG>}
                ADJ: {<JJ|JJR|JJS|VBD|VBN|VBG>+<ADJS>*}
                VERB: {<VB|VBG|VBN|VBP|VBZ|BEZ|HVZ>}
                NOUNS: {<CC><NN|NNS>}
                NOUN: {<NN|NNS>+<NOUNS>*}
                Sentence_pattern1: {<AT|DT>?<ADJ>+<NOUN>+}
                Sentence_pattern2: {<AT|DT>?<NOUN>+<IN><DT>*<NN|NNS>+<VERB><ADJ>}
                Sentence_pattern3: {<AT|DT>?<NOUN>+<VERB><ADJ>}
                Sentence_pattern4: {<AT|DT>?<NOUN>+<VERB><RB><ADJS>}
                Sentence_pattern5: {<ADJS><NOUN>}
                Sentence_pattern6: {<AT|DT>?<NOUN>+<BER><ADJ>}
                '''
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(sentence_tag)
            #tree.draw()

            attrs=[]
            for i in range(len(tree)):
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1','Sentence_pattern2','Sentence_pattern3','Sentence_pattern4','Sentence_pattern5','Sentence_pattern6']:
                    is_negative=False
                    for subtree in tree[i].subtrees():
                        for leaf in subtree.leaves():
                            if leaf[0] in ['no','not']:
                                is_negative=True
                    for j in range(i-1,-1,-1):
                        if type(tree[j]) == nltk.tree.Tree and tree[j].label() in ['Sentence_pattern1','Sentence_pattern2','Sentence_pattern3','Sentence_pattern4','Sentence_pattern5','Sentence_pattern6']:
                            break
                        if type(tree[j])!= nltk.tree.Tree:
                            if tree[j][0] in ['no','not']:
                                is_negative=True
                                break
                            continue
                        for leaf in tree[j].leaves():
                            if leaf[0] in ['no','not']:
                                is_negative=True
                    if is_negative:
                        continue
                    if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1','Sentence_pattern2','Sentence_pattern3','Sentence_pattern6']:
                        attr=[]
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label()=='ADJ':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label()=='NOUN':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        attrs.append(attr)

                    if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern4' :
                        attr = []
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label()=='ADJS':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label()=='NOUN':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        attrs.append(attr)

                    if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern5':
                        attr = []
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                                for leaf in subtree.leaves():
                                    if subtree.index(leaf)==0:
                                        continue
                                    attr.append(leaf[0])
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        attrs.append(attr)

            all_attr.append(attrs)
    elif split=='flower':
        btr = nltk.UnigramTagger(model=likely_tags)  # 查询标注器和正则标注器组合
        for cap in captions:
            while cap.count(', '):
                cap = cap.replace(', ', ' ')
            while cap.count(','):
                cap = cap.replace(',', ' ')
            while cap.count('.'):
                cap = cap.replace('.', '')
            while cap.count('that\'s'):
                cap = cap.replace('that\'s', 'that is')
            while cap.count('that is'):
                cap = cap.replace('that is', 'is')
            while cap.count('that are'):
                cap = cap.replace('that are', 'are')
            while cap.count('which is'):
                cap = cap.replace('which is', 'is')
            while cap.count('which are'):
                cap = cap.replace('which are', 'are')
            while cap.count('as well as'):
                cap = cap.replace('as well as', 'and')
            while cap.count('don\'t'):
                cap = cap.replace('don\'t', 'do not')
            while cap.count('doesn\'t'):
                cap = cap.replace('doesn\'t', 'does not')
            while cap.count('and are'):
                cap = cap.replace('and are', 'and')
            while cap.count('and is'):
                cap = cap.replace('and is', 'and')
            tokens = word_tokenize(cap.lower())
            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            tokens = tokens_new
            sentence_tag = btr.tag(tokens)
            default_tag = nltk.pos_tag(tokens)
            for i in range(sentence_tag.__len__()):
                if sentence_tag[i][1] == None:
                    sentence_tag[i] = default_tag[i]
            new_sentence_tag = []
            for i in range(sentence_tag.__len__()):
                if sentence_tag[i][1] == 'QL' or sentence_tag[i][1] == 'RB' or sentence_tag[i][1] == 'DTI':
                    continue
                else:
                    new_sentence_tag.append(sentence_tag[i])
            sentence_tag = new_sentence_tag
            grammar = r'''
                        ADJS: {<CC><JJ|JJR|JJS|VBD|VBN|VBG>}
                        ADJ: {<JJ|JJR|JJS|VBD|VBN|VBG>+<ADJS>*}
                        VERB: {<VB|VBG|VBN|VBP|VBZ|BEZ|HVZ>}
                        NOUNS: {<CC><NN|NNS>}
                        NOUN: {<NN|NNS>+<NOUNS>*}
                        Sentence_pattern1: {<AT|DT>?<ADJ>+<NOUN>+}
                        Sentence_pattern2: {<AT|DT>?<NOUN>+<IN><DT>*<NN|NNS>+<VERB><ADJ>}
                        Sentence_pattern3: {<AT|DT>?<NOUN>+<VERB><ADJ>}
                        Sentence_pattern4: {<AT|DT>?<NOUN>+<VERB><RB><ADJS>}
                        Sentence_pattern5: {<ADJS><NOUN>}
                        Sentence_pattern6: {<AT|DT>?<NOUN>+<BER><ADJ>}
                        '''
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(sentence_tag)
            attrs = []
            for i in range(len(tree)):
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2','Sentence_pattern3', 'Sentence_pattern4','Sentence_pattern5', 'Sentence_pattern6']:
                    is_negative = False
                    for subtree in tree[i].subtrees():
                        for leaf in subtree.leaves():
                            if leaf[0] in ['no', 'not']:
                                is_negative = True
                    for j in range(i - 1, -1, -1):
                        if type(tree[j]) == nltk.tree.Tree and tree[j].label() in ['Sentence_pattern1','Sentence_pattern2','Sentence_pattern3','Sentence_pattern4','Sentence_pattern5','Sentence_pattern6']:
                            break
                        if type(tree[j]) != nltk.tree.Tree:
                            if tree[j][0] in ['no', 'not']:
                                is_negative = True
                                break
                            continue
                        for leaf in tree[j].leaves():
                            if leaf[0] in ['no', 'not']:
                                is_negative = True
                    if is_negative:
                        continue
                    if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2','Sentence_pattern3','Sentence_pattern6']:
                        attr = []
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJ':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        attrs.append(attr)
                    if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern4':
                        attr = []
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        attrs.append(attr)
                    if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern5':
                        attr = []
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                                for leaf in subtree.leaves():
                                    if subtree.index(leaf) == 0:
                                        continue
                                    attr.append(leaf[0])
                        for subtree in tree[i].subtrees():
                            if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                                for leaf in subtree.leaves():
                                    attr.append(leaf[0])
                        attrs.append(attr)
            all_attr.append(attrs)
    elif split=='celeba':
        #likely_tags['teen']='AGE'
        #likely_tags['teenager']='AGE'
        likely_tags['thirties']='AGE'
        likely_tags['forties']='AGE'
        likely_tags['fifties']='AGE'
        likely_tags['sixties']='AGE'
        likely_tags['seventies']='AGE'
        likely_tags['eighties']='AGE'
        likely_tags['ninties']='AGE'
        likely_tags['no']='NEG'
        likely_tags['bit']='JJ'
        btr = nltk.UnigramTagger(model=likely_tags)  # 查询标注器和正则标注器组合
        for cap in captions:
            while cap.count(', '):
                cap = cap.replace(', ', ' ')
            while cap.count(','):
                cap = cap.replace(',', ' ')
            while cap.count('.'):
                cap = cap.replace('.', '')
            while cap.count('that\'s'):
                cap = cap.replace('that\'s', 'that is')
            while cap.count('that is'):
                cap = cap.replace('that is', 'is')
            while cap.count('that are'):
                cap = cap.replace('that are', 'are')
            while cap.count('which is'):
                cap = cap.replace('which is', 'is')
            while cap.count('which are'):
                cap = cap.replace('which are', 'are')
            while cap.count('as well as'):
                cap = cap.replace('as well as', 'and')
            while cap.count('don\'t have'):
                cap = cap.replace('don\'t have', 'have no')
            while cap.count('doesn\'t have'):
                cap = cap.replace('doesn\'t have', 'have no')
            while cap.count('don\'t'):
                cap = cap.replace('don\'t', 'do not')
            while cap.count('doesn\'t'):
                cap = cap.replace('doesn\'t', 'does not')
            while cap.count('and are'):
                cap = cap.replace('and are', 'and')
            while cap.count('and is'):
                cap = cap.replace('and is', 'and')
            tokens = word_tokenize(cap.lower())
            tokens_new = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0:
                    tokens_new.append(t)
            tokens = tokens_new
            sentence_tag = btr.tag(tokens)
            default_tag = nltk.pos_tag(tokens)
            for i in range(sentence_tag.__len__()):
                if sentence_tag[i][1] == None:
                    sentence_tag[i] = default_tag[i]
            new_sentence_tag = []
            for i in range(sentence_tag.__len__()):
                if sentence_tag[i][1] == 'QL' or sentence_tag[i][1] == 'RB' or sentence_tag[i][1] == 'DTI' or sentence_tag[i][1] == 'DT':
                    continue
                else:
                    new_sentence_tag.append(sentence_tag[i])
            sentence_tag = new_sentence_tag
            grammar = r'''
                                ADJS: {<CC><JJ|JJR|JJS|VBD|VBN|VBG>}
                                ADJ: {<JJ|JJR|JJS|VBD|VBN|VBG>+<ADJS>*}
                                VERB: {<VB|VBG|VBN|VBP|VBZ|BEZ|HVZ>}
                                NOUNS: {<CC><NN|NNS>}
                                NOUN: {<NN|NNS>+<NOUNS>*}
                                NEGATIVE: {<NEG><ADJ>*<NOUN>+}
                                Sentence_pattern1: {<AT|DT>?<ADJ>+<NOUN>+}
                                Sentence_pattern2: {<AT|DT>?<NOUN>+<IN><DT>*<NN|NNS>+<VERB><ADJ>}
                                Sentence_pattern3: {<AT|DT>?<NOUN>+<VERB><ADJ>}
                                Sentence_pattern4: {<AT|DT>?<NOUN>+<VERB><RB><ADJS>}
                                Sentence_pattern5: {<ADJS><NOUN>}
                                Sentence_pattern6: {<AT|DT>?<NOUN>+<BER><ADJ>}
                                '''
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(sentence_tag)
            attrs = []
            for i in range(len(tree)):
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() in ['Sentence_pattern1', 'Sentence_pattern2','Sentence_pattern3','Sentence_pattern6']:
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJ':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern4':
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
                if type(tree[i]) == nltk.tree.Tree and tree[i].label() == 'Sentence_pattern5':
                    attr = []
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'ADJS':
                            for leaf in subtree.leaves():
                                if subtree.index(leaf) == 0:
                                    continue
                                attr.append(leaf[0])
                    for subtree in tree[i].subtrees():
                        if type(subtree) == nltk.tree.Tree and subtree.label() == 'NOUN':
                            for leaf in subtree.leaves():
                                attr.append(leaf[0])
                    attrs.append(attr)
                if type(tree[i]) == nltk.tree.Tree and tree[i].label()=='NEGATIVE':
                    attr = []
                    for leaf in tree[i].leaves():
                        attr.append(leaf[0])
                    attrs.append(attr)
                if type(tree[i]) != nltk.tree.Tree and tree[i][1]=='AGE':
                    attr = []
                    attr.append(tree[i][0])
                    attrs.append(attr)
            all_attr.append(attrs)


    return all_attr

def DEA_GAN_get_attributes(captions):
    #https://github.com/hiarsal/DAE-GAN/blob/main/code/datasets.py
    all_captions = []
    all_attr = []
    for cap in captions:
        if len(cap) == 0:
            continue
        cap = cap.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())
        sentence_tag = nltk.pos_tag(tokens)
        # CUB
        grammar = "NP: {<DT>*<JJ>*<CC|IN>*<JJ>+<NN|NNS>+|<DT>*<NN|NNS>+<VBZ>+<JJ>+<IN|CC>*<JJ>*}"
        # COCO
        #  grammar = "NP: {<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+|<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+<IN>+<NN|NNS>+|<VB|VBD|VBG|VBN|VBP|VBZ>+<CD|DT>*<JJ|PRP$>*<NN|NNS>+|<IN>+<DT|CD|JJ|PRP$>*<NN|NNS>+<IN>*<CD|DT>*<JJ|PRP$>*<NN|NNS>*}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(sentence_tag)

        if len(tokens) == 0:
            print('cap', cap)
            return []

        attr_list = []
        for i in range(len(tree)):
            if type(tree[i]) == nltk.tree.Tree:
                attr = []
                for j in range(len(tree[i])):
                    attr.append(tree[i][j][0])
                attr_list.append(attr)
        # attribute end
        all_attr.append(attr_list)
        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)
        all_captions.append(tokens_new)  # [cnt, num_words]
    return all_captions, all_attr




def select_dataset(dataset):#select captions of the bird randomly to test the accuracy of the attributes extraction
    all_captions=[]
    if dataset=='CUB':
        txt_dir='./data/birds/text/'
        for sub_path in os .listdir(txt_dir):
            for txt_path in os.listdir(txt_dir+sub_path+'/'):
                with open(txt_dir+sub_path+'/'+txt_path,'r') as f:
                    line=f.readline()
                    while line:
                        line=line.replace('\n','')
                        line = line.replace('��', ' ')
                        line = line.replace('.', '')
                        all_captions.append(line)
                        line=f.readline()

    elif dataset=='flower':
        txt_dir='./data/cvpr2016_flowers/text_c10/'
        for sub_dir in os.listdir(txt_dir):
            if sub_dir.endswith('.t7'):
                continue
            for txt_path in os.listdir(txt_dir+sub_dir+'/'):
                if not txt_path.endswith('.txt'):
                    continue
                with open(txt_dir+sub_dir+'/'+txt_path) as f:
                    line=f.readline()
                    while line:
                        line=line.replace('\n','')
                        line = line.replace('.', '')
                        all_captions.append(line)
                        line=f.readline()
    elif dataset=='Celeba':
        import json
        with open("./data/captions_hq.json", 'r') as f:
            load_dict = json.load(f)
            for key in load_dict.keys():
                data=load_dict[key]
                caption=data['overall_caption']
                tokens=caption.split(' ')
                if tokens.__len__()>25:
                    continue
                all_captions.append(caption)

    idx = np.random.randint(0, len(all_captions), 900)
    all_captions = np.array(all_captions)
    all_captions = all_captions[idx]
    with open('./birds_data/Celeba_selected_captions.txt', 'w+') as f:
        for i in range(len(all_captions)):
            f.write(str(i) + ' ' + all_captions[i] + '\n')
    with open('./birds_data/Celeba_selected_captions_attributes.txt', 'w+') as f:
        attrs_arr = get_attributes(all_captions)
        for i in range(attrs_arr.__len__()):
            attrs=attrs_arr[i]
            attrline=''
            for attr in attrs:
                attrline=attrline+' '.join(attr)+';'

            f.write(str(i) + ' ' + attrline + '\n')
def calc_similarity(attrs,attrs_pre):
    if attrs.__len__()!=attrs_pre.__len__():
        print('the lengths are inconsistent.')
        return None
    cnt= attrs.__len__()
    sim=[]
    for i in range(cnt):
        for attr1 in attrs[i]:
            while attr1.count(','):
                attr1=attr1.replace(',','')
            attr1=attr1.split(' ')
            mx=0
            for attr2 in attrs_pre[i]:
                mx=max(mx,difflib.SequenceMatcher(None,attr1,attr2).ratio())
            sim.append(mx)
    return np.mean(sim)


class Bird_Dataset_DF_GAN(data.Dataset):
    def __init__(self,split):
        self.filenames=[]
        self.all_captions=[]
        self.bboxs=[]
        self.attrs=[]
        self.all_txt=[]
        import pickle
        if split=='train':
            load_file = open("filenames_train.pickle", "rb")
        elif split=='test':
            load_file = open("filenames_test.pickle", "rb")
        data = pickle.load(load_file)
        train_dataset = Bird_Dataset('train')
        test_dataset = Bird_Dataset('test')

        import pickle
        file = open('./captions_DAMSM.pickle', 'rb')
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
            filename = filename.replace('./data/birds/CUB_200_2011/images/', '')
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

        ix = list(np.arange(attr.__len__()))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:3]
        ix = np.sort(ix)
        attr = np.array(attr)[ix]

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

class Bird_Dataset_DAE_ATTRS(data.Dataset):
    def __init__(self,split):

        if split == 'train':
            npy = np.load('./birds_data/bird_dataset_DAE_train.npy', allow_pickle=True)
        else:
            npy = np.load('./birds_data/bird_dataset_DAE_test.npy', allow_pickle=True)
        self.ixtoword,self.wordtoix,self.filenames,self.all_txt,self.bboxs,self.attrs,self.class_id,self.all_captions=npy

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

        ix = list(np.arange(attr.__len__()))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:3]
        ix = np.sort(ix)
        attr = np.array(attr)[ix]

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


# import nltk
# nltk.download()
# nltk.download('averaged_perceptron_tagger')
class Coco_Dataset(data.Dataset):
    def __init__(self,split):
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])
        self.filenames, self.all_captions, self.attrs, self.ixtoword, self.wordtoix, self.n_words = self.load_coco_dataset(self.split)
        self.class_id = np.arange(self.filenames.__len__())

        # dict={'caps':self.all_captions,'attrs':self.attrs}
        # file=open('./data/coco/'+split+'.pickle','wb')
        # pickle.dump(dict,file)
        # file.close()

    def __getitem__(self, index):
        if self.filenames[index]=='COCO_train2014_000000167126':
            index=0

        img_path = self.filenames[index]
        if self.split=='train':
            img_path='./data/coco/images/train2014/'+img_path+'.jpg'
        else:
            img_path = './data/coco/images/test2014/' + img_path+'.jpg'
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = self.norm(img)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)

        cap_idx = random.randint(5 * index, 5 * index + 4)

        cap = self.all_captions[cap_idx]
        attr = self.attrs[cap_idx]


        attr = np.array(attr)

        new_attr=np.zeros([3,5])
        random.shuffle(attr)
        for idxi,lst in enumerate(attr,0):
            for idxj,val in enumerate(lst,0):
                new_attr[idxi][idxj]=val
                if idxj==4:
                    break
            if idxi==2:
                break
        attr=new_attr
        new_cap=np.zeros([30])
        len=min(30,cap.__len__())
        new_cap[:len]=cap[:len]
        cap=new_cap

        cap_len = len

        if np.count_nonzero(attr[1]) == 0:
            attr[1] = attr[0]
        if np.count_nonzero(attr[2]) == 0:
            attr[2] = attr[0]
        attr = attr.astype(np.int32)
        attr = torch.tensor(attr, dtype=torch.int32)
        img = Variable(img)
        attr = Variable(attr)

        img = Variable(img)
        cap = np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]

        return img, attr, cap, cap_len, class_id

    def __len__(self):
        return len(self.filenames)

    def load_coco_dataset(self,split):
        import pickle

        filepath = './data/coco/captions.pickle'
        file = open('./data/coco/train/filenames.pickle', 'rb')
        train_names = pickle.load(file)
        file = open('./data/coco/test/filenames.pickle', 'rb')
        test_names = pickle.load(file)

        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions, train_attrs, test_attrs = x[0], x[1], x[4], x[5]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            attrs = train_attrs
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            attrs = test_attrs
            filenames = test_names
        if(split=='train'):
            file = open('./data/coco/train.pickle', 'rb')
            train_names = pickle.load(file)
            captions=train_names['caps']
            attrs=train_names['attrs']
        else:
            file = open('./data/coco/test.pickle', 'rb')
            test_names = pickle.load(file)
            captions=test_names['caps']
            attrs=test_names['attrs']
        return filenames, captions, attrs, ixtoword, wordtoix, n_words
    def getattrs(self,captions):
        all_attrs=[]
        for cap in captions:
            tokens=[]
            for i in range(cap.__len__()):
                tokens.append(self.ixtoword[cap[i]])

            #cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            #tokenizer = RegexpTokenizer(r'\w+')
            #tokens = tokenizer.tokenize(cap.lower())
            # sentence = tokenizer.tokenize(text.lower())

            # Attribute extraction
            sentence_tag = nltk.pos_tag(tokens)

            grammar = "NP: {<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+|<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+<IN>+<NN|NNS>+|<VB|VBD|VBG|VBN|VBP|VBZ>+<CD|DT>*<JJ|PRP$>*<NN|NNS>+|<IN>+<DT|CD|JJ|PRP$>*<NN|NNS>+<IN>*<CD|DT>*<JJ|PRP$>*<NN|NNS>*}"
            cp = nltk.RegexpParser(grammar)
            tree = cp.parse(sentence_tag)

            attr_list = []

            for i in range(len(tree)):
                if type(tree[i]) == nltk.tree.Tree:
                    attr = []
                    for j in range(len(tree[i])):
                        attr.append(self.wordtoix[tree[i][j][0]])
                    attr_list.append(attr)
            all_attrs.append(attr_list)
        return all_attrs
    def load_captions(self,data_dir, filenames):
        all_captions = []
        all_attr = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    #    print(cnt, "cap:", cap)
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # sentence = tokenizer.tokenize(text.lower())

                    # Attribute extraction
                    sentence_tag = nltk.pos_tag(tokens)

                    grammar = "NP: {<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+|<CD|DT|JJ>*<JJ|PRP$>*<NN|NNS>+<IN>+<NN|NNS>+|<VB|VBD|VBG|VBN|VBP|VBZ>+<CD|DT>*<JJ|PRP$>*<NN|NNS>+|<IN>+<DT|CD|JJ|PRP$>*<NN|NNS>+<IN>*<CD|DT>*<JJ|PRP$>*<NN|NNS>*}"
                    cp = nltk.RegexpParser(grammar)
                    tree = cp.parse(sentence_tag)

                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    attr_list = []

                    for i in range(len(tree)):
                        if type(tree[i]) == nltk.tree.Tree:
                            attr = []
                            for j in range(len(tree[i])):
                                attr.append(tree[i][j][0])
                            attr_list.append(attr)
                    all_attr.append(attr_list)

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)  # [cnt, num_words]
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d, len(captions)=%d, embedding_num=%d'
                          % (filenames[i], cnt, len(captions), self.embeddings_num))
        return all_captions, all_attr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bird')  # [bird, flower, celeba]
    args = parser.parse_args()
    if args.dataset=='bird':
        captions=[]
        with open('./birds_data/bird_selected_captions.txt','r') as f:
            for i in range(300):
                cap=f.readline()
                cap=cap.replace('\n','')
                idx=cap.index(' ')
                cap=cap[idx+1:]
                captions.append(cap)
        attrs=[]
        with open('./birds_data/bird_selected_captions_attributes.txt','r') as f:
            for i in range(300):
                attr=f.readline()
                attr=attr.replace('\n','')
                idx=attr.index(' ')
                attr=attr[idx+1:]
                attrs.append(attr.split(';'))
        # captions=['the bird forehead is black and its eyes are white to the back pillow, forming a white pillow ring.',
        #           'this bird has black beak and feet, black forehead to neck, gray back, gray blue wings and tail, and white end of primary feather.',
        #           'the bird  brown upper body feathers from the top of the head to the back and shoulders, the coverts on the waist and tail turn bright brown.',
        #           'this bird has a long and strong mouth, a long head, neck and feet. Its mouth and feet are red. The feathers on the body are black except the chest and abdomen are pure white.',
        #           'this bird has brown belly and white head, with large white spots on the wing, which is very striking against the background of black wings and back.',
        #           'the bird feathers from the top of the head to the belly are red which is very striking against the background of wings and back',
        #           'this bird is very small in size with red color on the belly',
        #           'this bird is green with black and has a very short beak',
        #           'the bird has a long white tail and do not have a visible crown',
        #           'this particular bird has a belly that is black and yellow',
        #           'this bird is big with a red head and wings as white as snow']
        _, attrs_pre=DEA_GAN_get_attributes(captions)
        # attrs_pre = get_attributes(captions,args.dataset)
        print(attrs_pre)
        sim=calc_similarity(attrs,attrs_pre)
        print(sim)
    elif args.dataset=='flower':
        captions = []
        with open('./birds_data/flower_selected_captions.txt', 'r') as f:
            for i in range(300):
                cap = f.readline()
                cap = cap.replace('\n', '')
                idx = cap.index(' ')
                cap = cap[idx + 1:]
                captions.append(cap)
        attrs = []
        with open('./birds_data/flower_selected_captions_attributes.txt', 'r') as f:
            for i in range(300):
                attr = f.readline()
                attr = attr.replace('\n', '')
                idx = attr.index(' ')
                attr = attr[idx + 1:]
                attrs.append(attr.split(';'))
        attrs_pre = get_attributes(captions,args.dataset)
        print(attrs_pre)
        sim = calc_similarity(attrs, attrs_pre)
        print(sim)
    elif args.dataset=='celeba':
        captions = []
        with open('./birds_data/celeba_selected_captions.txt', 'r') as f:
            for i in range(300):
                cap = f.readline()
                cap = cap.replace('\n', '')
                idx = cap.index(' ')
                cap = cap[idx + 1:]
                captions.append(cap)
        attrs = []
        with open('./birds_data/celeba_selected_captions_attributes.txt', 'r') as f:
            for i in range(300):
                attr = f.readline()
                attr = attr.replace('\n', '')
                idx = attr.index(' ')
                attr = attr[idx + 1:]
                attrs.append(attr.split(';'))
        attrs_pre = get_attributes(captions,args.dataset)
        print(attrs_pre)
        sim = calc_similarity(attrs, attrs_pre)
        print(sim)

