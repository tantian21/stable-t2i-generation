
from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


from DF_GAN import RNN_ENCODER, CNN_ENCODER
from dataprocess import Bird_Dataset


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPDATE_INTERVAL = 300
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids = data
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices].to(device)
        imgs = imgs[sorted_cap_indices]
        imgs = imgs.to(device).requires_grad_()
        cap_lens=sorted_cap_lens
        class_ids=np.array(class_ids)

        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs)
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),0.25)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0 / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1 / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0 / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1 / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            #img_set, _ = build_super_images(imgs.cpu(), captions,ixtoword, attn_maps, att_sze,max_word_num=10)
            #if img_set is not None:
            #    im = Image.fromarray(img_set)
            #    fullpath = './attention_maps/attention_maps%d.png' % ( step)
            #    im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids = data

        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices].to(device)
        real_imgs = real_imgs[sorted_cap_indices]
        real_imgs = real_imgs.to(device).requires_grad_()
        cap_lens=sorted_cap_lens
        class_ids=np.array(class_ids)


        words_features, sent_code = cnn_model(real_imgs)
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss / step
    w_cur_loss = w_total_loss / step

    return s_cur_loss, w_cur_loss


def build_models(train_dataset):
    # build model ############################################################
    text_encoder = RNN_ENCODER(train_dataset.ixtoword.__len__(), nhidden=256)
    image_encoder = CNN_ENCODER(256)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    print(device)
    text_encoder = text_encoder.to(device)
    image_encoder = image_encoder.to(device)
    labels = labels.to(device)

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    args.manualSeed = 100
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    cudnn.benchmark = True

    imsize = 256
    batch_size = 24


    train_dataset = Flower_Dataset('train')
    test_dataset = Flower_Dataset('test')
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': True,
        'drop_last': True,
    }
    print(train_dataset.__len__())
    print(test_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_kwargs)
    text_encoder, image_encoder, labels, start_epoch = build_models(train_dataset)


    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = 2e-4
        for epoch in range(start_epoch, 600):
            #optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            optimizer = optim.Adam(para, lr=lr)
            epoch_start_time = time.time()
            count = train(train_loader, image_encoder, text_encoder,batch_size, labels, optimizer, epoch,train_dataset.ixtoword, '')
            print('-' * 89)
            if len(test_loader) > 0:
                s_loss, w_loss = evaluate(test_loader, image_encoder,
                                          text_encoder, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > 2e-4/10.:
                lr *= 0.98

            if (epoch%10 == 0):
                torch.save(image_encoder.state_dict(),'./Flower_DAMSM_new/image_encoder%d.pth' % ( epoch))
                torch.save(text_encoder.state_dict(),'./Flower_DAMSM_new/text_encoder%d.pth' % ( epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
