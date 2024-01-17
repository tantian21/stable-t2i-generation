from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import Bird_Dataset_DF_GAN
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import RNN_ENCODER, CNN_ENCODER,G_DCGAN, G_NET
from miscc.utils import weights_init
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
import torch.nn as nn
from miscc.losses import words_loss,sent_loss, KL_loss
from tqdm import tqdm
from model import D_NET64, D_NET128, D_NET256
from cal_fid import InceptionV3, calculate,calculate_frechet_distance

from IS import inception_score
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)




device = torch.device("cuda")

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file',default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=100,help='manual seed')
    parser.add_argument('--noise_dim', type=int, default=100,help='noise dim')
    parser.add_argument('--batch_size', default=8, type=int)#24 or 12 or 8
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--sample_times',default=10,type=int)
    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get data loader
    train_dataset = Bird_Dataset_DF_GAN('train')
    test_dataset = Bird_Dataset_DF_GAN('test')
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    image_encoder = CNN_ENCODER(256)
    state_dict = torch.load('../DAMSMencoders/bird/image_encoder200.pth', map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()
    image_encoder=image_encoder.to(device)

    text_encoder = RNN_ENCODER(train_dataset.ixtoword.__len__(), nhidden=256)
    state_dict = torch.load('../DAMSMencoders/bird/text_encoder200.pth', map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()
    text_encoder=text_encoder.to(device)

    netG = G_NET()
    netsD=[]
    netsD.append(D_NET64())
    netsD.append(D_NET128())
    netsD.append(D_NET256())
    netG.apply(weights_init)
    for i in range(len(netsD)):
        netsD[i].apply(weights_init).to(device)
    netG=netG.to(device)

    optimizersD = []
    num_Ds = len(netsD)
    #for i in range(num_Ds):
    #    opt = optim.Adam(netsD[i].parameters(), lr=0.0002, betas=(0.5, 0.999))
    #    optimizersD.append(opt)
    #optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(), lr=4e-5)
        optimizersD.append(opt)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-5)

    batch_size = args.batch_size
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(device)
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(device)
    match_labels = Variable(torch.LongTensor(range(batch_size))).to(device)


    noise = Variable(torch.FloatTensor(batch_size, args.noise_dim)).to(device)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx]).to(device)

    from torch.nn.functional import adaptive_avg_pool2d
    dl_length = test_loader.__len__()
    imgs_num = dl_length * args.batch_size
    pred_arr = np.zeros((imgs_num, 2048))
    for step, data in enumerate(test_loader, 0):
        start = step * args.batch_size
        end = start + args.batch_size
        imgs, _, _, _, _ = data
        imgs = np.array(imgs[-1])
        imgs = torch.tensor(imgs)
        imgs = imgs.to(torch.device("cuda"))

        pred = inception_v3(imgs)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(args.batch_size, -1)
    m1 = np.mean(pred_arr, axis=0)
    s1 = np.cov(pred_arr, rowvar=False)

    checkpoint = torch.load('../models/bird_AttnGAN2.pth',map_location=lambda storage, loc: storage)
    netG.load_state_dict(checkpoint)
    checkpoint = torch.load('./checkpoints/state_epoch_best_022.pth', map_location=lambda storage, loc: storage)
    #netG.load_state_dict(checkpoint['model']['netG'])
    netsD[0].load_state_dict(checkpoint['model']['netD64'])
    netsD[1].load_state_dict(checkpoint['model']['netD128'])
    netsD[2].load_state_dict(checkpoint['model']['netD256'])

    for p in netG.parameters():
        p.requires_grad = False
    netG.eval()

    mn_fid = calculate(args, test_loader, text_encoder, netG, inception_v3, m1, s1)
    print(mn_fid)

    iscore = inception_score(args, test_loader, text_encoder, netG, resize=True, splits=1)[0]
    print(iscore)
    mn_fid=200

    for p in netG.parameters():
        p.requires_grad = True
    netG.train()

    for epoch in range(1000):
        loop = tqdm(total=int(train_loader.__len__()))
        for step, data in enumerate(train_loader, 0):
            imgs, attrs,caps,cap_lens,class_ids = data

            sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
            caps = caps[sorted_cap_indices].to(device)
            imgs64,imgs128,imgs256=imgs[0],imgs[1],imgs[2]
            imgs64,imgs128,imgs256 = imgs64[sorted_cap_indices].to(device),imgs128[sorted_cap_indices].to(device),imgs256[sorted_cap_indices].to(device)
            imgs=[imgs64,imgs128,imgs256]
            attrs = attrs[sorted_cap_indices].to(device)
            class_ids=class_ids[sorted_cap_indices]
            class_ids=np.array(class_ids)

            with torch.no_grad():
                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder(caps, sorted_cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                attrs_len = torch.Tensor([5] * cap_lens.size(0))
                words_embs0, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attrs_len, hidden)
                words_embs1, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attrs_len, hidden)
                words_embs2, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attrs_len, hidden)
                attr_emb0,attr_emb1,attr_emb2=attr_emb0.detach(),attr_emb1.detach(),attr_emb2.detach()
                words_embs0,words_embs1,words_embs2=words_embs0.detach(),words_embs1.detach(),words_embs2.detach()

            mask = (caps == 0)
            mask0 = (attrs[:, 0:1, :].squeeze() == 0)
            mask1 = (attrs[:, 1:2, :].squeeze() == 0)
            mask2 = (attrs[:, 2:3, :].squeeze() == 0)
            mask0[:,0]=False
            mask1[:,0]=False
            mask2[:,0]=False
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            noise.data.normal_(0, 1)
            fake_imgs, _, mu, logvar= netG(noise, sent_emb, words_embs, mask,attr_emb0,attr_emb1,attr_emb2,mask0,mask1,mask2,words_embs0,words_embs1,words_embs2)
            errD_total = 0
            for i in range(len(netsD)):
                netD=netsD[i]
                netD.zero_grad()

                real_imgs=imgs[i]
                fake_img=fake_imgs[i]

                real_features = netD(real_imgs)
                fake_features = netD(fake_img.detach())

                cond_real_logits = netD.COND_DNET(real_features, sent_emb)
                cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
                cond_fake_logits = netD.COND_DNET(fake_features, sent_emb)
                cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)

                batch_size = real_features.size(0)
                cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
                cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

                if netD.UNCOND_DNET is not None:
                    real_logits = netD.UNCOND_DNET(real_features)
                    fake_logits = netD.UNCOND_DNET(fake_features)
                    real_errD = nn.BCELoss()(real_logits, real_labels)
                    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
                    errD = ((real_errD + cond_real_errD) / 2. +(fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
                else:
                    errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.

                errD.backward()
                optimizersD[i].step()
                errD_total += errD


            netG.zero_grad()
            errG_total = 0
            for i in range(len(netsD)):
                features = netsD[i](fake_imgs[i])
                cond_logits = netsD[i].COND_DNET(features, sent_emb)
                cond_errG = nn.BCELoss()(cond_logits, real_labels)
                if netsD[i].UNCOND_DNET is not None:
                    logits = netsD[i].UNCOND_DNET(features)
                    errG = nn.BCELoss()(logits, real_labels)
                    g_loss = errG + cond_errG
                else:
                    g_loss = cond_errG
                errG_total += g_loss
                if i == (len(netsD) - 1):
                    region_features, cnn_code = image_encoder(fake_imgs[i])
                    w_loss0, w_loss1, _ = words_loss(region_features, words_embs,match_labels, cap_lens,class_ids, batch_size)
                    w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
                    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,match_labels, class_ids, batch_size)
                    s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
                    errG_total += w_loss + s_loss

            init_errG_total=errG_total

            kl_loss = KL_loss(mu, logvar)
            errG_total += kl_loss
            errG_total.backward()
            optimizerG.step()


            loop.update(1)
            loop.set_description(f'Training:')
            loop.set_postfix()
        loop.close()

        if epoch%1==0:

            for p in netG.parameters():
                p.requires_grad = False
            netG.eval()

            fid = calculate(args, test_loader, text_encoder, netG, inception_v3, m1, s1)
            print('fid:',fid)

            iscore = inception_score(args, test_loader, text_encoder, netG, resize=True, splits=1)[0]
            print('IS:',iscore)

            if True:#mn_fid>fid:
                mn_fid=min(mn_fid,fid)

                state = {'model': {'netG': netG.state_dict(),
                                   'netD64': netsD[0].state_dict(),
                                   'netD128': netsD[1].state_dict(),
                                   'netD256': netsD[2].state_dict(),},
                         'optimizers': {'optimizerG': optimizerG.state_dict(),
                                        'optimizerD64': optimizersD[0].state_dict(),
                                        'optimizerD128': optimizersD[1].state_dict(),
                                        'optimizerD256': optimizersD[2].state_dict()}
                         }
                torch.save(state, './checkpoints/state_epoch_%03d.pth' % (epoch))

            for param in netG.named_parameters():
                param[1].requires_grad = True
            netG.train()

if __name__ == "__main__":
    train()