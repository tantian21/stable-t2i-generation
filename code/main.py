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
sys.path.append("..")
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import parser
from PIL import Image
from torch.autograd import Variable
import torchvision.utils
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from dataprocess import Bird_Dataset_AFM,Coco_Dataset
from Encoder import RNN_ENCODER,CNN_ENCODER,NetG,NetD,NetC
from fid_cal import get_m1_s1,calculate,InceptionV3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2,2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


#########   MAGP   ########
def MA_GP(img, sent,out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,create_graph=True,only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = 2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def predict_loss(predictor, img_feature, attr_feature, negtive):
    output = predictor(img_feature, attr_feature)
    output=torch.squeeze(output)
    if negtive == False:
        err = torch.nn.ReLU()(1.0 - output).mean()
    else:
        err = torch.nn.ReLU()(1.0 + output).mean()
    return output,err

def train(args):
    # text encoder
    # GAN models
    netG = NetG(args.nf, args.noise_dim, args.input_dim, args.imsize, args.ch_dim, args.dataset).to(device)
    netD = NetD(args.nf, 256, args.ch_dim).to(device)
    netC = NetC(args.nf, args.input_dim).to(device)
    netG.train()
    netD.train()
    netC.train()

    G_params = list(netG.parameters())
    optimizerG = torch.optim.Adam(G_params, lr=0.0001 * 0.1, betas=(0.0, 0.9))
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=0.0004 * 0.1, betas=(0.0, 0.9))


    if args.load_model_path!=None and args.dataset=='coco':
        model_path = args.load_model_path
        checkpoint = torch.load(model_path)
        G_checkpoints=checkpoint['model']['netG']
        netG.load_state_dict(G_checkpoints)

        for param in netG.named_parameters():
            param[1].requires_grad = True
        netG.train()

        D_checkpoints = checkpoint['model']['netD']
        netD.load_state_dict(D_checkpoints)
        for p in netD.parameters():
            p.requires_grad = True
        netD.train()

        C_checkpoints = checkpoint['model']['netC']
        netC.load_state_dict(C_checkpoints)
        for p in netC.parameters():
            p.requires_grad = True
        netC.train()


    elif args.load_model_path!=None:
        model_path = args.load_model_path
        checkpoint = torch.load(model_path)
        netG.load_state_dict(checkpoint['model']['netG'])

        for param in netG.named_parameters():
            param[1].requires_grad = True
        netG.train()
        netD.load_state_dict(checkpoint['model']['netD'])
        for p in netD.parameters():
            p.requires_grad = True
        netD.train()
        netC.load_state_dict(checkpoint['model']['netC'])
        for p in netC.parameters():
            p.requires_grad = True
        netC.train()

    batch_size=args.batch_size

    if args.dataset=='bird':
        train_dataset = Bird_Dataset_AFM('train')
        test_dataset = Bird_Dataset_AFM('test')
    elif args.dataset=='coco':
        train_dataset = Coco_Dataset('train')
        test_dataset = Coco_Dataset('test')
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    text_encoder = RNN_ENCODER(train_dataset.ixtoword.__len__(), nhidden=256)
    if args.dataset=='bird':
        state_dict = torch.load('../data/birds/DAMSMencoder/text_encoder200.pth', map_location='cpu')
    elif args.dataset=='coco':
        state_dict = torch.load('../data/coco/DAMSMencoder/text_encoder100.pth', map_location='cpu')
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx]).to(device)
    for p in inception_v3.parameters():
        p.requires_grad = False
    inception_v3.eval()
    if args.dataset=='bird' or args.dataset=='coco':
        m1, s1 = get_m1_s1(args.npz_path, inception_v3)
    else:
        from torch.nn.functional import adaptive_avg_pool2d
        dl_length = test_loader.__len__()
        imgs_num = dl_length * args.batch_size
        pred_arr = np.zeros((imgs_num, 2048))
        for step, data in enumerate(test_loader, 0):
            start = step * args.batch_size
            end = start + args.batch_size
            imgs, _,_, _, _ = data
            imgs = np.array(imgs)
            imgs = torch.tensor(imgs)
            imgs = imgs.to(torch.device("cuda"))

            pred = inception_v3(imgs)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(args.batch_size, -1)
        m1 = np.mean(pred_arr, axis=0)
        s1 = np.cov(pred_arr, rowvar=False)

    if args.load_model_path != None:
        for p in netG.parameters():
            p.requires_grad = False
        netG.eval()
        mn_fid=calculate(args, test_loader, text_encoder, netG, inception_v3, m1, s1)
        print('fid:',mn_fid)
        from IS import inception_score
        iscore = inception_score(args, test_loader, text_encoder, netG, resize=True, splits=1)[0]
        print('iscore:',iscore)
        for param in netG.named_parameters():
            param[1].requires_grad = True
        netG.train()


    for epoch in range(2000):
        loop = tqdm(total=int(train_loader.__len__()))
        for step, data in enumerate(train_loader, 0):
            imgs, attrs,caps,cap_lens,_ = data

            sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
            caps = caps[sorted_cap_indices].to(device)
            imgs=imgs[sorted_cap_indices].to(device).requires_grad_()
            attrs=attrs[sorted_cap_indices].to(device)


            with torch.no_grad():
                hidden = text_encoder.init_hidden(caps.size(0))
                words_embs, sent_emb = text_encoder(caps, sorted_cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                attrs_len = torch.Tensor([5] * cap_lens.size(0))
                _, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attrs_len, hidden)
                _, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attrs_len, hidden)
                _, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attrs_len, hidden)
                attr_emb0,attr_emb1,attr_emb2=attr_emb0.detach(),attr_emb1.detach(),attr_emb2.detach()
            sent_emb=sent_emb.to(device).requires_grad_()
            attr_emb0,attr_emb1,attr_emb2=attr_emb0.to(device).requires_grad_(),attr_emb1.to(device).requires_grad_(),attr_emb2.to(device).requires_grad_()

            real_features= netD(imgs)
            pred_real, errD_real = predict_loss(netC, real_features,sent_emb, negtive=False)  
            mis_features = torch.cat((real_features[batch_size//2:], real_features[:batch_size//2]), dim=0)
            _, errD_mis = predict_loss(netC, mis_features, sent_emb, negtive=True)  

            noise = torch.randn(args.batch_size, args.noise_dim).to(device)

            fake= netG(noise, sent_emb,words_embs,attr_emb0,attr_emb1,attr_emb2)
            fake_features = netD(fake.detach())

            _, errD_fake = predict_loss(netC, fake_features, sent_emb, negtive=True)  

            errD_MAGP = MA_GP(imgs, sent_emb, pred_real)  # MAGP loss

            errD = errD_real + (errD_mis+errD_fake)/2.0 + errD_MAGP 
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()

            fake_features = netD(fake)
            output = netC(fake_features, sent_emb)
            
            errG = -output.mean()

            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()
            loop.update(1)
            loop.set_description(f'Training:')
            loop.set_postfix()


        loop.close()
        if epoch%10 == 0:
            print('saved',epoch)
            state = {'model': {'netG': netG.state_dict(),
                           'netD': netD.state_dict(), 'netC': netC.state_dict()},
                'optimizers': {'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict()}
                 }
            torch.save(state, './Coco_new_checkpoints/state_epoch_%03d.pth' % (epoch))

            for p in netG.parameters():
                p.requires_grad = False
            netG.eval()

            fid = calculate(args, test_loader, text_encoder, netG, inception_v3, m1, s1)
            print('epoch:', epoch, ' fid:', fid)
            from IS import inception_score
            iscore = inception_score(args, test_loader, text_encoder, netG, resize=True, splits=1)[0]

            print('epoch:', epoch, 'iscore:',iscore)

            for param in netG.named_parameters():
                param[1].requires_grad = True
            netG.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bird')#[bird, coco]
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--imsize', type=int, default=256,help='input imsize')
    parser.add_argument('--train', type=bool, default=True,help='train or test')
    parser.add_argument('--noise_dim', type=int, default=100,help='noise dim')
    parser.add_argument('--input_dim', type=int, default=256,help='input dim')
    parser.add_argument('--seed', type=int, default=100,help='seed')
    parser.add_argument('--nf', type=int, default=32,help='nf')
    parser.add_argument('--ch_dim', type=int, default=3,help='channel dim')
    parser.add_argument('--truncation', type=bool, default=True,help='is truncation')
    parser.add_argument('--trunc_rate', type=float, default=0.88,help='truncation rate')
    parser.add_argument('--image_embending_dim',type=int,default=256,help='image encoder dim')
    parser.add_argument('--npz_path',default='../data/birds/npz/bird_val256_FIDK0.npz',type=str)
    #parser.add_argument('--npz_path',default='../data/coco/npz/coco_val256_FIDK0.npz',type=str)
    parser.add_argument('--sample_times',default=10,type=int)#for CUB
    #parser.add_argument('--sample_times',default=1,type=int)#for COCO
    parser.add_argument('--load_model_path',default='./saved_models/bird/CUB.pth')
    #parser.add_argument('--load_model_path',default='./saved_models/coco/coco.pth')
    parser.add_argument('--device',default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help='cuda or cpu')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    train(args)
