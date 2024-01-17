from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import torch
import io
import time
import numpy as np
import torchvision.utils
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET
import random
import torchvision.transforms as transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import warnings
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


device = torch.device("cuda")

class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad
    def forward(self, inp):
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear', align_corners=True)
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
def calculate(args,test_loader,text_encoder, netG,inception_v3,m1, s1):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dl_length=test_loader.__len__()
    imgs_num = dl_length *  args.batch_size * args.sample_times
    pred_arr = np.zeros((imgs_num,2048))
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    loop = tqdm(total=int(dl_length * args.sample_times))
    for time in range(args.sample_times):
        for step, data in enumerate(test_loader, 0):
            start = step * args.batch_size + time * dl_length  * args.batch_size
            end = start + args.batch_size
            _, attrs,caps,cap_lens,_ = data
            sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
            caps = caps[sorted_cap_indices].to(device)
            attrs = attrs[sorted_cap_indices].to(device)
            with torch.no_grad():
                hidden = text_encoder.init_hidden(caps.size(0))
                words_embs, sent_emb = text_encoder(caps, sorted_cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                attrs_len = torch.Tensor([5] * cap_lens.size(0))
                words_embs0, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attrs_len, hidden)
                words_embs1, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attrs_len, hidden)
                words_embs2, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attrs_len, hidden)
                attr_emb0,attr_emb1,attr_emb2=attr_emb0.detach(),attr_emb1.detach(),attr_emb2.detach()
                words_embs0,words_embs1,words_embs2=words_embs0.detach(),words_embs1.detach(),words_embs2.detach()

                noise = Variable(torch.FloatTensor(args.batch_size, args.noise_dim), volatile=True)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
                noise.data.normal_(0, 1)
                mask = (caps == 0)
                mask0 = (attrs[:, 0:1, :].squeeze() == 0)
                mask1 = (attrs[:, 1:2, :].squeeze() == 0)
                mask2 = (attrs[:, 2:3, :].squeeze() == 0)
                mask0[:, 0] = False
                mask1[:, 0] = False
                mask2[:, 0] = False
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                mask=mask.to(device)
                fake,_,_,_= netG(noise, sent_emb, words_embs,mask,attr_emb0,attr_emb1,attr_emb2,mask0,mask1,mask2,words_embs0,words_embs1,words_embs2)
                fake=fake[-1]

                fake = norm(fake)
                pred = inception_v3(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred_arr[start:end] = pred.cpu().data.numpy().reshape(args.batch_size, -1)
                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()
    loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value




'''
def inception_score(args, test_loader, text_encoder, netG, resize=False, splits=1):
    dl_length = test_loader.__len__()
    imgs_num = dl_length * args.batch_size
    from torchvision.models.inception import inception_v3
    from scipy.stats import entropy
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear')
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
    # Get predictions
    preds = np.zeros((imgs_num, 1000))

    loop = tqdm(total=int(dl_length ))
    for step, data in enumerate(test_loader, 0):
        start = step * args.batch_size
        end = start + args.batch_size
        _, attrs, caps, cap_lens, _ = data
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        caps = caps[sorted_cap_indices].to(device)
        attrs = attrs[sorted_cap_indices].to(device)
        sorted_cap_lens = sorted_cap_lens.to(device)
        with torch.no_grad():
            hidden = text_encoder.init_hidden(caps.size(0))
            words_embs, sent_emb = text_encoder(caps, sorted_cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            attrs_len = torch.Tensor([5] * cap_lens.size(0))
            _, attr_emb0 = text_encoder(attrs[:, 0:1, :].squeeze(), attrs_len, hidden)
            _, attr_emb1 = text_encoder(attrs[:, 1:2, :].squeeze(), attrs_len, hidden)
            _, attr_emb2 = text_encoder(attrs[:, 2:3, :].squeeze(), attrs_len, hidden)
            attr_emb0, attr_emb1, attr_emb2 = attr_emb0.detach(), attr_emb1.detach(), attr_emb2.detach()

            noise = Variable(torch.FloatTensor(args.batch_size, args.noise_dim), volatile=True)
            noise = torch.tensor(noise, dtype=torch.float).to(device)
            noise.data.normal_(0, 1)
            mask = (caps == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            mask = mask.to(device)
            fake, _, _, _, _, _, _ = netG(noise, sent_emb, words_embs, mask, attr_emb0, attr_emb1, attr_emb2)
            fake = fake[-1]

            preds[start:end] = get_pred(fake)

            loop.update(1)
            loop.set_description(f'Evaluating:')
            loop.set_postfix()

    loop.close()

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (imgs_num // splits): (k+1) * (imgs_num // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)
'''