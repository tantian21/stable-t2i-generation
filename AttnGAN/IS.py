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

device = torch.device("cuda")
def inception_score(args, test_loader, text_encoder, netG, resize=False, splits=1):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
            mask0[:,0]=False
            mask1[:,0]=False
            mask2[:,0]=False
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            mask = mask.to(device)
            fake, _, _, _, = netG(noise, sent_emb, words_embs, mask, attr_emb0, attr_emb1, attr_emb2,mask0,mask1,mask2,words_embs0,words_embs1,words_embs2)
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
