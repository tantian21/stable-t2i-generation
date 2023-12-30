import os, sys
import os.path as osp
import time
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image, make_grid
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataprocess import Bird_Dataset
from Encoder import RNN_ENCODER,CNN_ENCODER,NetG,NetD,NetC
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from scipy import linalg
import warnings
import scipy
import cv2

from vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class Attention(nn.Module):
    def __init__(self, hidden_size=768):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = value
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32)
        )
        attention_weights = nn.functional.softmax(scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V) + value
        result = self.output_layer(weighted_values) + weighted_values
        return result

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
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers(default: 4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true', default=True,
                        help='whether to sample the dataset with random sampler')
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--seed', type=int, default=100,help='seed')
    parser.add_argument('--save_file', default='./samples/', type=str,help='seed')
    parser.add_argument('--image_embending_dim',type=int,default=256,help='image encoder dim')
    parser.add_argument('--npz_path',default='./data/birds/npz/bird_val256_FIDK0.npz',type=str)
    parser.add_argument('--nf', type=int, default=32,help='nf')
    parser.add_argument('--noise_dim', type=int, default=100,help='noise dim')
    parser.add_argument('--input_dim', type=int, default=256,help='input dim')
    parser.add_argument('--imsize', type=int, default=256,help='input imsize')
    parser.add_argument('--ch_dim', type=int, default=3,help='channel dim')
    parser.add_argument('--sample_times',default=10,type=int)
    args = parser.parse_args()
    return args

def truncated_noise(batch_size=1, dim_z=100, truncation=1.0, seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    #values = truncnorm.rvs(-2.4, 2.4, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    #return 1.72**values-1.08+0.1*values
    values = truncnorm.rvs(-2.0, 2.0, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation*values
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
            sorted_cap_lens=sorted_cap_lens.to(device)
            with torch.no_grad():
                hidden = text_encoder.init_hidden(caps.size(0))
                words_embs, sent_emb = text_encoder(caps, sorted_cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                #In coco dataset, we use phrases in DAE-GAN.
                # attrs_len = torch.Tensor([3] * cap_lens.size(0)).to(device)
                #The original phrase length was 5. We found that truncating longer phrases has a better effect on fid.
                attrs_len = torch.Tensor([5] * cap_lens.size(0)).to(device)

                _, attr_emb0 = text_encoder(attrs[:, 0, :], attrs_len, hidden)
                _, attr_emb1 = text_encoder(attrs[:, 1, :], attrs_len, hidden)
                _, attr_emb2 = text_encoder(attrs[:, 2, :], attrs_len, hidden)
                attr_emb0,attr_emb1,attr_emb2=attr_emb0.detach(),attr_emb1.detach(),attr_emb2.detach()
                attr_emb0,attr_emb1,attr_emb2=attr_emb0.to(device),attr_emb1.to(device),attr_emb2.to(device)


                noise = truncated_noise(args.batch_size, args.noise_dim, 0.9)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
                fake = netG(noise, sent_emb, words_embs, attr_emb0,attr_emb1,attr_emb2)

                fake = norm(fake)


                pred = inception_v3(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred_arr[start:end] = pred.cpu().data.numpy().reshape(args.batch_size, -1)

                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()


    loop.close()

    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def get_m1_s1(path,inception_v3):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
        return m, s
    else:
        imgs_num = len(os.listdir(path))
        pred_arr = np.zeros((imgs_num,2048))
        loop = tqdm(total=int(imgs_num))

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.RandomCrop(299),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
        with torch.no_grad():
            for i,img_path in enumerate(os.listdir(path)):
                img_path=path+img_path
                img=Image.open(img_path)
                img=transform(img).to(device)
                if img.shape[0]!=3:
                    img=img.repeat(3,1,1)
                img=torch.unsqueeze(img,0)
                pred=inception_v3(img)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                pred_arr[i] = pred.cpu().data.numpy().reshape(1, -1)

                loop.update(1)
                loop.set_description(f'test_gt:')
                loop.set_postfix()

        m = np.mean(pred_arr, axis=0)
        s = np.cov(pred_arr, rowvar=False)
        loop.close()

        return m, s
