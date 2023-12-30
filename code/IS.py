import numpy as np
import os
from PIL import Image
import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

from scipy.stats import entropy
from torch.utils.data import DataLoader
from dataprocess import Bird_Dataset
from Encoder import RNN_ENCODER,CNN_ENCODER,NetG,NetD,NetC
import argparse
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d

from vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2,2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def inception_score(args, test_loader, text_encoder, netG, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    dl_length = test_loader.__len__()
    imgs_num = dl_length * args.batch_size

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


    loop = tqdm(total=int(dl_length))
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

            attrs_len = torch.Tensor([3] * cap_lens.size(0)).to(device)
            _, attr_emb0 = text_encoder(attrs[:, 0, :], attrs_len, hidden)
            _, attr_emb1 = text_encoder(attrs[:, 1, :], attrs_len, hidden)
            _, attr_emb2 = text_encoder(attrs[:, 2, :], attrs_len, hidden)
            attr_emb0,attr_emb1,attr_emb2=attr_emb0.detach(),attr_emb1.detach(),attr_emb2.detach()
            attr_emb0,attr_emb1,attr_emb2=attr_emb0.to(device).requires_grad_(),attr_emb1.to(device).requires_grad_(),attr_emb2.to(device).requires_grad_()


            #noise = torch.randn(args.batch_size, args.noise_dim).to(device)

            noise = truncated_noise(args.batch_size, args.noise_dim, 0.88)
            noise = torch.tensor(noise, dtype=torch.float).to(device)
            fake = netG(noise, sent_emb, words_embs, attr_emb0,attr_emb1,attr_emb2)


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


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers(default: 4)')
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--sample_times',default=1,type=int)
    args = parser.parse_args()
    return args
