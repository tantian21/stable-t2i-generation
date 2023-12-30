import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
import os
import sys
sys.path.append("..")
from torch.autograd import Variable

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = 256  # define a uniform ranker
        model = models.inception_v3(pretrained=True)#加载inception_v3模型
        for param in model.parameters():
            param.requires_grad = False

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3#卷积
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = nn.Conv2d(768, self.nef, kernel_size=(1,1), stride=(1,1),padding=0, bias=False)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = nn.functional.interpolate(x,size=(299, 299), mode='bilinear', align_corners=False)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        features = x
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.view(x.size(0), -1)
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 18
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions
        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden,self.nlayers, batch_first=True,dropout=self.drop_prob,bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,self.nlayers, batch_first=True,dropout=self.drop_prob,bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]
        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb,sent_emb

class NetG(nn.Module):
    def __init__(self, nf, noise_dim, input_dim, imsize, ch_size,dataset=None):
        super(NetG, self).__init__()
        self.nf = nf
        # input noise (batch_size, 100)
        self.fc = nn.Linear(noise_dim, nf*8*4*4)
        # build GBlocks
        self.dataset=dataset

        if dataset=='coco':
            self.GBlocks = nn.ModuleList([])
            for idx, (in_ch, out_ch) in enumerate([(32*8, 32*8),(32*8, 32*8),(32*8, 32*8),(32*8, 32*4),(32*4, 32*2),(32*2, 32*1)]):
                self.GBlocks.append(G_Block(input_dim+noise_dim , in_ch, out_ch, upsample=True))
        else:
            self.g_block0=G_Block(input_dim+noise_dim, 32*8, 32*8, upsample=True)
            self.g_block1=G_Block(input_dim+noise_dim, 32*8, 32*8, upsample=True)
            self.g_block2=G_Block(input_dim+noise_dim, 32*8, 32*8, upsample=True)
            self.g_block3=G_Block(input_dim+noise_dim, 32*8, 32*4, upsample=True)
            self.g_block4=G_Block(input_dim+noise_dim, 32*4, 32*2, upsample=True)
            self.g_block5=G_Block(input_dim+noise_dim, 32*2, 32*1, upsample=True)
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 3, (3,3), (1,1), (1,1)),
            nn.Tanh(),
            )
    def forward(self, noise, sent_emb, words_embs,attr_emb0,attr_emb1,attr_emb2):
        out = self.fc(noise)
        out = out.view(noise.size(0), 8*self.nf, 4, 4)#[batch_size,8*32,4,4]

        cond = torch.cat((noise,sent_emb), dim=1)#[batch_size,100+256]
        cond_attr0 = torch.cat((noise,attr_emb0), dim=1)
        cond_attr1 = torch.cat((noise,attr_emb1), dim=1)
        cond_attr2 = torch.cat((noise,attr_emb2), dim=1)

        if self.dataset=='coco':
            for GBlock in self.GBlocks:
                out_attr0 = GBlock(out, cond_attr0)
                out_attr1 = GBlock(out, cond_attr1)
                out_attr2 = GBlock(out, cond_attr2)
                out = GBlock(out, cond)
                w0 = torch.cosine_similarity(out, out_attr0, dim=1, eps=1e-8)
                w0 = torch.unsqueeze(w0, 1)
                mn = torch.min(w0)
                mx = torch.max(w0)
                w0 = (w0 - mn) / (mx - mn + 1e-8)
                w0 = 1 - w0

                w1 = torch.cosine_similarity(out, out_attr1, dim=1, eps=1e-8)
                w1 = torch.unsqueeze(w1, 1)
                mn = torch.min(w1)
                mx = torch.max(w1)
                w1 = (w1 - mn) / (mx - mn + 1e-8)
                w1 = 1 - w1

                w2 = torch.cosine_similarity(out, out_attr2, dim=1, eps=1e-8)
                w2 = torch.unsqueeze(w2, 1)
                mn = torch.min(w2)
                mx = torch.max(w2)
                w2 = (w2 - mn) / (mx - mn + 1e-8)
                w2 = 1 - w2

                out = out + (w0 * out_attr0 + w1 * out_attr1 + w2 * out_attr2) / 3
        else:
            for g_block in [self.g_block0,self.g_block1,self.g_block2,self.g_block3,self.g_block4,self.g_block5]:
                out_attr0 = g_block(out, cond_attr0)
                out_attr1 = g_block(out, cond_attr1)
                out_attr2 = g_block(out, cond_attr2)
                out = g_block(out, cond)
                w0 = torch.cosine_similarity(out, out_attr0, dim=1, eps=1e-8)
                w0 = torch.unsqueeze(w0, 1)
                mn = torch.min(w0)
                mx = torch.max(w0)
                w0 = (w0 - mn) / (mx - mn + 1e-8)
                w0 = 1 - w0


                w1 = torch.cosine_similarity(out, out_attr1, dim=1, eps=1e-8)
                w1 = torch.unsqueeze(w1, 1)
                mn = torch.min(w1)
                mx = torch.max(w1)
                w1 = (w1 - mn) / (mx - mn + 1e-8)
                w1 = 1 - w1


                w2 = torch.cosine_similarity(out, out_attr2, dim=1, eps=1e-8)
                w2 = torch.unsqueeze(w2, 1)
                mn = torch.min(w2)
                mx = torch.max(w2)
                w2 = (w2 - mn) / (mx - mn + 1e-8)
                w2 = 1 - w2

                out = out + (w0 * out_attr0 + w1 * out_attr1+ w2 * out_attr2)/3

        img = self.to_rgb(out)
        return img#, w0,w1,w2


class NetD(nn.Module):
    def __init__(self, ndf, imsize=256, ch_size=3):
        super(NetD, self).__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        layer_num = int(np.log2(imsize)) - 1
        channel_nums = [ndf * min(2 ** idx, 8) for idx in range(layer_num)]
        in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))
    def forward(self,x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out


class NetC(nn.Module):
    def __init__(self, ndf, cond_dim=256):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf*8+cond_dim, ndf*2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
        )
    def forward(self, out, y):
        y = y.view(-1, self.cond_dim, 1, 1)
        y = y.repeat(1, 1, 4, 4)#[bitch_size,256,4,4]
        h_c_code = torch.cat((out, y), 1)#[bitch_size,512,4,4]
        out = self.joint_conv(h_c_code)#[bitch_size,1,1,1]
        return out



class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, x, y):#[batch,ch,w,h],[batch,dim]
        if self.upsample==True:
            x = F.interpolate(x, scale_factor=2)
        return self.shortcut(x) + self.residual(x, y)


class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        #return x + res
        return x + self.gamma*res


class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):#[batch,ch,w,h],[batch,dim]
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


class Affine(nn.Module):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):#[batch,ch,w,h],[batch,dim]
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

