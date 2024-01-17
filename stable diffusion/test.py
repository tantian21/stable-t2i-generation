import argparse

import PIL.Image
import matplotlib.pyplot as plt
import torch
import transformers.image_transforms
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import importlib
from ldm.models.diffusion.ddim import DDIMSampler
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
import torchvision
from ldm.models.diffusion.ddpm import LatentDiffusion
import numpy as np
import yaml
from safetensors.torch import load_file,save_file
from safetensors import safe_open
from ldm.modules.diffusionmodules.util import make_ddim_timesteps,noise_like
from tqdm import tqdm
import multiprocessing
import threading
import math
import torch.nn as nn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model_path",type=str,default='./models/Stable-diffusion/v1-5-pruned-emaonly.safetensors',help="Path to pretrained model of stable diffusion.",)
    parser.add_argument("--batch_size",type=int,default=1,help="Batch size.",)
    parser.add_argument("--seed",type=int,default=42,help="the seed (for reproducible sampling)",)
    parser.add_argument("--model_path",type=str,default='C:/Users/TanTian/pythonproject/stable-diffusion-main/models/ldm/stable-diffusion-v1/v1-5-pruned2.ckpt',help="Model path.",)
    parser.add_argument("--config_path",type=str,default="C:/Users/TanTian/pythonproject/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml",help="Model path.",) 
    parser.add_argument("--prompt",type=str,default="a girl has long hair and in a black dress",help="Model path.",)
    parser.add_argument("--H",type=int,default=512,help="image height, in pixel space",)
    parser.add_argument("--W",type=int,default=512,help="image width, in pixel space",)
    parser.add_argument("--C",type=int,default=4,help="latent channels",)
    parser.add_argument("--f",type=int,default=8,help="downsampling factor",)
    parser.add_argument("--ddim_steps",type=int,default=20,help="number of ddim sampling steps",)
    parser.add_argument("--scale",type=float,default=7.5,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
    parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling",)
    parser.add_argument("--num_workers",type=int,default=1,help="Num work.",)
    parser.add_argument("--num_epoch", type=int, default=50,help="Num epoch.", ) 
    parser.add_argument("--lr", type=int, default=1e-5,help="Learning rate.", )
    args = parser.parse_args()
    return args



@torch.no_grad()
def sampling(args,tot_txt,attrs,shape,uc,sampler):
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0., verbose=False)
    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=args.ddim_steps, num_ddpm_timesteps=sampler.model.num_timesteps,verbose=False)
    b = args.batch_size
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    C, H, W = shape
    shape = (b, C, H, W)
    img=torch.randn(shape).to(device)
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    tot_c = sampler.model.get_learned_conditioning([tot_txt[0]])

    attr_c=[sampler.model.get_learned_conditioning([attrs[0][0]]),
            sampler.model.get_learned_conditioning([attrs[1][0]]),
            sampler.model.get_learned_conditioning([attrs[2][0]]),]

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        s = torch.zeros([b]).to(device)

        img, _ = sampler.p_sample_ddim(img, tot_c, attr_c, s, ts, index=index,
                            use_original_steps=False,
                            quantize_denoised=False, temperature=1.,
                            noise_dropout=0., score_corrector=None,
                            corrector_kwargs=None,
                            unconditional_guidance_scale=args.scale,
                            unconditional_conditioning=uc)
    return img



def train_one_step(args,imgs, attrs, tot_txt,model):
    imgs=imgs.to(device)

    uc = model.get_learned_conditioning(args.batch_size * [""])

    tot_c = model.get_learned_conditioning([tot_txt[0]])

    t = torch.randint(0, 1000, (args.batch_size,), device=device).long().to(device)

    encoder_feature = model.encode_first_stage(imgs)
    encoder_feature = model.get_first_stage_encoding(encoder_feature).detach()


    attr_c=[model.get_learned_conditioning([attrs[0][0]]).detach(),
            model.get_learned_conditioning([attrs[1][0]]).detach(),
            model.get_learned_conditioning([attrs[2][0]]).detach(),]

    #x_noisy = torch.randn([args.batch_size,4,64,64]).to(device)
    x_noisy=model.q_sample(encoder_feature,t)
    model_output = model.apply_model(x_noisy, t, tot_c,attr_c,None)


    loss_simple = model.get_loss(model_output, encoder_feature, mean=False).mean([1, 2, 3])
    model.logvar = model.logvar.to(device)
    logvar_t = model.logvar[t].to(device)
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    loss = model.l_simple_weight * loss.mean()
    loss_vlb = model.get_loss(model_output, encoder_feature, mean=False).mean(dim=(1, 2, 3))
    loss_vlb = (model.lvlb_weights[t] * loss_vlb).mean()
    loss += (model.original_elbo_weight * loss_vlb)

    return loss


def train():
    args = parse_args()
    seed_everything(args.seed)

    config = OmegaConf.load(args.config_path) 
    config = config.model
    model=LatentDiffusion(**config.get("params", dict()))

    checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    del checkpoint

    model.train()
    model.requires_grad_(True)
    model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0., verbose=False)

    params_to_optimize = [{'params': model.model.parameters(), 'lr': 1e-08}]
    optimizer = AdamW8bit(params_to_optimize)


    from dataprocess import Coco_Dataset
    from torch.utils.data import DataLoader
    train_dataset = Coco_Dataset('train')
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)



    accelerator = Accelerator( 
        gradient_accumulation_steps=1, 
        mixed_precision='no', 
        cpu=False,  
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_loader
    )

    for epoch in range(args.num_epoch):
        for step, data in enumerate(train_loader, 0):
            imgs, attrs, caps, cap_len, class_id=data
            imgs = imgs.to(device).requires_grad_()

            loss_total=train_one_step(args,imgs, attrs, caps,model)

            accelerator.backward(loss_total)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if((step+1)%200==0):
                print("step:",step,"loss:",loss_total)

            del loss_total
            del imgs

            if((step+1)%5000==0):
                state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
                torch.save(state, './newwork_checkpoints/state_epoch_%03d.pth' % (epoch%5+4))
                print('saved')

def test():
    args = parse_args()
    seed_everything(args.seed) 
    config = OmegaConf.load(args.config_path) 
    config = config.model
    model = LatentDiffusion(**config.get("params", dict()))

    checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    del checkpoint
    del model
    del config


    from dataprocess import Bird_Dataset_DF_GAN
    from torch.utils.data import DataLoader
    test_dataset = Bird_Dataset_DF_GAN('test')
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': False,
        'drop_last': True,
    }
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    print('tot:', len(test_loader))
    with torch.no_grad():
        for step, data in enumerate(test_loader, 0):
            print(step)
            _, attrs, caps,= data
            uc = sampler.model.get_learned_conditioning(args.batch_size * [''])
            shape = [args.C, args.H // args.f, args.W // args.f]
            samples_ddim = sampling(args, caps,attrs, shape,  uc,sampler)

            x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clip(x_samples_ddim, -1, 1)
            x_samples_ddim = x_samples_ddim.float()

            img=x_samples_ddim[0]
            img=torch.clip(img,-1,1)
            img=(img+1)/2
            img=torch.transpose(img,0,2)
            img=torch.transpose(img,0,1)
            img=img.cpu().detach()
            plt.imsave('./attnmap1/'+str(step)+'.png',np.array(img))
            plt.imshow(img)
            plt.show()



if __name__ == '__main__':
    test()
