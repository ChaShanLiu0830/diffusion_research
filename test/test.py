import sys
sys.path.append("../")
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, ErrorDiffusion
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset
import argparse
from utils.wandb_init import init_wandb
from datetime import datetime 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_error", action="store_true")
    parser.add_argument("--eff_rate",type = float, default= 1.)
    parser.add_argument("--seed", type = int, default= 10)
    args = parser.parse_args()
    args.filename_prefix = f"train_error{args.train_error}_{args.eff_rate}_{datetime.strftime(datetime.now(), '%Y%m%d')}" if args.train_error else f"train_error{args.train_error}_{datetime.strftime(datetime.now(), '%Y%m%d')}"
    wandb_run = init_wandb(proj_name = "diffusion_research", config = args, name = args.filename_prefix )

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean = (0, 0, 0), std = (1, 1, 1))
    ])
    # transform = transforms.Compose([
    # ])
    class CIFARset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.unnorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            return self.dataset[index][0]
        def unnormalize(self, data):
            return self.unnorm(data)
            
    
    demo_noise = torch.from_numpy(pd.read_pickle("./demo_noise.pkl"))
    # Load CIFAR-10 dataset
    totalset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    totalset = CIFARset(totalset)
    testset = CIFARset(testset)
    total_index = np.arange(0, len(totalset))
    # np.random.shuffle(total_index)
    # trainset = Subset(totalset,total_index[:int(len(total_index)*0.8)])
    validset = Subset(totalset,total_index[int(len(total_index)*0.8):])
    
    
    model = Unet(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    # diffusion = GaussianDiffusion(
    #     model,
    #     image_size = 32,
    #     timesteps = 1000,           # number of steps
    #     sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # )
    diffusion = ErrorDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 1000,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        auto_normalize=True
    )
    

    trainer = Trainer(
        diffusion,
        trainset= totalset,
        validset= validset,
        testset = testset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every= 2500,
        amp = True,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        train_error = args.train_error,
        eff_rate = args.eff_rate, 
        wandb_run = wandb_run,
        fix_noise= demo_noise
    )

    trainer.train()
    
if __name__ == "__main__":
    main()