import sys
sys.path.append("../")
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, ErrorDiffusion
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_error", action="store_true")
    parser.add_argument("--eff_rate",type = float, default= 1.)

    
    args = parser.parse_args()
    
    torch.random.manual_seed(10)
    np.random.seed(10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transform = transforms.Compose([
    # ])
    class CIFARset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            return self.dataset[index][0]
    

    # Load CIFAR-10 dataset
    totalset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    totalset = CIFARset(totalset)
    total_index = np.arange(0, len(totalset))
    np.random.shuffle(total_index)
    trainset = Subset(totalset,total_index[:int(len(total_index)*0.8)])
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
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )
    

    trainer = Trainer(
        diffusion,
        trainset= trainset,
        validset= validset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        train_error = args.train_error,
        eff_rate = args.eff_rate
    )

    trainer.train()
    
if __name__ == "__main__":
    main()