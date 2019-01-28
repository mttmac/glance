import os, math, time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from itertools import product
import scipy.stats as stats

depth = 64      # initial depth to convolve channels into
n_channels = 3  # number of channels (RGB)
filt_size = 4   # convolution filter size
stride = 2      # stride for conv
pad = 1         # padding added for conv

class VAE2D(nn.Module):
    def __init__(self, img_size, n_latent=300):
        
        # Model setup
        #############
        super(VAE2D, self).__init__()
        self.n_latent = n_latent
        n = math.log2(img_size)
        assert n == round(n), 'Image size must be a power of 2'  # restrict image input sizes permitted
        assert n >= 3, 'Image size must be at least 8'           # low dimensional data won't work well
        n = int(n)

        # Encoder - first half of VAE
        #############################
        self.encoder = nn.Sequential()  
        # input: n_channels x img_size x img_size
        # ouput: depth x conv_img_size^2
        # conv_img_size = (img_size - filt_size + 2 * pad) / stride + 1
        self.encoder.add_module('input-conv', nn.Conv2d(n_channels, depth, filt_size, stride, pad,
                                                        bias=True))
        self.encoder.add_module('input-relu', nn.ReLU(inplace=True))
        
        # Add conv layer for each power of 2 over 3 (min size)
        # Pyramid strategy with batch normalization added
        for i in range(n - 3):
            # input: depth x conv_img_size^2
            # output: o_depth x conv_img_size^2
            # i_depth = o_depth of previous layer
            i_depth = depth * 2 ** i
            o_depth = depth * 2 ** (i + 1)
            self.encoder.add_module(f'pyramid_{i_depth}-{o_depth}_conv',
                                    nn.Conv2d(i_depth, o_depth, filt_size, stride, pad, bias=True))
            self.encoder.add_module(f'pyramid_{o_depth}_batchnorm',
                                    nn.BatchNorm2d(o_depth))
            self.encoder.add_module(f'pyramid_{o_depth}_relu',
                                    nn.ReLU(inplace=True))
        
        # Latent representation
        #######################
        # Convolve the encoded image into the latent space, once for mu and once for logvar
        max_depth = depth * 2 ** (n - 3)
        self.conv_mu = nn.Conv2d(max_depth, n_latent, filt_size)      # return the mean of the latent space 
        self.conv_logvar = nn.Conv2d(max_depth, n_latent, filt_size)  # return the log variance of the same
        
        
        # Decoder - second half of VAE
        ##############################
        self.decoder = nn.Sequential()
        # input: max_depth x conv_img_size^2 (8 x 8)  TODO double check sizes
        # output: n_latent x conv_img_size^2 (8 x 8)
        # default stride=1, pad=0 for this layer
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(n_latent, max_depth, filt_size, bias=True))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(max_depth))
        self.decoder.add_module('input-relu', nn.ReLU(inplace=True))
    
        # Reverse the convolution pyramids used in the encoder
        for i in range(n - 3, 0, -1):
            i_depth = depth * 2 ** i
            o_depth = depth * 2 ** (i - 1)
            self.decoder.add_module(f'pyramid_{i_depth}-{o_depth}_conv',
                                    nn.ConvTranspose2d(i_depth, o_depth, filt_size, stride, pad, bias=True))
            self.decoder.add_module(f'pyramid_{o_depth}_batchnorm',
                                    nn.BatchNorm2d(o_depth))
            self.decoder.add_module(f'pyramid_{o_depth}_relu', nn.ReLU(inplace=True))
        
        # Final transposed convolution to return to img_size
        # Final activation is tanh instead of relu to allow negative pixel output
        self.decoder.add_module('output-conv', nn.ConvTranspose2d(depth, n_channels,
                                                                  filt_size, stride, pad, bias=True))
        self.decoder.add_module('output-tanh', nn.Tanh())

        # Model weights init
        ####################
        # Randomly initialize the model weights using kaiming method
        # Reference: "Delving deep into rectifiers: Surpassing human-level
        # performance on ImageNet classification" - He, K. et al. (2015)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, imgs):
        """
        Encode the images into latent space vectors (mean and log variance representation)
        input:  imgs   [batch_size, 3, 256, 256]
        output: mu     [batch_size, n_latent, 1, 1]
                logvar [batch_size, n_latent, 1, 1]
        """
        output = self.encoder(imgs)
        output = output.squeeze(-1).squeeze(-1)
        return [self.conv_mu(output), self.conv_logvar(output)]

    def sample(self, mu, logvar):
        """
        Generates a random latent vector using the trained mean and log variance representation
        input:  mu     [batch_size, n_latent, 1, 1]
                logvar [batch_size, n_latent, 1, 1]
        output: gen    [batch_size, n_latent, 1, 1]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            gen = torch.randn_like(std)
            return gen.mul(std).add_(mu)
        else:
            return mu  # most likely representation

    def decode(self, gen):
        """
        Restores an image representation from the generated latent vector
        input:  gen      [batch_size, n_latent, 1, 1]
        output: gen_imgs [batch_size, 3, 256, 256]
        """
        return self.decoder(gen)

    def forward(self, imgs):
        """
        Generates reconstituted images from input images based on learned representation
        input: imgs     [batch_size, 3, 256, 256]
        ouput: gen_imgs [batch_size, 3, 256, 256]
               mu       [batch_size, n_latent]
               logvar   [batch_size, n_latent]
        """
        mu, logvar = self.encode(imgs)
        gen = self.sample(mu, logvar)
        for tensor in (mu, logvar):
            tensor = tensor.squeeze(-1).squeeze(-1)
        return self.decode(gen), mu, logvar

    
class VAE2DLoss(nn.Module):

    def __init__(self, kl_weight=1):
        super(VAE2DLoss, self).__init__()
        self.kl_weight = kl_weight

    def forward(self, gen_imgs, imgs, mu, logvar, reduce=True):
        """
        input:  gen_imgs [batch_size, n_channels, img_size, img_size]
                imgs     [batch_size, n_channels, img_size, img_size]
                mu       [batch_size, n_latent]
                logvar   [batch_size, n_latent]
        output: loss      scalar (-ELBO)
                loss_desc {'KL': KL, 'logp': gen_err}
        """
        batch_size = imgs.shape[0]
        gen_err = (imgs - gen_imgs).pow(2).reshape(batch_size, -1)
        gen_err = 0.5 * torch.sum(gen_err, dim=-1)
        if reduce:
            gen_err = torch.mean(gen_err)

        # KL(q || p) = -log_sigma + sigma^2/2 + mu^2/2 - 1/2
        KL = (-logvar + logvar.exp() + mu.pow(2) - 1) * 0.5
        KL = torch.sum(KL, dim=-1)
        if reduce:
            KL = torch.mean(KL)

        loss = gen_err + self.kl_weight * KL
        return loss, {'KL': KL, 'logp': -gen_err}

    
def load_datasets(data_path, img_size, batch_size=32):
    """
    Load the image datasets from vae_train and vae_test
    Transform to correct image size
    """
    data_path = Path(data_path)
    train_path = data_path / 'train/train/'
    val_path = data_path / 'train/val/'
    test_path = data_path / 'test/'
    
    norm_args = {'mean': [0.5] * n_channels,
                 'std': [0.5] * n_channels}
    jitter_args = {'brightness': 0.1,
                   'contrast': 0.1,
                   'saturation': 0.1}  # hue unchanged
    
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),          # vary horizontal position
        transforms.RandomHorizontalFlip(p=0.25),  # vary photo orientation
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ColorJitter(**jitter_args),    # vary photo lighting
        transforms.ToTensor(),
        transforms.Normalize(**norm_args)])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),  # assume center is most important
        transforms.ToTensor(),
        transforms.Normalize(**norm_args)])

    train_ds = datasets.ImageFolder(train_path, train_transform)
    val_ds = datasets.ImageFolder(val_path, test_transform)
    test_ds = datasets.ImageFolder(test_path, test_transform)
    
    
    loader_args = {'shuffle': True,
                   'num_workers': 4}
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, **loader_args)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, **loader_args)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, ** loader_args)
    
    return train_dl, val_dl, test_dl