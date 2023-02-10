#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:29:01 2022

@author: jianyu

This script contains the three vae models and two MLPs.

The vae implementation is based on:
https://github.com/AntixK/PyTorch-VAE

The resnet implementation is based on:
https://github.com/d2l-ai/d2l-zh

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# %% CVML_base network
def CVML_base():

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.cov = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2),
                nn.ELU(), nn.BatchNorm2d(128),

                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.ELU(), nn.BatchNorm2d(256),

                nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
                nn.ELU(), nn.BatchNorm2d(256),

                nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
                nn.ELU(), nn.BatchNorm2d(512),

                nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
                nn.ELU(), nn.BatchNorm2d(512),
                nn.Flatten())

            self.fc_mu = nn.Linear(512*4*4, 128)
            self.fc_logvar = nn.Linear(512*4*4, 128)

        def forward(self, x):
            x = self.cov(x)
            x_mu = self.fc_mu(x)
            x_logvar = self.fc_logvar(x)
            return x_mu, x_logvar

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.dense = nn.Sequential(
                nn.Linear(128, 512*4*4),
                nn.ELU(), nn.BatchNorm1d(512*4*4),
                nn.Unflatten(1, (512, 4, 4)))

            self.conv = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
                nn.ELU(), nn.BatchNorm2d(512),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2),
                nn.ELU(), nn.BatchNorm2d(256),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                nn.ELU(), nn.BatchNorm2d(256),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
                nn.ELU(), nn.BatchNorm2d(128),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid())

        def forward(self, x):
            x = self.dense(x)
            x = self.conv(x)
            return x

    class Conv_VAE(nn.Module):
        def __init__(self):
            super(Conv_VAE, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            latent_mu, latent_logvar = self.encoder(x)
            latent_s = self.latent_sample(latent_mu, latent_logvar)
            x_recon = self.decoder(latent_s)
            return x_recon, latent_mu, latent_logvar

        def latent_sample(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent_var = eps.mul(std).add_(mu)
            return latent_var

    return Conv_VAE()


# %% residual connection

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=5, padding=2, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=5, padding=2)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.elu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.elu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


# %% CVML_18 network
def CVML_18():

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.cov = nn.Sequential(

                nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(128), nn.ELU(),

                *resnet_block(128, 128, 2, first_block=True),

                *resnet_block(128, 256, 2),

                *resnet_block(256, 256, 2),

                *resnet_block(256, 512, 2),

                *resnet_block(512, 512, 2),

                nn.Flatten())

            self.fc_mu = nn.Linear(512*4*4, 128)
            self.fc_logvar = nn.Linear(512*4*4, 128)

        def forward(self, x):
            x = self.cov(x)
            x_mu = self.fc_mu(x)
            x_logvar = self.fc_logvar(x)
            return x_mu, x_logvar

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.dense = nn.Sequential(
                nn.Linear(128, 512*4*4),
                nn.BatchNorm1d(512*4*4), nn.ELU(),
                nn.Unflatten(1, (512, 4, 4)))

            self.conv = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(512), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid())

        def forward(self, x):
            x = self.dense(x)
            x = self.conv(x)
            return x

    class Conv_VAE(nn.Module):
        def __init__(self):
            super(Conv_VAE, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            latent_mu, latent_logvar = self.encoder(x)
            latent_s = self.latent_sample(latent_mu, latent_logvar)
            x_recon = self.decoder(latent_s)
            return x_recon, latent_mu, latent_logvar

        def latent_sample(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent_var = eps.mul(std).add_(mu)
            return latent_var

    return Conv_VAE()


# %% CVML_34 network
def CVML_34():

    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.cov = nn.Sequential(

                nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(128), nn.ELU(),

                *resnet_block(128, 128, 3, first_block=True),

                *resnet_block(128, 256, 4),

                *resnet_block(256, 256, 6),

                *resnet_block(256, 512, 3),

                *resnet_block(512, 512, 2),

                nn.Flatten())

            self.fc_mu = nn.Linear(512*4*4, 128)
            self.fc_logvar = nn.Linear(512*4*4, 128)

        def forward(self, x):
            x = self.cov(x)
            x_mu = self.fc_mu(x)
            x_logvar = self.fc_logvar(x)
            return x_mu, x_logvar

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.dense = nn.Sequential(
                nn.Linear(128, 512*4*4),
                nn.BatchNorm1d(512*4*4), nn.ELU(),
                nn.Unflatten(1, (512, 4, 4)))
            self.conv = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(512), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(128), nn.ELU(),

                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid())

        def forward(self, x):
            x = self.dense(x)
            x = self.conv(x)
            return x

    class Conv_VAE(nn.Module):
        def __init__(self):
            super(Conv_VAE, self).__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, x):
            latent_mu, latent_logvar = self.encoder(x)
            latent_s = self.latent_sample(latent_mu, latent_logvar)
            x_recon = self.decoder(latent_s)
            return x_recon, latent_mu, latent_logvar

        def latent_sample(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent_var = eps.mul(std).add_(mu)
            return latent_var

    return Conv_VAE()


# %% MLP
def R_MLP():

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(128, 64), nn.ELU(),
                nn.Linear(64, 32), nn.ELU(),
                nn.Linear(32, 16), nn.ELU(),
                nn.Linear(16, 6))

        def forward(self, x):
            x = self.net(x)
            return x

    return MLP()


def T_MLP():

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(133, 64), nn.ELU(),
                nn.Linear(64, 32), nn.ELU(),
                nn.Linear(32, 16), nn.ELU(),
                nn.Linear(16, 8), nn.ELU(),
                nn.Linear(8, 2))

        def forward(self, x):
            x = self.net(x)
            return x

    return MLP()