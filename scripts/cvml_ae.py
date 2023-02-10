#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 23:29:16 2022

@author: jianyu

This script is used to train the variational autoencoder.

"""

# %% import modules
import os
import torch
import numpy as np
import random
import glob
import argparse
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import CVML_base, CVML_18, CVML_34
from engine import Merge_Data, vae_loss

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, data information
parser = argparse.ArgumentParser(description='train CVML-AE')
parser.add_argument('--object', type=int, help='which object')
parser.add_argument('--model', type=str, help='cvml models',
                    choices=['CVML_base', 'CVML_18', 'CVML_34'])
args = parser.parse_args()

parent_dir = os.getcwd()

if args.model == 'CVML_base':
    network = CVML_base()
    model_path = parent_dir + '/trained_model/CVML_base/VAE/'

elif args.model == 'CVML_18':
    network = CVML_18()
    model_path = parent_dir + '/trained_model/CVML_18/VAE/'

elif args.model == 'CVML_34':
    network = CVML_34()
    model_path = parent_dir + '/trained_model/CVML_34/VAE/'

if os.path.exists(model_path) is False:
    os.makedirs(model_path)

# %% train CVML-AE on each object
object_number = args.object
ob_id = '%06i' % (object_number)
model_name = ob_id + '_vae.pth'

# online augmentation
aug_transform = nn.Sequential(
    transforms.RandomApply(
        transforms=[
            transforms.ColorJitter(brightness=[0.4, 2.3], contrast=[0.4, 2.3],
                                   saturation=[0.8, 1.2], hue=0)],
        p=0.3),

    transforms.RandomApply(
        transforms=[
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.2))],
        p=0.3),

    transforms.RandomApply(
        transforms=[
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1))],
        p=0.3))

PBR_image_path = parent_dir + '/processed data/pbr_train/' + ob_id
Recon_image_path = parent_dir + '/processed data/recon_train/' + ob_id

PBR_image = sorted(glob.glob(PBR_image_path + "/**/*.png", recursive=True))
Recon_image = sorted(glob.glob(Recon_image_path + "/**/*.png", recursive=True))
Train_dataset = Merge_Data(PBR_image, Recon_image)

# data split
a = int(len(PBR_image)*0.9)
b = len(PBR_image) - a
lengths = [a, b]

train, valid = random_split(Train_dataset,
                            lengths,
                            generator=torch.Generator().manual_seed(0))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

train_dataloader = DataLoader(
    train,
    batch_size=128,
    num_workers=multiprocessing.Pool()._processes,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
    shuffle=True
)

valid_dataloader = DataLoader(
    valid,
    batch_size=128,
    num_workers=multiprocessing.Pool()._processes,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
    shuffle=True
)

device = torch.device("cuda:0")
num_epochs = 1000
learning_rate = 1e-4
optimizer = torch.optim.AdamW(params=network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, threshold=0.0001,
    threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)

min_valid_loss = np.inf
network = network.to(device)

for epoch in range(num_epochs):

    train_loss = 0.0
    network.train()
    for input_image, target_image in train_dataloader:
        optimizer.zero_grad()
        input_image = aug_transform(input_image)
        input_image = input_image.to(device)
        target_image = target_image.to(device)

        x_recon, latent_mu, latent_logvar = network(input_image)
        loss = vae_loss(x_recon, target_image, latent_mu, latent_logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    network.eval()
    with torch.no_grad():
        for inputs, targets in valid_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            x_recon, _, _ = network(inputs)
            loss = F.mse_loss(x_recon.view(-1, 128*128),
                              targets.view(-1, 128*128), reduction='sum')
            valid_loss += loss.item()

    scheduler.step(valid_loss)
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(valid_dataloader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        epochs_no_improve = 0
        early_stop = False
        torch.save(network.state_dict(), model_path + model_name)
    elif (min_valid_loss < valid_loss and
          optimizer.param_groups[0]['lr'] == 1e-6):
        epochs_no_improve += 1

    if epochs_no_improve == 30:
        print('Finish Training')
        break
