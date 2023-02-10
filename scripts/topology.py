#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 23:29:16 2022

@author: jianyu

This script is used to train vae on multiple objects, generate the latent
variables based on the test images, and visualize the variables using t-sne.

"""

# %% import modules
import os
import torch
import numpy as np
import random
import glob
import multiprocessing
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from engine import Generate, Merge_Data, vae_loss
from model import CVML_base
from numpy.random import RandomState
from sklearn import manifold

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, data information
parent_dir = os.getcwd()
model_path = parent_dir + '/trained_model/topology/VAE/'
if os.path.exists(model_path) is False:
    os.makedirs(model_path)
PBR_image_path = parent_dir + '/processed data/pbr_train/'
Recon_image_path = parent_dir + '/processed data/recon_train/'

# %% train CVML-base on four objects
device = torch.device("cuda:0")
num_epochs = 1000
learning_rate = 1e-4

model_name = 'topology_and_class_vae.pth'

object_number = [1, 5, 9, 13]
All_train = []
All_valid = []

for number in object_number:
    ob_id = '%06i' % (number)
    Train_image = sorted(glob.glob(PBR_image_path + ob_id +
                                   "/**/*.png", recursive=True))
    GT_image = sorted(glob.glob(Recon_image_path + ob_id +
                                "/**/*.png", recursive=True))
    PBR_dataset = Merge_Data(Train_image, GT_image)
    a = int(len(PBR_dataset)*0.9)
    b = len(PBR_dataset) - a
    lengths = [a, b]
    train, valid = random_split(PBR_dataset,
                                lengths,
                                generator=torch.Generator().manual_seed(0))
    All_train.append(train)
    All_valid.append(valid)

All_train = ConcatDataset(All_train)
All_valid = ConcatDataset(All_valid)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

train_dataloader = DataLoader(
    All_train,
    batch_size=128,
    num_workers=multiprocessing.Pool()._processes,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
    shuffle=True
)

valid_dataloader = DataLoader(
    All_valid,
    batch_size=128,
    num_workers=multiprocessing.Pool()._processes,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
    shuffle=True
)

network = CVML_base()
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
        torch.save(network.state_dict(), model_path + model_name)
    elif (min_valid_loss < valid_loss and
          optimizer.param_groups[0]['lr'] == 1e-6):
        epochs_no_improve += 1

    if epochs_no_improve == 30:
        print('Finish Training')
        break

# %% get latent variables
lm_test_latent_path = parent_dir + '/trained_model/topology/lm_test_latent/'
if os.path.exists(lm_test_latent_path) is False:
    os.makedirs(lm_test_latent_path)

network.load_state_dict(torch.load(model_path + model_name))

for obj in object_number:

    ob_id = '%06i' % (obj)
    lm_test_gt_datafold = datasets.ImageFolder(
        parent_dir + '/processed data/lm_test_gt/' + ob_id,
        transform=transforms.ToTensor())
    lm_test_gt_dataloader = DataLoader(
        lm_test_gt_datafold, batch_size=128, pin_memory=True, shuffle=False,
        num_workers=multiprocessing.Pool()._processes)

    lm_test_gt_latent_mean = Generate(network, lm_test_gt_dataloader, device)
    lm_test_gt_latent_mean = lm_test_gt_latent_mean.cpu().numpy()

    lm_latent_test = lm_test_latent_path + ob_id + '_test'

    np.savez(lm_latent_test, lm_test_gt_latent_mean=lm_test_gt_latent_mean)

obj_1, obj_5, obj_9, obj_13 = object_number

ob_id_1 = '%06i' % (obj_1)
ob_id_5 = '%06i' % (obj_5)
ob_id_9 = '%06i' % (obj_9)
ob_id_13 = '%06i' % (obj_13)

X1 = np.load(lm_test_latent_path + ob_id_1 + '_test.npz')
Test1 = X1['lm_test_gt_latent_mean']
X5 = np.load(lm_test_latent_path + ob_id_5 + '_test.npz')
Test5 = X5['lm_test_gt_latent_mean']
X9 = np.load(lm_test_latent_path + ob_id_9 + '_test.npz')
Test9 = X9['lm_test_gt_latent_mean']
X13 = np.load(lm_test_latent_path + ob_id_13 + '_test.npz')
Test13 = X13['lm_test_gt_latent_mean']

# %% t-sne for classification
s1 = np.zeros(len(Test1))
s5 = np.ones(len(Test5))*3
s9 = np.ones(len(Test9))
s13 = np.ones(len(Test13))*2

S_color = np.concatenate((s1, s9, s13, s5), axis=0)
S_points = np.concatenate((Test1, Test9, Test13, Test5), axis=0)

t_sne = manifold.TSNE(
    n_components=2,
    learning_rate="auto",
    perplexity=10,
    n_iter=2000,
    init="random",
    random_state=RandomState(0),
)

S_t_sne = t_sne.fit_transform(S_points)
x, y = S_t_sne.T
categories = S_color
colormap = np.array(['#1f77b4', '#89bedc',  '#2ca02c', '#ff7f0e'])
categories = categories.astype(int)

plt.figure(figsize=(5, 5), dpi=500)
plt.scatter(x, y, s=5, c=colormap[categories])

pop_a = mpatches.Patch(color='#1f77b4', label='ape')
pop_b = mpatches.Patch(color='#89bedc', label='duck')
pop_c = mpatches.Patch(color='#2ca02c', label='iron')
pop_d = mpatches.Patch(color='#ff7f0e', label='can')

plt.legend(handles=[pop_a, pop_b, pop_c, pop_d])
plt.savefig(parent_dir + '/trained_model/topology/tsne_class.png')

# %% # %% t-sne for genus
rng = RandomState(0)

s1 = np.zeros(len(Test1))
s5 = np.ones(len(Test5))*2
s9 = np.zeros(len(Test9))
s13 = np.ones(len(Test13))

S_color = np.concatenate((s1, s9, s13, s5), axis=0)
S_points = np.concatenate((Test1, Test9, Test13, Test5), axis=0)

rng = RandomState(0)
t_sne = manifold.TSNE(
    n_components=2,
    learning_rate="auto",
    perplexity=10,
    n_iter=2000,
    init="random",
    random_state=rng,
)

S_t_sne = t_sne.fit_transform(S_points)

x, y = S_t_sne.T

categories = S_color

colormap = np.array(['#1f77b4', '#2ca02c',  '#ff7f0e'])

categories = categories.astype(int)

plt.figure(figsize=(5, 5), dpi=500)
plt.scatter(x, y, s=5, c=colormap[categories])

pop_a = mpatches.Patch(color='#1f77b4', label='genus-0')
pop_b = mpatches.Patch(color='#2ca02c', label='genus-1')
pop_c = mpatches.Patch(color='#ff7f0e', label='genus-2')

plt.legend(handles=[pop_a, pop_b, pop_c])
plt.savefig(parent_dir + '/trained_model/topology/tsne_genus.png')