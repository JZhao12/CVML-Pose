#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:29:01 2022

@author: jianyu

This script is used to train the mlp for rotation estimation.

"""

# %% import modules
import torch
import os
import argparse
import multiprocessing
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from engine import rotation_6d_to_matrix
from model import R_MLP

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, argparse, data information
parent_dir = os.getcwd()

parser = argparse.ArgumentParser(description='train rotation mlp')
parser.add_argument('--object', type=int, help='which object')
parser.add_argument('--model', type=str, help='cvml models',
                    choices=['CVML_base', 'CVML_18', 'CVML_34'])
args = parser.parse_args()

if args.model == 'CVML_base':
    latent_path = parent_dir + '/trained_model/CVML_base/'

elif args.model == 'CVML_18':
    latent_path = parent_dir + '/trained_model/CVML_18/'

elif args.model == 'CVML_34':
    latent_path = parent_dir + '/trained_model/CVML_34/'

object_number = args.object

mlp_path = latent_path + '/trained_mlp/'
if os.path.exists(mlp_path) is False:
    os.makedirs(mlp_path)

LMO_objects = [1, 5, 6, 8, 9, 10, 11, 12]

ob_id = '%06i' % (object_number)

# %% train rotational MLP

lm_latent_path = latent_path + 'lm_latent/'
lmo_latent_path = latent_path + 'lmo_latent/'

lm_latent_train = lm_latent_path + ob_id + '_train.npz'
lm_latent_valid = lm_latent_path + ob_id + '_valid.npz'

with np.load(lm_latent_train) as X:
    lm_train_latent_mean = X['lm_train_latent_mean']
    lm_train_RM = X['lm_train_RM']

with np.load(lm_latent_valid) as X:
    lm_valid_latent_mean = X['lm_valid_latent_mean']
    lm_valid_RM = X['lm_valid_RM']

XTrain = torch.tensor(lm_train_latent_mean)
YTrain = torch.tensor(lm_train_RM, dtype=torch.float)

XValid = torch.tensor(lm_valid_latent_mean)
YValid = torch.tensor(lm_valid_RM, dtype=torch.float)

mlp_train_data = []
for i in range(len(XTrain)):
    mlp_train_data.append([XTrain[i], YTrain[i]])

mlp_valid_data = []
for i in range(len(XValid)):
    mlp_valid_data.append([XValid[i], YValid[i]])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

mlp_train_data = DataLoader(
    mlp_train_data,
    batch_size=len(mlp_train_data),
    num_workers=multiprocessing.Pool()._processes,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
    shuffle=True
)

mlp_valid_data = DataLoader(
    mlp_valid_data,
    batch_size=len(mlp_valid_data),
    num_workers=multiprocessing.Pool()._processes,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True,
    shuffle=True
)

device = torch.device("cuda:0")
mlp = R_MLP()
mlp = mlp.to(device)

learning_rate = 1e-3
num_epochs = 100000

loss_func = nn.L1Loss()
optimizer = torch.optim.AdamW(params=mlp.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000,
    threshold=0.0001, threshold_mode='rel', cooldown=0,
    min_lr=1e-7, eps=1e-08, verbose=True)
min_valid_loss = np.inf

for epoch in range(num_epochs):

    train_loss = 0.0
    mlp.train()

    for inputs, targets in mlp_train_data:

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = mlp(inputs)
        outputRM = rotation_6d_to_matrix(output)

        loss = loss_func(outputRM, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    valid_loss = 0.0
    mlp.eval()
    with torch.no_grad():

        for inputs, targets in mlp_valid_data:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = mlp(inputs)
            outputRM = rotation_6d_to_matrix(output)
            loss = loss_func(outputRM, targets)
            valid_loss += loss.item()

    scheduler.step(valid_loss)
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(mlp_train_data)} \t\t Validation Loss: {valid_loss / len(mlp_valid_data)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        epochs_no_improve = 0
        torch.save(mlp.state_dict(), mlp_path + ob_id + '_rotation.pth')
    elif (min_valid_loss < valid_loss and
          optimizer.param_groups[0]['lr'] == 1e-7):
        epochs_no_improve += 1

    if epochs_no_improve == 1000:
        print('Finish Training')
        break

# %% lm test gt

result_path = latent_path + '/lm_results_gt/'
if os.path.exists(result_path) is False:
    os.makedirs(result_path)

mlp.load_state_dict(torch.load(mlp_path + ob_id + '_rotation.pth'))

lm_latent_test_gt = lm_latent_path + ob_id + '_test_gt.npz'
with np.load(lm_latent_test_gt) as X:
    lm_test_gt_latent_mean = X['lm_test_gt_latent_mean']

XTest = torch.tensor(lm_test_gt_latent_mean)
Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

mlp.eval()
Predicted = []

for inputs in Test_data:

    with torch.no_grad():
        inputs = inputs.to(device)
        output = mlp(inputs)
        Predicted.append(output)

Predicted_Candidate = torch.cat(Predicted)
PredictedRM = rotation_6d_to_matrix(Predicted_Candidate)
PredictedRM = PredictedRM.cpu().numpy()

Rotation_result = result_path + ob_id + '_rotation'
np.savez(Rotation_result, PredictedRM=PredictedRM)

# %% lm test maskrcnn

result_path = latent_path + '/lm_results_maskrcnn/'
if os.path.exists(result_path) is False:
    os.makedirs(result_path)

mlp.load_state_dict(torch.load(mlp_path + ob_id + '_rotation.pth'))

lm_latent_test_maskrcnn = lm_latent_path + ob_id + '_test_maskrcnn.npz'
with np.load(lm_latent_test_maskrcnn) as X:
    lm_test_maskrcnn_latent_mean = X['lm_test_maskrcnn_latent_mean']

XTest = torch.tensor(lm_test_maskrcnn_latent_mean)
Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

mlp.eval()
Predicted = []

for inputs in Test_data:

    with torch.no_grad():
        inputs = inputs.to(device)
        output = mlp(inputs)
        Predicted.append(output)

Predicted_Candidate = torch.cat(Predicted)
PredictedRM = rotation_6d_to_matrix(Predicted_Candidate)
PredictedRM = PredictedRM.cpu().numpy()

Rotation_result = result_path + ob_id + '_rotation'
np.savez(Rotation_result, PredictedRM=PredictedRM)

# %% lmo test gt

if object_number in LMO_objects:

    result_path = latent_path + '/lmo_results_gt/'
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    mlp.load_state_dict(torch.load(mlp_path + ob_id + '_rotation.pth'))

    lmo_latent_test_gt = lmo_latent_path + ob_id + '_test_gt.npz'
    with np.load(lmo_latent_test_gt) as X:
        lmo_test_gt_latent_mean = X['lmo_test_gt_latent_mean']

    XTest = torch.tensor(lmo_test_gt_latent_mean)
    Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

    mlp.eval()
    Predicted = []

    for inputs in Test_data:

        with torch.no_grad():
            inputs = inputs.to(device)
            output = mlp(inputs)
            Predicted.append(output)

    Predicted_Candidate = torch.cat(Predicted)
    PredictedRM = rotation_6d_to_matrix(Predicted_Candidate)
    PredictedRM = PredictedRM.cpu().numpy()

    Rotation_result = result_path + ob_id + '_rotation'
    np.savez(Rotation_result, PredictedRM=PredictedRM)

# %% lmo test maskrcnn

if object_number in LMO_objects:

    result_path = latent_path + '/lmo_results_maskrcnn/'
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    mlp.load_state_dict(torch.load(mlp_path + ob_id + '_rotation.pth'))

    lmo_latent_test_maskrcnn = lmo_latent_path + ob_id + '_test_maskrcnn.npz'
    with np.load(lmo_latent_test_maskrcnn) as X:
        lmo_test_maskrcnn_latent_mean = X['lmo_test_maskrcnn_latent_mean']

    XTest = torch.tensor(lmo_test_maskrcnn_latent_mean)
    Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

    mlp.eval()
    Predicted = []

    for inputs in Test_data:

        with torch.no_grad():
            inputs = inputs.to(device)
            output = mlp(inputs)
            Predicted.append(output)

    Predicted_Candidate = torch.cat(Predicted)
    PredictedRM = rotation_6d_to_matrix(Predicted_Candidate)
    PredictedRM = PredictedRM.cpu().numpy()

    Rotation_result = result_path + ob_id + '_rotation'
    np.savez(Rotation_result, PredictedRM=PredictedRM)