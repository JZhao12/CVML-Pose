#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:29:05 2022

@author: jianyu

This script is used to train the mlp to predict 2D projection of the object
centre, and the knn to predict projective distance, finally calculate the
translation vector.

"""

# %% import modules
import argparse
import multiprocessing
import torch
import os
import torch.nn as nn
import numpy as np
import joblib
import json
import random
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from model import T_MLP

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# %% path, argparse, data information
parent_dir = os.getcwd()

parser = argparse.ArgumentParser(description='train translation mlp')
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

LMO_objects = [1, 5, 6, 8, 9, 10, 11, 12]

ob_id = '%06i' % (object_number)

mlp_path = latent_path + '/trained_mlp/'
if os.path.exists(mlp_path) is False:
    os.makedirs(mlp_path)

# %% train translation MLP
lm_latent_path = latent_path + 'lm_latent/'
lmo_latent_path = latent_path + 'lmo_latent/'

lm_latent_train = lm_latent_path + ob_id + '_train.npz'
lm_latent_valid = lm_latent_path + ob_id + '_valid.npz'

with np.load(lm_latent_train) as X:
    lm_train_latent_mean = X['lm_train_latent_mean']
    lm_train_TV = X['lm_train_TV']
    lm_train_scale = X['lm_train_scale']
    lm_train_bb = X['lm_train_bb']
    lm_train_ObjectCx = X['lm_train_ObjectCx']
    lm_train_ObjectCy = X['lm_train_ObjectCy']

with np.load(lm_latent_valid) as X:
    lm_valid_latent_mean = X['lm_valid_latent_mean']
    lm_valid_TV = X['lm_valid_TV']
    lm_valid_scale = X['lm_valid_scale']
    lm_valid_bb = X['lm_valid_bb']
    lm_valid_ObjectCx = X['lm_valid_ObjectCx']
    lm_valid_ObjectCy = X['lm_valid_ObjectCy']

Bx = np.expand_dims(lm_train_bb[:, 0], axis=1)
By = np.expand_dims(lm_train_bb[:, 1], axis=1)
Train_width = np.expand_dims(lm_train_bb[:, 2], axis=1)
Train_height = np.expand_dims(lm_train_bb[:, 3], axis=1)

XTrain = np.concatenate((lm_train_latent_mean, Bx, By, Train_width,
                         Train_height, lm_train_scale), axis=1)
YTrain = np.concatenate((lm_train_ObjectCx, lm_train_ObjectCy), axis=1)

XTrain = torch.tensor(XTrain)
YTrain = torch.tensor(YTrain, dtype=torch.float)

validBx = np.expand_dims(lm_valid_bb[:, 0], axis=1)
validBy = np.expand_dims(lm_valid_bb[:, 1], axis=1)
valid_width = np.expand_dims(lm_valid_bb[:, 2], axis=1)
valid_height = np.expand_dims(lm_valid_bb[:, 3], axis=1)

XValid = np.concatenate((lm_valid_latent_mean, validBx, validBy, valid_width,
                         valid_height, lm_valid_scale), axis=1)
YValid = np.concatenate((lm_valid_ObjectCx, lm_valid_ObjectCy), axis=1)

XValid = torch.tensor(XValid)
YValid = torch.tensor(YValid, dtype=torch.float)

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
mlp = T_MLP()
mlp = mlp.to(device)

learning_rate = 1e-3
num_epochs = 100000

loss_func = nn.L1Loss()
optimizer = torch.optim.AdamW(params=mlp.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000, threshold=0.0001,
    threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)
min_valid_loss = np.inf

for epoch in range(num_epochs):

    train_loss = 0.0
    mlp.train()

    for inputs, targets in mlp_train_data:

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = mlp(inputs)

        loss = loss_func(output, targets)
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

            loss = loss_func(output, targets)
            valid_loss += loss.item()

    scheduler.step(valid_loss)
    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(mlp_train_data)} \t\t Validation Loss: {valid_loss / len(mlp_valid_data)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        epochs_no_improve = 0
        torch.save(mlp.state_dict(), mlp_path + ob_id + '_centre.pth')
    elif (min_valid_loss < valid_loss and
          optimizer.param_groups[0]['lr'] == 1e-7):
        epochs_no_improve += 1

    if epochs_no_improve == 1000:
        print('Finish Training')
        break

# %% train KNN

# Camera
sym_camera = json.load(open(
    parent_dir +
    '/original data/lm/train_pbr/000000/scene_camera.json'))
sym_Camera_IM = np.reshape(sym_camera['0']['cam_K'], (3, 3))

"""
fx,fy,cx,cy = sym_Camera_IM[0,0],sym_Camera_IM[1,1],
sym_Camera_IM[0,2],sym_Camera_IM[1,2]
"""

Train_width = np.expand_dims(lm_train_bb[:, 2], axis=1)
Train_height = np.expand_dims(lm_train_bb[:, 3], axis=1)

XTrain = np.concatenate((lm_train_latent_mean, Train_width, Train_height,
                         lm_train_scale), axis=1)
YTrain = np.expand_dims(lm_train_TV[:, 2], axis=1)

XValid = np.concatenate((lm_valid_latent_mean, valid_width, valid_height,
                         lm_valid_scale), axis=1)

valid_err = []
for neigh_number in range(1, 20):
    TX = []
    TY = []
    TZ = []
    neigh = KNeighborsRegressor(n_neighbors=neigh_number, weights='distance')
    neigh.fit(XTrain, YTrain)
    PredictedTz_Valid = neigh.predict(XValid)
    for i in range((len(PredictedTz_Valid))):
        PredictTz = PredictedTz_Valid[i]
        OCx = lm_valid_ObjectCx[i]
        OCy = lm_valid_ObjectCy[i]
        PredictTx = (OCx-sym_Camera_IM[0, 2])*PredictTz/sym_Camera_IM[0, 0]
        PredictTy = (OCy-sym_Camera_IM[1, 2])*PredictTz/sym_Camera_IM[1, 1]

        TX.append(PredictTx)
        TY.append(PredictTy)
        TZ.append(PredictTz)

    TX = np.array(TX, dtype='float32')
    TY = np.array(TY, dtype='float32')
    TZ = np.array(TZ, dtype='float32')
    PredictedTV = np.concatenate((TX, TY, TZ), axis=1)
    err = mean_absolute_error(lm_valid_TV, PredictedTV)
    valid_err.append(err)

final_k = valid_err.index(min(valid_err)) + 1

Final_neigh = KNeighborsRegressor(n_neighbors=final_k, weights='distance')
Final_neigh.fit(XTrain, YTrain)

knn_path = latent_path + '/trained_knn/'
if os.path.exists(knn_path) is False:
    os.makedirs(knn_path)

knn_name = knn_path + ob_id + '_knn.sav'
joblib.dump(Final_neigh, knn_name)

# %% lm test gt

# Camera
test_camera = json.load(open(
    parent_dir + '/original data/lm/test/' +
    ob_id + '/scene_camera.json'))
test_Camera_IM = np.reshape(test_camera['0']['cam_K'], (3, 3))

"""
fx,fy,cx,cy = test_Camera_IM[0,0],test_Camera_IM[1,1],
test_Camera_IM[0,2],test_Camera_IM[1,2]
"""

Camera_ratio = np.sqrt(
    test_Camera_IM[0, 0]**2 +
    test_Camera_IM[1, 1]**2)/np.sqrt(sym_Camera_IM[0, 0]**2 +
                                     sym_Camera_IM[1, 1]**2)

mlp.load_state_dict(torch.load(mlp_path + ob_id + '_centre.pth'))

lm_latent_test_gt = lm_latent_path + ob_id + '_test_gt.npz'
with np.load(lm_latent_test_gt) as X:
    lm_test_gt_latent_mean = X['lm_test_gt_latent_mean']
    lm_test_gt_scale = X['lm_test_gt_scale']
    lm_test_gt_bbox = X['lm_test_gt_bbox']

TestBx = np.expand_dims(lm_test_gt_bbox[:, 0], axis=1)
TestBy = np.expand_dims(lm_test_gt_bbox[:, 1], axis=1)
Test_width = np.expand_dims(lm_test_gt_bbox[:, 2], axis=1)
Test_height = np.expand_dims(lm_test_gt_bbox[:, 3], axis=1)

XTest = np.concatenate((lm_test_gt_latent_mean, TestBx, TestBy, Test_width,
                        Test_height, lm_test_gt_scale), axis=1)
XTest = torch.tensor(XTest, dtype=torch.float)
Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

Predict_Test_center = []
mlp.eval()

for inputs in Test_data:

    with torch.no_grad():
        inputs = inputs.to(device)
        output = mlp(inputs)
        Predict_Test_center.append(output)

Predict_Test_center = torch.cat(Predict_Test_center)
Predict_Test_center = Predict_Test_center.cpu().numpy()
Predict_Test_OCX = Predict_Test_center[:, 0]
Predict_Test_OCY = Predict_Test_center[:, 1]

result_path = latent_path + '/lm_results_gt/'
if os.path.exists(result_path) is False:
    os.makedirs(result_path)

Object_centre_result = result_path + ob_id + '_centre'
np.savez(Object_centre_result,
         Predict_Test_OCX=Predict_Test_OCX,
         Predict_Test_OCY=Predict_Test_OCY)

Final_neigh = joblib.load(knn_name)

XTest = np.concatenate((lm_test_gt_latent_mean, Test_width, Test_height,
                        lm_test_gt_scale), axis=1)
PredictedTz_Candidate = Final_neigh.predict(XTest)
PredictedTz_Candidate = PredictedTz_Candidate*Camera_ratio

TX = []
TY = []
TZ = []
for i in range((len(PredictedTz_Candidate))):
    PredictTz = PredictedTz_Candidate[i]
    OCx = Predict_Test_OCX[i]
    OCy = Predict_Test_OCY[i]
    PredictTx = (OCx-test_Camera_IM[0, 2])*PredictTz/test_Camera_IM[0, 0]
    PredictTy = (OCy-test_Camera_IM[1, 2])*PredictTz/test_Camera_IM[1, 1]

    TX.append(PredictTx)
    TY.append(PredictTy)
    TZ.append(PredictTz)

TX = np.array(TX, dtype='float32')
TY = np.array(TY, dtype='float32')
TZ = np.array(TZ, dtype='float32')
PredictedTV = np.concatenate((TX, TY, TZ), axis=1)

Translation_result = result_path + ob_id + '_translation'
np.savez(Translation_result, PredictedTV=PredictedTV)

# %% lm test maskrcnn

mlp.load_state_dict(torch.load(mlp_path + ob_id + '_centre.pth'))

lm_latent_test_maskrcnn = lm_latent_path + ob_id + '_test_maskrcnn.npz'
with np.load(lm_latent_test_maskrcnn) as X:
    lm_test_maskrcnn_latent_mean = X['lm_test_maskrcnn_latent_mean']
    lm_test_maskrcnn_scale = X['lm_test_maskrcnn_scale']
    lm_test_maskrcnn_bbox = X['lm_test_maskrcnn_bbox']

TestBx = np.expand_dims(lm_test_maskrcnn_bbox[:, 0], axis=1)
TestBy = np.expand_dims(lm_test_maskrcnn_bbox[:, 1], axis=1)
Test_width = np.expand_dims(lm_test_maskrcnn_bbox[:, 2], axis=1)
Test_height = np.expand_dims(lm_test_maskrcnn_bbox[:, 3], axis=1)

XTest = np.concatenate(
    (lm_test_maskrcnn_latent_mean, TestBx, TestBy, Test_width,
     Test_height, lm_test_maskrcnn_scale), axis=1)
XTest = torch.tensor(XTest, dtype=torch.float)
Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

Predict_Test_center = []
mlp.eval()

for inputs in Test_data:

    with torch.no_grad():
        inputs = inputs.to(device)
        output = mlp(inputs)
        Predict_Test_center.append(output)

Predict_Test_center = torch.cat(Predict_Test_center)
Predict_Test_center = Predict_Test_center.cpu().numpy()
Predict_Test_OCX = Predict_Test_center[:, 0]
Predict_Test_OCY = Predict_Test_center[:, 1]

result_path = latent_path + '/lm_results_maskrcnn/'
if os.path.exists(result_path) is False:
    os.makedirs(result_path)

Object_centre_result = result_path + ob_id + '_centre'
np.savez(Object_centre_result,
         Predict_Test_OCX=Predict_Test_OCX,
         Predict_Test_OCY=Predict_Test_OCY)

Final_neigh = joblib.load(knn_name)
XTest = np.concatenate(
    (lm_test_maskrcnn_latent_mean, Test_width, Test_height,
     lm_test_maskrcnn_scale), axis=1)
PredictedTz_Candidate = Final_neigh.predict(XTest)
PredictedTz_Candidate = PredictedTz_Candidate*Camera_ratio

TX = []
TY = []
TZ = []
for i in range((len(PredictedTz_Candidate))):
    PredictTz = PredictedTz_Candidate[i]
    OCx = Predict_Test_OCX[i]
    OCy = Predict_Test_OCY[i]
    PredictTx = (OCx-test_Camera_IM[0, 2])*PredictTz/test_Camera_IM[0, 0]
    PredictTy = (OCy-test_Camera_IM[1, 2])*PredictTz/test_Camera_IM[1, 1]

    TX.append(PredictTx)
    TY.append(PredictTy)
    TZ.append(PredictTz)

TX = np.array(TX, dtype='float32')
TY = np.array(TY, dtype='float32')
TZ = np.array(TZ, dtype='float32')
PredictedTV = np.concatenate((TX, TY, TZ), axis=1)

Translation_result = result_path + ob_id + '_translation'
np.savez(Translation_result, PredictedTV=PredictedTV)

# %% lmo test gt

if object_number in LMO_objects:

    mlp.load_state_dict(torch.load(mlp_path + ob_id + '_centre.pth'))

    lmo_latent_test_gt = lmo_latent_path + ob_id + '_test_gt.npz'
    with np.load(lmo_latent_test_gt) as X:
        lmo_test_gt_latent_mean = X['lmo_test_gt_latent_mean']
        lmo_test_gt_scale = X['lmo_test_gt_scale']
        lmo_test_gt_bbox = X['lmo_test_gt_bbox']

    TestBx = np.expand_dims(lmo_test_gt_bbox[:, 0], axis=1)
    TestBy = np.expand_dims(lmo_test_gt_bbox[:, 1], axis=1)
    Test_width = np.expand_dims(lmo_test_gt_bbox[:, 2], axis=1)
    Test_height = np.expand_dims(lmo_test_gt_bbox[:, 3], axis=1)

    XTest = np.concatenate(
        (lmo_test_gt_latent_mean, TestBx, TestBy, Test_width,
         Test_height, lmo_test_gt_scale), axis=1)
    XTest = torch.tensor(XTest, dtype=torch.float)
    Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

    Predict_Test_center = []
    mlp.eval()

    for inputs in Test_data:

        with torch.no_grad():
            inputs = inputs.to(device)
            output = mlp(inputs)
            Predict_Test_center.append(output)

    Predict_Test_center = torch.cat(Predict_Test_center)
    Predict_Test_center = Predict_Test_center.cpu().numpy()
    Predict_Test_OCX = Predict_Test_center[:, 0]
    Predict_Test_OCY = Predict_Test_center[:, 1]

    result_path = latent_path + '/lmo_results_gt/'
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    Object_centre_result = result_path + ob_id + '_centre'
    np.savez(Object_centre_result,
             Predict_Test_OCX=Predict_Test_OCX,
             Predict_Test_OCY=Predict_Test_OCY)

    Final_neigh = joblib.load(knn_name)

    XTest = np.concatenate(
        (lmo_test_gt_latent_mean, Test_width, Test_height,
         lmo_test_gt_scale), axis=1)
    PredictedTz_Candidate = Final_neigh.predict(XTest)
    PredictedTz_Candidate = PredictedTz_Candidate*Camera_ratio

    TX = []
    TY = []
    TZ = []
    for i in range((len(PredictedTz_Candidate))):
        PredictTz = PredictedTz_Candidate[i]
        OCx = Predict_Test_OCX[i]
        OCy = Predict_Test_OCY[i]
        PredictTx = (OCx-test_Camera_IM[0, 2])*PredictTz/test_Camera_IM[0, 0]
        PredictTy = (OCy-test_Camera_IM[1, 2])*PredictTz/test_Camera_IM[1, 1]

        TX.append(PredictTx)
        TY.append(PredictTy)
        TZ.append(PredictTz)

    TX = np.array(TX, dtype='float32')
    TY = np.array(TY, dtype='float32')
    TZ = np.array(TZ, dtype='float32')
    PredictedTV = np.concatenate((TX, TY, TZ), axis=1)

    Translation_result = result_path + ob_id + '_translation'
    np.savez(Translation_result, PredictedTV=PredictedTV)

# %% lmo test maskrcnn

if object_number in LMO_objects:

    mlp.load_state_dict(torch.load(mlp_path + ob_id + '_centre.pth'))

    lmo_latent_test_maskrcnn = lmo_latent_path + ob_id + '_test_maskrcnn.npz'
    with np.load(lmo_latent_test_maskrcnn) as X:
        lmo_test_maskrcnn_latent_mean = X['lmo_test_maskrcnn_latent_mean']
        lmo_test_maskrcnn_scale = X['lmo_test_maskrcnn_scale']
        lmo_test_maskrcnn_bbox = X['lmo_test_maskrcnn_bbox']

    TestBx = np.expand_dims(lmo_test_maskrcnn_bbox[:, 0], axis=1)
    TestBy = np.expand_dims(lmo_test_maskrcnn_bbox[:, 1], axis=1)
    Test_width = np.expand_dims(lmo_test_maskrcnn_bbox[:, 2], axis=1)
    Test_height = np.expand_dims(lmo_test_maskrcnn_bbox[:, 3], axis=1)

    XTest = np.concatenate(
        (lmo_test_maskrcnn_latent_mean, TestBx, TestBy, Test_width,
         Test_height, lmo_test_maskrcnn_scale), axis=1)
    XTest = torch.tensor(XTest, dtype=torch.float)
    Test_data = DataLoader(XTest, batch_size=len(XTest), shuffle=False)

    Predict_Test_center = []
    mlp.eval()

    for inputs in Test_data:

        with torch.no_grad():
            inputs = inputs.to(device)
            output = mlp(inputs)
            Predict_Test_center.append(output)

    Predict_Test_center = torch.cat(Predict_Test_center)
    Predict_Test_center = Predict_Test_center.cpu().numpy()
    Predict_Test_OCX = Predict_Test_center[:, 0]
    Predict_Test_OCY = Predict_Test_center[:, 1]

    result_path = latent_path + '/lmo_results_maskrcnn/'
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    Object_centre_result = result_path + ob_id + '_centre'
    np.savez(Object_centre_result,
             Predict_Test_OCX=Predict_Test_OCX,
             Predict_Test_OCY=Predict_Test_OCY)

    Final_neigh = joblib.load(knn_name)

    XTest = np.concatenate(
        (lmo_test_maskrcnn_latent_mean, Test_width, Test_height,
         lmo_test_maskrcnn_scale), axis=1)
    PredictedTz_Candidate = Final_neigh.predict(XTest)
    PredictedTz_Candidate = PredictedTz_Candidate*Camera_ratio

    TX = []
    TY = []
    TZ = []
    for i in range((len(PredictedTz_Candidate))):
        PredictTz = PredictedTz_Candidate[i]
        OCx = Predict_Test_OCX[i]
        OCy = Predict_Test_OCY[i]
        PredictTx = (OCx-test_Camera_IM[0, 2])*PredictTz/test_Camera_IM[0, 0]
        PredictTy = (OCy-test_Camera_IM[1, 2])*PredictTz/test_Camera_IM[1, 1]

        TX.append(PredictTx)
        TY.append(PredictTy)
        TZ.append(PredictTz)

    TX = np.array(TX, dtype='float32')
    TY = np.array(TY, dtype='float32')
    TZ = np.array(TZ, dtype='float32')
    PredictedTV = np.concatenate((TX, TY, TZ), axis=1)

    Translation_result = result_path + ob_id + '_translation'
    np.savez(Translation_result, PredictedTV=PredictedTV)