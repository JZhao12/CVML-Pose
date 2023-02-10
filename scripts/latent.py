# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:07:51 2022

@author: Eddie

This script is used to generate the latent variables from the trained vae.

"""

# %% import modules
import torch
import os
import numpy as np
import json
import argparse
import multiprocessing
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import CVML_base, CVML_18, CVML_34
from engine import Generate

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %% path, argparse, data information
parser = argparse.ArgumentParser(description='get latent variables')
parser.add_argument('--object', type=int, help='object latent variables')
parser.add_argument('--model', type=str, help='cvml models',
                    choices=['CVML_base', 'CVML_18', 'CVML_34'])
args = parser.parse_args()

parent_dir = os.getcwd()

if args.model == 'CVML_base':
    network = CVML_base()
    model_path = parent_dir + '/trained_model/CVML_base/VAE/'
    latent_path = parent_dir + '/trained_model/CVML_base/'
elif args.model == 'CVML_18':
    network = CVML_18()
    model_path = parent_dir + '/trained_model/CVML_18/VAE/'
    latent_path = parent_dir + '/trained_model/CVML_18/'
elif args.model == 'CVML_34':
    network = CVML_34()
    model_path = parent_dir + '/trained_model/CVML_34/VAE/'
    latent_path = parent_dir + '/trained_model/CVML_34/'

object_number = args.object
LMO_objects = [1, 5, 6, 8, 9, 10, 11, 12]

lm_latent_path = latent_path + 'lm_latent/'
lmo_latent_path = latent_path + 'lmo_latent/'

if os.path.exists(lm_latent_path) is False:
    os.makedirs(lm_latent_path)
if os.path.exists(lmo_latent_path) is False:
    os.makedirs(lmo_latent_path)

# %% generate latent variables
ob_id = '%06i' % (object_number)
model_name = ob_id + '_vae.pth'
device = torch.device("cuda:0")
network.load_state_dict(torch.load(model_path + model_name,
                                   map_location='cuda:0'))
network = network.to(device)

pbr_train_datafold = datasets.ImageFolder(
    parent_dir + '/processed data/pbr_train/' + ob_id,
    transform=transforms.ToTensor())
pbr_loader = DataLoader(pbr_train_datafold, batch_size=128,
                        pin_memory=True, shuffle=False,
                        num_workers=multiprocessing.Pool()._processes)

pbr_train_latent_mean = Generate(network, pbr_loader, device)
pbr_train_latent_mean = pbr_train_latent_mean.cpu().numpy()

pbr_train_gt_json = json.load(open(
    parent_dir + '/processed data/pbr_train/' + ob_id + '_all.json'))

sym_camera_json = json.load(open(
    parent_dir +
    '/original data/lm/train_pbr/000000/scene_camera.json'))
sym_camera_matrix = np.reshape(sym_camera_json['0']['cam_K'], (3, 3))

pbr_train_RM = []
pbr_train_TV = []
pbr_train_scale = []
pbr_train_bb = []
pbr_train_ObjectCx = []
pbr_train_ObjectCy = []

for column in range(len(pbr_train_gt_json)):
    RM = pbr_train_gt_json[column]['RM']
    RM = np.reshape(RM, (3, 3))
    TV = pbr_train_gt_json[column]['TV']
    bbox = pbr_train_gt_json[column]['bbox']
    scale = 128/max(bbox[2], bbox[3])
    ObjectCx = TV[0]*sym_camera_matrix[0, 0]/TV[2]+sym_camera_matrix[0, 2]
    ObjectCy = TV[1]*sym_camera_matrix[1, 1]/TV[2]+sym_camera_matrix[1, 2]

    pbr_train_RM.append(RM)
    pbr_train_TV.append(TV)
    pbr_train_scale.append(scale)
    pbr_train_bb.append(bbox)
    pbr_train_ObjectCx.append(ObjectCx)
    pbr_train_ObjectCy.append(ObjectCy)

pbr_train_RM = np.array(pbr_train_RM, dtype='float32')
pbr_train_TV = np.array(pbr_train_TV, dtype='float32')
pbr_train_scale = np.expand_dims(np.array(pbr_train_scale,
                                          dtype='float32'), axis=1)
pbr_train_bb = np.array(pbr_train_bb, dtype='float32')
pbr_train_ObjectCx = np.expand_dims(np.array(pbr_train_ObjectCx,
                                             dtype='float32'), axis=1)
pbr_train_ObjectCy = np.expand_dims(np.array(pbr_train_ObjectCy,
                                             dtype='float32'), axis=1)

# data split
a = int(len(pbr_train_datafold)*0.9)
b = len(pbr_train_datafold) - a
train, valid = random_split(pbr_train_datafold, [a, b],
                            generator=torch.Generator().manual_seed(0))

# train/validation latent variables
train_idx = train.indices
lm_train_latent_mean = pbr_train_latent_mean[train_idx, :]
lm_train_RM = pbr_train_RM[train_idx, :, :]
lm_train_TV = pbr_train_TV[train_idx, :]
lm_train_scale = pbr_train_scale[train_idx, :]
lm_train_bb = pbr_train_bb[train_idx, :]
lm_train_ObjectCx = pbr_train_ObjectCx[train_idx, :]
lm_train_ObjectCy = pbr_train_ObjectCy[train_idx, :]

valid_idx = valid.indices
lm_valid_latent_mean = pbr_train_latent_mean[valid_idx, :]
lm_valid_RM = pbr_train_RM[valid_idx, :, :]
lm_valid_TV = pbr_train_TV[valid_idx, :]
lm_valid_scale = pbr_train_scale[valid_idx, :]
lm_valid_bb = pbr_train_bb[valid_idx, :]
lm_valid_ObjectCx = pbr_train_ObjectCx[valid_idx, :]
lm_valid_ObjectCy = pbr_train_ObjectCy[valid_idx, :]

# ground truth Linemod test latent variables
lm_test_gt_datafold = datasets.ImageFolder(
    parent_dir + '/processed data/lm_test_gt/' + ob_id,
    transform=transforms.ToTensor())
lm_test_gt_loader = DataLoader(lm_test_gt_datafold, batch_size=128,
                               pin_memory=True, shuffle=False,
                               num_workers=multiprocessing.Pool()._processes)

lm_test_gt_latent_mean = Generate(network, lm_test_gt_loader, device)
lm_test_gt_latent_mean = lm_test_gt_latent_mean.cpu().numpy()

lm_test_gt_json = json.load(open(
    parent_dir + '/processed data/lm_test_gt/' + ob_id + '.json'))

lm_test_gt_RM = []
lm_test_gt_TV = []
lm_test_gt_scale = []
lm_test_gt_bbox = []

for column in range(len(lm_test_gt_json)):
    RM = lm_test_gt_json[column]['RM']
    RM = np.reshape(RM, (3, 3))
    TV = lm_test_gt_json[column]['TV']
    bbox = lm_test_gt_json[column]['bbox']
    scale = lm_test_gt_json[column]['scale']

    lm_test_gt_RM.append(RM)
    lm_test_gt_TV.append(TV)
    lm_test_gt_scale.append(scale)
    lm_test_gt_bbox.append(bbox)

lm_test_gt_RM = np.array(lm_test_gt_RM, dtype='float32')
lm_test_gt_TV = np.array(lm_test_gt_TV, dtype='float32')
lm_test_gt_scale = np.expand_dims(np.array(lm_test_gt_scale, dtype='float32'),
                                  axis=1)
lm_test_gt_bbox = np.array(lm_test_gt_bbox, dtype='float32')

# ground truth Linemod-occlusion test latent variables
if object_number in LMO_objects:

    lmo_test_gt_fold = datasets.ImageFolder(
        parent_dir + '/processed data/lmo_test_bop19_gt/' + ob_id,
        transform=transforms.ToTensor())
    lmo_test_gt_loader = DataLoader(
        lmo_test_gt_fold, batch_size=32, pin_memory=True, shuffle=False,
        num_workers=multiprocessing.Pool()._processes)

    lmo_test_gt_latent_mean = Generate(network, lmo_test_gt_loader, device)
    lmo_test_gt_latent_mean = lmo_test_gt_latent_mean.cpu().numpy()

    lmo_test_gt_json = json.load(open(
        parent_dir + '/processed data/lmo_test_bop19_gt/' + ob_id + '.json'))

    lmo_test_gt_RM = []
    lmo_test_gt_TV = []
    lmo_test_gt_scale = []
    lmo_test_gt_bbox = []

    for column in range(len(lmo_test_gt_json)):
        RM = lmo_test_gt_json[column]['RM']
        RM = np.reshape(RM, (3, 3))
        TV = lmo_test_gt_json[column]['TV']
        bbox = lmo_test_gt_json[column]['bbox']
        scale = lmo_test_gt_json[column]['scale']

        lmo_test_gt_RM.append(RM)
        lmo_test_gt_TV.append(TV)
        lmo_test_gt_scale.append(scale)
        lmo_test_gt_bbox.append(bbox)

    lmo_test_gt_RM = np.array(lmo_test_gt_RM, dtype='float32')
    lmo_test_gt_TV = np.array(lmo_test_gt_TV, dtype='float32')
    lmo_test_gt_scale = np.expand_dims(np.array(lmo_test_gt_scale,
                                                dtype='float32'), axis=1)
    lmo_test_gt_bbox = np.array(lmo_test_gt_bbox, dtype='float32')
    lmo_latent_test_gt = lmo_latent_path + ob_id + '_test_gt'

    np.savez(lmo_latent_test_gt,
             lmo_test_gt_latent_mean=lmo_test_gt_latent_mean,
             lmo_test_gt_RM=lmo_test_gt_RM,
             lmo_test_gt_TV=lmo_test_gt_TV,
             lmo_test_gt_scale=lmo_test_gt_scale,
             lmo_test_gt_bbox=lmo_test_gt_bbox)

# detected Linemod test latent variables
lm_test_maskrcnn_datafold = datasets.ImageFolder(
    parent_dir + '/processed data/lm_test_maskrcnn/' + ob_id,
    transform=transforms.ToTensor())

lm_test_maskrcnn_dataloader = DataLoader(
    lm_test_maskrcnn_datafold, batch_size=128, pin_memory=True, shuffle=False,
    num_workers=multiprocessing.Pool()._processes)

lm_test_maskrcnn_latent_mean = Generate(network,
                                        lm_test_maskrcnn_dataloader, device)
lm_test_maskrcnn_latent_mean = lm_test_maskrcnn_latent_mean.cpu().numpy()

lm_test_maskrcnn_json = json.load(open(
    parent_dir + '/processed data/lm_test_maskrcnn/' + ob_id + '.json'))

lm_test_maskrcnn_RM = []
lm_test_maskrcnn_TV = []
lm_test_maskrcnn_scale = []
lm_test_maskrcnn_bbox = []

for column in range(len(lm_test_maskrcnn_json)):
    RM = lm_test_maskrcnn_json[column]['RM']
    RM = np.reshape(RM, (3, 3))
    TV = lm_test_maskrcnn_json[column]['TV']
    bbox = lm_test_maskrcnn_json[column]['bbox']
    scale = lm_test_maskrcnn_json[column]['scale']

    lm_test_maskrcnn_RM.append(RM)
    lm_test_maskrcnn_TV.append(TV)
    lm_test_maskrcnn_scale.append(scale)
    lm_test_maskrcnn_bbox.append(bbox)

lm_test_maskrcnn_RM = np.array(lm_test_maskrcnn_RM, dtype='float32')
lm_test_maskrcnn_TV = np.array(lm_test_maskrcnn_TV, dtype='float32')
lm_test_maskrcnn_scale = np.expand_dims(np.array(lm_test_maskrcnn_scale,
                                                 dtype='float32'), axis=1)
lm_test_maskrcnn_bbox = np.array(lm_test_maskrcnn_bbox, dtype='float32')

# detected Linemod-occlusion test latent variables
if object_number in LMO_objects:

    lmo_test_maskrcnn_datafold = datasets.ImageFolder(
        parent_dir + '/processed data/lmo_test_bop19_maskrcnn/' + ob_id,
        transform=transforms.ToTensor())
    lmo_test_maskrcnn_dataloader = DataLoader(
        lmo_test_maskrcnn_datafold, batch_size=32, pin_memory=True,
        shuffle=False, num_workers=multiprocessing.Pool()._processes)

    lmo_test_maskrcnn_latent_mean = Generate(network,
                                             lmo_test_maskrcnn_dataloader,
                                             device)
    lmo_test_maskrcnn_latent_mean = lmo_test_maskrcnn_latent_mean.cpu().numpy()

    lmo_test_maskrcnn_json = json.load(open(
        parent_dir + '/processed data/lmo_test_bop19_maskrcnn/' +
        ob_id + '.json'))

    lmo_test_maskrcnn_RM = []
    lmo_test_maskrcnn_TV = []
    lmo_test_maskrcnn_scale = []
    lmo_test_maskrcnn_bbox = []

    for column in range(len(lmo_test_maskrcnn_json)):
        RM = lmo_test_maskrcnn_json[column]['RM']
        RM = np.reshape(RM, (3, 3))
        TV = lmo_test_maskrcnn_json[column]['TV']
        bbox = lmo_test_maskrcnn_json[column]['bbox']
        scale = lmo_test_maskrcnn_json[column]['scale']

        lmo_test_maskrcnn_RM.append(RM)
        lmo_test_maskrcnn_TV.append(TV)
        lmo_test_maskrcnn_scale.append(scale)
        lmo_test_maskrcnn_bbox.append(bbox)

    lmo_test_maskrcnn_RM = np.array(lmo_test_maskrcnn_RM, dtype='float32')
    lmo_test_maskrcnn_TV = np.array(lmo_test_maskrcnn_TV, dtype='float32')
    lmo_test_maskrcnn_scale = np.expand_dims(np.array(lmo_test_maskrcnn_scale,
                                                      dtype='float32'), axis=1)
    lmo_test_maskrcnn_bbox = np.array(lmo_test_maskrcnn_bbox, dtype='float32')

    lmo_latent_test_maskrcnn = lmo_latent_path + ob_id + '_test_maskrcnn'

    np.savez(lmo_latent_test_maskrcnn,
             lmo_test_maskrcnn_latent_mean=lmo_test_maskrcnn_latent_mean,
             lmo_test_maskrcnn_RM=lmo_test_maskrcnn_RM,
             lmo_test_maskrcnn_TV=lmo_test_maskrcnn_TV,
             lmo_test_maskrcnn_scale=lmo_test_maskrcnn_scale,
             lmo_test_maskrcnn_bbox=lmo_test_maskrcnn_bbox)

lm_latent_train = lm_latent_path + ob_id + '_train'
lm_latent_valid = lm_latent_path + ob_id + '_valid'
lm_latent_test_gt = lm_latent_path + ob_id + '_test_gt'
lm_latent_test_maskrcnn = lm_latent_path + ob_id + '_test_maskrcnn'

np.savez(lm_latent_train,
         lm_train_latent_mean=lm_train_latent_mean,
         lm_train_RM=lm_train_RM,
         lm_train_TV=lm_train_TV,
         lm_train_scale=lm_train_scale,
         lm_train_bb=lm_train_bb,
         lm_train_ObjectCx=lm_train_ObjectCx,
         lm_train_ObjectCy=lm_train_ObjectCy)

np.savez(lm_latent_valid,
         lm_valid_latent_mean=lm_valid_latent_mean,
         lm_valid_RM=lm_valid_RM,
         lm_valid_TV=lm_valid_TV,
         lm_valid_scale=lm_valid_scale,
         lm_valid_bb=lm_valid_bb,
         lm_valid_ObjectCx=lm_valid_ObjectCx,
         lm_valid_ObjectCy=lm_valid_ObjectCy)

np.savez(lm_latent_test_gt,
         lm_test_gt_latent_mean=lm_test_gt_latent_mean,
         lm_test_gt_RM=lm_test_gt_RM,
         lm_test_gt_TV=lm_test_gt_TV,
         lm_test_gt_scale=lm_test_gt_scale,
         lm_test_gt_bbox=lm_test_gt_bbox)

np.savez(lm_latent_test_maskrcnn,
         lm_test_maskrcnn_latent_mean=lm_test_maskrcnn_latent_mean,
         lm_test_maskrcnn_RM=lm_test_maskrcnn_RM,
         lm_test_maskrcnn_TV=lm_test_maskrcnn_TV,
         lm_test_maskrcnn_scale=lm_test_maskrcnn_scale,
         lm_test_maskrcnn_bbox=lm_test_maskrcnn_bbox)