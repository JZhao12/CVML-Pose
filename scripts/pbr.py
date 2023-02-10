#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:20:06 2022

@author: bogdan

This script is used to process the LineMod PBR images

"""

# %% import modules
import imageio
import numpy as np
import pandas as pd
import json
import glob
import cv2
import os
import math
from torchvision import datasets, transforms

# %% PBR_train path
parent_dir = os.getcwd()
num_models = len(glob.glob(parent_dir + '/original data/lm/models/' +
                           '*.ply'))
pbr_save_dir = parent_dir + '/processed data/pbr_train/'

# %% create folders
for i in range(1, num_models+1):
    folderid = str(i).zfill(6)
    PBR_train_dir = pbr_save_dir + folderid
    for k in range(50):
        directory = str(k).zfill(6)
        path = os.path.join(PBR_train_dir, directory)
        if os.path.exists(path) is False:
            os.makedirs(path)

# %% LineMod PBR: extract and save into new folders
for foldername in range(50):

    folderid = str(foldername).zfill(6)
    PBR_folder = (parent_dir +
                  '/original data/lm/train_pbr/' + folderid)

    scene_gt_info_path = PBR_folder + '/scene_gt_info.json'
    scene_gt_path = PBR_folder + '/scene_gt.json'

    scene_gt_info = json.load(open(scene_gt_info_path))
    scene_gt = json.load(open(scene_gt_path))

    rgb_images = sorted(glob.glob(PBR_folder + '/rgb/' + "*.jpg"))
    objectidx = 0

    name = []
    GTRM = []
    obID = []
    GTTV = []
    BBOX = []

    for key in scene_gt_info:

        object_gt = scene_gt[key]
        object_gt_info = scene_gt_info[key]
        rgb = cv2.cvtColor(cv2.imread(rgb_images[int(key)]), cv2.COLOR_BGR2RGB)

        for column in range(len(object_gt_info)):

            obj_id = object_gt[column]["obj_id"]
            cam_R_m2c = object_gt[column]["cam_R_m2c"]
            cam_t_m2c = object_gt[column]["cam_t_m2c"]
            visib_fract = object_gt_info[column]["visib_fract"]
            bbox_obj = object_gt_info[column]["bbox_visib"]
            x, y, w, h = bbox_obj

            if visib_fract >= 0.1:

                if w > h:
                    new_x = x
                    new_y = math.floor(y-(w-h)/2)
                    new_w = w
                    new_h = w
                else:
                    new_x = math.floor(x-(h-w)/2)
                    new_y = y
                    new_w = h
                    new_h = h

                left_trunc = np.maximum(new_x, 0)
                right_trunc = np.minimum(new_x + new_w, rgb.shape[1])
                top_trunc = np.maximum(new_y, 0)
                bottom_trunc = np.minimum(new_y + new_h, rgb.shape[0])

                ROI = rgb[top_trunc:bottom_trunc, left_trunc:right_trunc]
                resized = cv2.resize(ROI, (128, 128),
                                     interpolation=cv2.INTER_CUBIC)

                objectid = str(obj_id).zfill(6)
                filename = '%s_%s_%06i.png' % (objectid, folderid, objectidx)
                path = os.path.join(pbr_save_dir, objectid,
                                    folderid, filename)
                imageio.imwrite(path, resized)
                print(path)

                name.append(filename)
                GTRM.append(cam_R_m2c)
                GTTV.append(cam_t_m2c)
                obID.append(obj_id)
                BBOX.append(bbox_obj)

            objectidx = objectidx + 1

    csvfile = {'filename': name, 'RM': GTRM,
               'TV': GTTV, 'bbox': BBOX, 'ID': obID}

    df = pd.DataFrame(csvfile)

    df.set_index("ID", inplace=True)
    df.head()

    for obidx in range(1, num_models+1):
        subset = df.loc[obidx]
        subset = subset.to_dict(orient='records')

        json_name = '%s.json' % (folderid)
        IDobject = str(obidx).zfill(6)
        json_path = os.path.join(pbr_save_dir, IDobject, json_name)

        json_file = open(json_path, "w")
        json.dump(subset, json_file)

# %% Check if there are any errors
for obidx in range(1, num_models+1):
    folderid = str(obidx).zfill(6)
    GT_Train_path = pbr_save_dir + folderid + '/'
    GT_Train_jsons = sorted(glob.glob(GT_Train_path + "*.json"))
    jsonreader = json.load(open(GT_Train_jsons[0]))

    for sceneid in range(1, len(GT_Train_jsons)):
        temp = json.load(open(GT_Train_jsons[sceneid]))
        jsonreader = jsonreader + temp

    train_dataset = datasets.ImageFolder(pbr_save_dir + folderid,
                                         transform=transforms.ToTensor())

    if len(train_dataset) == len(jsonreader):
        json_name = '%s_all.json' % (folderid)
        json_path = os.path.join(pbr_save_dir, json_name)
        json_file = open(json_path, "w")
        json.dump(jsonreader, json_file)
    else:
        print('error')