#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:44:59 2022

@author: bogdan

This script is used to process the ground truth LineMod and LineMod-Occlusion
test images

"""
# %% import modules
import numpy as np
import pandas as pd
import glob
import cv2
import os
import imageio
import json
import math

# %% LineMod test: path, data information
parent_dir = os.getcwd()
LM_test_original = parent_dir + '/original data/lm/test/'
LM_test_split = parent_dir + '/original data/LINEMOD'
Num_models = len(glob.glob(
    parent_dir + '/original data/lm/models/' + '*.ply'))
imagesave_dir = parent_dir + '/processed data/lm_test_gt/'

LM_objects = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]

# %% LineMod test: create folders
directory = "1"

for i in LM_objects:

    folderid = str(i).zfill(6)
    fold_dir = imagesave_dir + folderid
    path = os.path.join(fold_dir, directory)
    if os.path.exists(path) is False:
        os.makedirs(path)

# %% LineMod test: extract from the list, and save into new folders
for i in LM_objects:

    if i == 1:
        test_split_file = LM_test_split + '/ape/test.txt'
    elif i == 2:
        test_split_file = LM_test_split + '/benchvise/test.txt'
    elif i == 4:
        test_split_file = LM_test_split + '/cam/test.txt'
    elif i == 5:
        test_split_file = LM_test_split + '/can/test.txt'
    elif i == 6:
        test_split_file = LM_test_split + '/cat/test.txt'
    elif i == 8:
        test_split_file = LM_test_split + '/driller/test.txt'
    elif i == 9:
        test_split_file = LM_test_split + '/duck/test.txt'
    elif i == 10:
        test_split_file = LM_test_split + '/eggbox/test.txt'
    elif i == 11:
        test_split_file = LM_test_split + '/glue/test.txt'
    elif i == 12:
        test_split_file = LM_test_split + '/holepuncher/test.txt'
    elif i == 13:
        test_split_file = LM_test_split + '/iron/test.txt'
    elif i == 14:
        test_split_file = LM_test_split + '/lamp/test.txt'
    elif i == 15:
        test_split_file = LM_test_split + '/phone/test.txt'

    file = open(test_split_file, "r")
    file_read = file.readlines()
    test_list = []

    for test_candidates in file_read:
        m = test_candidates[-11:-5]
        m = int(m)
        test_list.append(m)

    if i == 2:
        test_list = test_list[:-1]

    GTRM = []
    obID = []
    GTTV = []
    BBOX = []
    Scale = []

    folderid = str(i).zfill(6)

    scene_gt_info_path = LM_test_original + folderid + '/scene_gt_info.json'
    scene_gt_path = LM_test_original + folderid + '/scene_gt.json'

    scene_gt_info = json.load(open(scene_gt_info_path))
    scene_gt = json.load(open(scene_gt_path))

    rgb_path = LM_test_original + folderid + '/rgb/'

    rgb_images = sorted(glob.glob(rgb_path + "*.png"))

    for column in scene_gt_info:

        int_id = int(column)

        if int_id in test_list:

            rgb = cv2.cvtColor(cv2.imread(rgb_images[int_id]),
                               cv2.COLOR_BGR2RGB)

            obj_id = scene_gt[column][0]["obj_id"]
            cam_R_m2c = scene_gt[column][0]["cam_R_m2c"]
            cam_t_m2c = scene_gt[column][0]["cam_t_m2c"]
            bbox_obj = scene_gt_info[column][0]["bbox_visib"]

            x, y, w, h = bbox_obj
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

            s = max(w, h)
            scale_factor = 128/s

            filename = imagesave_dir + folderid + '/1' + '/%06i.png' % (int_id)
            imageio.imwrite(filename, resized)
            print(filename)

            GTRM.append(cam_R_m2c)
            GTTV.append(cam_t_m2c)
            obID.append(obj_id)
            BBOX.append(bbox_obj)
            Scale.append(scale_factor)

    csvfile = {'RM': GTRM, 'TV': GTTV, 'bbox': BBOX,
               'scale': Scale, 'ID': obID}

    df = pd.DataFrame(csvfile)
    df.set_index("ID", inplace=True)
    df.head()
    subset = df.to_dict(orient='records')

    IDobject = str(i).zfill(6)
    json_name = '%s.json' % (IDobject)
    json_path = os.path.join(imagesave_dir, json_name)
    json_file = open(json_path, "w")
    json.dump(subset, json_file)

# %% LineMod-Occlusion test: path, data information
LMO_test_BOP19 = parent_dir + '/original data/lmo/test/000002/'
imagesave_dir = parent_dir + '/processed data/lmo_test_bop19_gt/'

LMO_objects = [1, 5, 6, 8, 9, 10, 11, 12]

# %% LineMod-Occlusion test: create folders
directory = "1"

for i in LMO_objects:

    folderid = str(i).zfill(6)
    PBR_train_dir = imagesave_dir + folderid
    path = os.path.join(PBR_train_dir, directory)
    if os.path.exists(path) is False:
        os.makedirs(path)

# %% LineMod test: extract and save into new folders
scene_gt_info_path = LMO_test_BOP19 + 'scene_gt_info.json'
scene_gt_path = LMO_test_BOP19 + 'scene_gt.json'
scene_gt_info = json.load(open(scene_gt_info_path))
scene_gt = json.load(open(scene_gt_path))

rgb_path = LMO_test_BOP19 + '/rgb/'

name = []
GTRM = []
obID = []
GTTV = []
BBOX = []
Scale = []

objectidx = 0

for key in scene_gt_info:

    object_gt = scene_gt[key]
    object_gt_info = scene_gt_info[key]
    a = rgb_path + key.zfill(6) + '.png'
    rgb = cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB)

    for column in range(len(object_gt_info)):
        obj_id = object_gt[column]["obj_id"]
        cam_R_m2c = object_gt[column]["cam_R_m2c"]
        cam_t_m2c = object_gt[column]["cam_t_m2c"]
        bbox_obj = object_gt_info[column]["bbox_visib"]

        x, y, w, h = bbox_obj

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

        resized = cv2.resize(ROI, (128, 128), interpolation=cv2.INTER_CUBIC)
        s = max(w, h)
        scale_factor = 128/s

        objectid = str(obj_id).zfill(6)
        filename = '%s_%s.png' % (objectid, key.zfill(6))
        path = os.path.join(imagesave_dir, objectid, directory, filename)
        imageio.imwrite(path, resized)
        print(path)

        name.append(filename)
        GTRM.append(cam_R_m2c)
        GTTV.append(cam_t_m2c)
        obID.append(obj_id)
        BBOX.append(bbox_obj)
        Scale.append(scale_factor)

csvfile = {'filename': name, 'RM': GTRM, 'TV': GTTV,
           'bbox': BBOX, 'scale': Scale, 'ID': obID}

df = pd.DataFrame(csvfile)

df.set_index("ID", inplace=True)
df.head()

for obidx in range(len(LMO_objects)):
    subset = df.loc[LMO_objects[obidx]]
    subset = subset.to_dict(orient='records')

    IDobject = str(LMO_objects[obidx]).zfill(6)
    json_name = '%s.json' % (IDobject)
    json_path = os.path.join(imagesave_dir, json_name)

    json_file = open(json_path, "w")
    json.dump(subset, json_file)
