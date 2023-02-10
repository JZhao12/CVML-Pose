#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:38:53 2022

@author: bogdan

This script is used to process the detected LineMod and LineMod-Occlusion
test images

The detector model is based on:
https://github.com/ylabbe/cosypose

Some lines of code are also based on:
https://github.com/ylabbe/cosypose

"""
# %% import modules
import sys
# please change the below line of code to your cosypose path
sys.path.append('/home/jianyu/CVML-Pose/cosypose/')

import yaml
import torch
import cosypose
import glob
import os
import json
import cv2
import math
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector)
from cosypose.integrated.detector import Detector

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
EXP_DIR = Path(cosypose.__file__).parent.parent

# %% functions
def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.Loader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def getModel():
    detector_run_id = 'local_data/experiments/detector-bop-lmo-pbr--517542'
    detector = load_detector(detector_run_id)
    return detector


def inference(detector, image):
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    box_detections = detector.get_detections(
        images=images, one_instance_per_class=False,
        detection_th=0.8, output_masks=False, mask_th=0.9)

    if len(box_detections) == 0:
        return None

    return box_detections.cpu()


# %% LineMod: path, information
detector = getModel()

parent_dir = os.getcwd()
LM_test_original = parent_dir + '/original data/lm/test/'
LM_test_split = parent_dir + '/original data/LINEMOD'
imagesave_dir = parent_dir + '/processed data/lm_test_maskrcnn/'

LM_objects = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]

directory = "1"
for i in LM_objects:
    folderid = str(i).zfill(6)
    fold_dir = imagesave_dir + folderid
    path = os.path.join(fold_dir, directory)
    if os.path.exists(path) is False:
        os.makedirs(path)

# %% LineMod: extract test from list, detect, and crop
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
    Scale = []
    BBOX = []

    folderid = str(i).zfill(6)
    indexstr = 'obj_' + folderid

    scene_gt_path = LM_test_original + folderid + '/scene_gt.json'

    scene_gt = json.load(open(scene_gt_path))

    rgb_path = LM_test_original + folderid + '/rgb/'

    rgb_images = sorted(glob.glob(rgb_path + "*.png"))

    for column in scene_gt:

        int_id = int(column)

        if int_id in test_list:

            rgb = cv2.cvtColor(cv2.imread(rgb_images[int_id]),
                               cv2.COLOR_BGR2RGB)

            pred = inference(detector, rgb)
            info = pred.infos
            bboxes = pred.bboxes
            deter = info["label"] == indexstr
            if deter.any():
                idx = info.index[info["label"] == indexstr].tolist()
                if len(idx) > 1:
                    idx = [idx[0]]
                bb = bboxes[idx]
                bb = bb.tolist()[0]

                X1, Y1, X2, Y2 = bb
                X1 = math.floor(X1)
                Y1 = math.floor(Y1)
                X2 = math.ceil(X2)
                Y2 = math.ceil(Y2)

                bbox_obj = [X1, Y1, X2-X1, Y2-Y1]

                obj_id = scene_gt[column][0]["obj_id"]
                cam_R_m2c = scene_gt[column][0]["cam_R_m2c"]
                cam_t_m2c = scene_gt[column][0]["cam_t_m2c"]
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

                filename = (imagesave_dir + folderid +
                            '/1' + '/%06i.png' % (int_id))
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

# %% LineMod-Occlusion: path, information
detector = getModel()

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

LMO_test_BOP19 = parent_dir + '/original data/lmo/test/000002/'
imagesave_dir = parent_dir + '/processed data/lmo_test_bop19_maskrcnn/'

LMO_objects = [1, 5, 6, 8, 9, 10, 11, 12]

directory = "1"
for i in LMO_objects:

    folderid = str(i).zfill(6)
    PBR_train_dir = imagesave_dir + folderid
    path = os.path.join(PBR_train_dir, directory)
    if os.path.exists(path) is False:
        os.makedirs(path)

# %% LineMod-Occlusion: extract and crop
scene_gt_path = LMO_test_BOP19 + 'scene_gt.json'
scene_gt = json.load(open(scene_gt_path))
rgb_path = LMO_test_BOP19 + '/rgb/'

name = []
GTRM = []
obID = []
GTTV = []
BBOX = []
Scale = []

for key in scene_gt:

    object_gt = scene_gt[key]
    a = rgb_path + key.zfill(6) + '.png'
    rgb = cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB)

    pred = inference(detector, rgb)
    info = pred.infos
    bboxes = pred.bboxes
    bboxes = bboxes.tolist()
    label = pred.infos.label.tolist()

    for a in LMO_objects:
        str_a = 'obj_' + str(a).zfill(6)
        if str_a in label:
            bb_idx = label.index(str_a)
            bb = bboxes[bb_idx]

            X1, Y1, X2, Y2 = bb
            X1 = math.floor(X1)
            Y1 = math.floor(Y1)
            X2 = math.ceil(X2)
            Y2 = math.ceil(Y2)
            bbox_obj = [X1, Y1, X2-X1, Y2-Y1]

            for column in range(len(object_gt)):

                obj_id = object_gt[column]["obj_id"]

                if obj_id == a:
                    cam_R_m2c = object_gt[column]["cam_R_m2c"]
                    cam_t_m2c = object_gt[column]["cam_t_m2c"]
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

                    objectid = str(obj_id).zfill(6)
                    filename = '%s.png' % (key.zfill(6))
                    path = os.path.join(imagesave_dir, objectid,
                                        directory, filename)
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