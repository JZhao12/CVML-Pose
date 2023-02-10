# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 14:07:51 2022

@author: Eddie

This script is used to evaluate the estimated pose.

"""

# %% import modules
import sys
# please change the below line of code to your bop_toolkit path
sys.path.append('/home/jianyu/CVML-Pose/bop_toolkit/')

import os
import numpy as np
import json
import argparse
from bop_toolkit_lib import pose_error
from bop_toolkit_lib.inout import load_ply
from bop_toolkit_lib.misc import get_symmetry_transformations

# %% path, argparse, data information
parent_dir = os.getcwd()

parser = argparse.ArgumentParser(description='evaluate lm and lmo')
parser.add_argument('--model', type=str, help='cvml models',
                    choices=['CVML_base', 'CVML_18', 'CVML_34'])
parser.add_argument('--data', type=str, help='which test data',
                    choices=['lm', 'lmo'])
parser.add_argument('--type', type=str, help='which type of data',
                    choices=['gt', 'mask'])
args = parser.parse_args()

if args.model == 'CVML_base':
    latent_path = parent_dir + '/trained_model/CVML_base/'
elif args.model == 'CVML_18':
    latent_path = parent_dir + '/trained_model/CVML_18/'
elif args.model == 'CVML_34':
    latent_path = parent_dir + '/trained_model/CVML_34/'

LM_objects = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
LMO_objects = [1, 5, 6, 8, 9, 10, 11, 12]
symmetry = [10, 11]
asymmetry = [1, 2, 4, 5, 6, 8, 9, 12, 13, 14, 15]
MeshDir = parent_dir + '/original data/lm/models_eval/'


"""
The ADD, MSSD, MSPD metrics are based on:
https://github.com/thodan/bop_toolkit

"""
# %% evaluation on the ground truth LineMod test images
if args.data == 'lm' and args.type == 'gt':

    result_path = latent_path + '/lm_results_gt/'
    print('LineMod ground truth test images, use ADD evaluation')
    Overall = []

    for ob_id in LM_objects:

        X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
        PredictedRM = X['PredictedRM']

        X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
        PredictedTV = X['PredictedTV']

        X = np.load(latent_path + '/lm_latent/' +
                    str(ob_id).zfill(6) + '_test_gt.npz')
        GT_RM = X['lm_test_gt_RM']
        GT_TV = X['lm_test_gt_TV']

        Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
        Modelpoint = load_ply(Target_model)['pts']
        model_info = json.load(open(MeshDir + 'models_info.json'))
        diameter = model_info[str(ob_id)]['diameter']

        radio = 10
        good_prediction = 0
        All_distance = []

        for i in range(len(GT_TV)):

            if ob_id in asymmetry:
                distance = pose_error.add(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)
            elif ob_id in symmetry:
                distance = pose_error.adi(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)

            All_distance.append(distance)
            if distance < diameter*radio/100:
                good_prediction += 1

        Accuracy = good_prediction/len(GT_TV) * 100
        print('Number of tested instance: %d' % len(GT_TV))
        print(Accuracy)
        Overall.append(Accuracy)
    print("AP(ADD) : ", np.mean(Overall))

# %% evaluation on the detected LineMod images
elif args.data == 'lm' and args.type == 'mask':

    result_path = latent_path + '/lm_results_maskrcnn/'
    print('LineMod detected test images, use ADD evaluation')
    Overall = []

    for ob_id in LM_objects:

        X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
        PredictedRM = X['PredictedRM']

        X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
        PredictedTV = X['PredictedTV']

        X = np.load(latent_path + '/lm_latent/' +
                    str(ob_id).zfill(6) + '_test_maskrcnn.npz')
        GT_RM = X['lm_test_maskrcnn_RM']
        GT_TV = X['lm_test_maskrcnn_TV']

        Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
        Modelpoint = load_ply(Target_model)['pts']
        model_info = json.load(open(MeshDir + 'models_info.json'))
        diameter = model_info[str(ob_id)]['diameter']

        radio = 10
        good_prediction = 0
        All_distance = []

        for i in range(len(GT_TV)):

            if ob_id in asymmetry:
                distance = pose_error.add(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)
            elif ob_id in symmetry:
                distance = pose_error.adi(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)

            All_distance.append(distance)
            if distance < diameter*radio/100:
                good_prediction += 1

        Accuracy = good_prediction/len(GT_TV) * 100
        print('Number of tested instance: %d' % len(GT_TV))
        print(Accuracy)
        Overall.append(Accuracy)
    print("AP(ADD) : ", np.mean(Overall))

# %% evaluation on the ground truth BOP version of the LineMod-Occlusion test images
elif args.data == 'lmo' and args.type == 'gt':

    result_path = latent_path + '/lmo_results_gt/'
    print('LM-O ground truth test images, use ADD evaluation')
    Overall = []

    for ob_id in LMO_objects:

        X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
        PredictedRM = X['PredictedRM']

        X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
        PredictedTV = X['PredictedTV']

        X = np.load(latent_path + '/lmo_latent/' +
                    str(ob_id).zfill(6) + '_test_gt.npz')
        GT_RM = X['lmo_test_gt_RM']
        GT_TV = X['lmo_test_gt_TV']

        Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
        Modelpoint = load_ply(Target_model)['pts']
        model_info = json.load(open(MeshDir + 'models_info.json'))
        diameter = model_info[str(ob_id)]['diameter']

        radio = 10
        good_prediction = 0
        All_distance = []

        for i in range(len(GT_TV)):

            if ob_id in asymmetry:
                distance = pose_error.add(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)
            elif ob_id in symmetry:
                distance = pose_error.adi(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)

            All_distance.append(distance)
            if distance < diameter*radio/100:
                good_prediction += 1

        Accuracy = good_prediction/len(GT_TV) * 100
        print('Number of tested instance: %d' % len(GT_TV))
        print(Accuracy)
        Overall.append(Accuracy)
    print("AP(ADD) : ", np.mean(Overall))

# %% evaluation on the detected BOP version of the LineMod-Occlusion test images
elif args.data == 'lmo' and args.type == 'mask':

    result_path = latent_path + '/lmo_results_maskrcnn/'
    print('LM-O detected test images, use ADD, MSSD, MSPD')

    # ADD
    print('Metric: ADD')
    Overall = []

    for ob_id in LMO_objects:

        X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
        PredictedRM = X['PredictedRM']

        X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
        PredictedTV = X['PredictedTV']

        X = np.load(latent_path + '/lmo_latent/' +
                    str(ob_id).zfill(6) + '_test_maskrcnn.npz')
        GT_RM = X['lmo_test_maskrcnn_RM']
        GT_TV = X['lmo_test_maskrcnn_TV']

        Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
        Modelpoint = load_ply(Target_model)['pts']
        model_info = json.load(open(MeshDir + 'models_info.json'))
        diameter = model_info[str(ob_id)]['diameter']

        radio = 10
        good_prediction = 0
        All_distance = []

        for i in range(len(GT_TV)):

            if ob_id in asymmetry:
                distance = pose_error.add(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)
            elif ob_id in symmetry:
                distance = pose_error.adi(PredictedRM[i], PredictedTV[i],
                                          GT_RM[i], GT_TV[i], Modelpoint)

            All_distance.append(distance)
            if distance < diameter*radio/100:
                good_prediction += 1

        Accuracy = good_prediction/len(GT_TV) * 100
        print('Number of tested instance: %d' % len(GT_TV))
        print(Accuracy)
        Overall.append(Accuracy)
    print("AP(ADD) : ", np.mean(Overall))

    # MSSD
    print('Metric: MSSD')

    Overall = []

    for ob_id in LMO_objects:

        X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
        PredictedRM = X['PredictedRM']

        X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
        PredictedTV = X['PredictedTV']

        X = np.load(latent_path + '/lmo_latent/' +
                    str(ob_id).zfill(6) + '_test_maskrcnn.npz')
        GT_RM = X['lmo_test_maskrcnn_RM']
        GT_TV = X['lmo_test_maskrcnn_TV']

        Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
        Modelpoint = load_ply(Target_model)['pts']
        model_info = json.load(open(MeshDir + 'models_info.json'))
        s_trans = get_symmetry_transformations(model_info[str(ob_id)], 0.01)
        diameter = model_info[str(ob_id)]['diameter']

        radio_range = np.arange(0.05, 0.51, 0.05)
        MSSD = []
        for radio_idx in range(len(radio_range)):
            radio = radio_range[radio_idx]

            good_prediction = 0
            All_distance = []

            for i in range(len(GT_TV)):
                R_est = PredictedRM[i]
                t_est = np.reshape(PredictedTV[i], (3, 1))
                R_gt = GT_RM[i]
                t_gt = np.reshape(GT_TV[i], (3, 1))

                distance = pose_error.mssd(R_est, t_est, R_gt, t_gt,
                                           Modelpoint, s_trans)

                All_distance.append(distance)
                if distance < diameter*radio:
                    good_prediction += 1

            Accuracy = good_prediction/len(GT_TV) * 100
            MSSD.append(Accuracy)

        error_MSSD = np.mean(MSSD)
        print('Number of obejcts detected: %d' % len(GT_TV))
        print(error_MSSD)
        Overall.append(error_MSSD)
    print("AR_MSSD : ", np.mean(Overall))

    print('Metric: MSPD')

    Overall = []

    for ob_id in LMO_objects:

        X = np.load(result_path + str(ob_id).zfill(6) + '_rotation.npz')
        PredictedRM = X['PredictedRM']

        X = np.load(result_path + str(ob_id).zfill(6) + '_translation.npz')
        PredictedTV = X['PredictedTV']

        X = np.load(latent_path + '/lmo_latent/' +
                    str(ob_id).zfill(6) + '_test_maskrcnn.npz')
        GT_RM = X['lmo_test_maskrcnn_RM']
        GT_TV = X['lmo_test_maskrcnn_TV']

        Target_model = MeshDir + 'obj_' + str(ob_id).zfill(6) + '.ply'
        Modelpoint = load_ply(Target_model)['pts']
        model_info = json.load(open(MeshDir + 'models_info.json'))
        s_trans = get_symmetry_transformations(model_info[str(ob_id)], 0.01)

        test_camera = json.load(open(parent_dir + '/original data/' +
                                     'lmo/test/000002/' +
                                     'scene_camera.json'))
        K = np.reshape(test_camera['3']['cam_K'], (3, 3))

        radio_range = np.arange(5, 51, 5)
        w = 640
        r = w/640

        MSPD = []
        for radio_idx in range(len(radio_range)):
            radio = radio_range[radio_idx]

            good_prediction = 0
            All_distance = []

            for i in range(len(GT_TV)):
                R_est = PredictedRM[i]
                t_est = np.reshape(PredictedTV[i], (3, 1))
                R_gt = GT_RM[i]
                t_gt = np.reshape(GT_TV[i], (3, 1))

                distance = pose_error.mspd(R_est, t_est, R_gt, t_gt,
                                           K, Modelpoint, s_trans)

                All_distance.append(distance)
                if distance < r*radio:
                    good_prediction += 1

            Accuracy = good_prediction/len(GT_TV) * 100
            MSPD.append(Accuracy)

        error_MSPD = np.mean(MSPD)
        print('Number of obejcts detected: %d' % len(GT_TV))
        print(error_MSPD)
        Overall.append(error_MSPD)
    print("AR_MSPD : ", np.mean(Overall))