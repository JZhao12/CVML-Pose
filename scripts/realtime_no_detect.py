#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:18:37 2022

@author: bogdan

This script is used for realtime implementation with no detector.

The real-time rendering is based on:
https://github.com/mmatl/pyrender

"""
# %% import modules
import torch
import joblib
import json
import trimesh
import pyrender
import imageio
import cv2
import math
import os
import argparse
import numpy as np
from torchvision import transforms
from model import CVML_18, R_MLP, T_MLP
from engine import rotation_6d_to_matrix

# %% settings for camera and models

parser = argparse.ArgumentParser(description='realtime with no detection')
parser.add_argument('--object', type=int, help='object to test')
args = parser.parse_args()

cap = cv2.VideoCapture(0)
bgr = (255, 0, 0)

parent_dir = os.getcwd()
model_path = parent_dir + '/trained_model/CVML_18'

obj = args.object
obj_id = str(obj).zfill(6)
indexstr = 'obj_' + obj_id
device = torch.device("cuda:0")

# %% load vae, mlp, knn
vae_path = model_path + '/VAE/' + obj_id + '_vae.pth'
R_mlp_path = model_path + '/trained_mlp/' + obj_id + '_rotation.pth'
T_mlp_path = model_path + '/trained_mlp/' + obj_id + '_centre.pth'
knn_path = model_path + '/trained_knn/' + obj_id + '_knn.sav'

vae = CVML_18()
vae = vae.to(device)
vae.load_state_dict(torch.load(vae_path, map_location='cuda:0'))
vae.eval()
convert_tensor = transforms.ToTensor()

R_mlp = R_MLP()
R_mlp = R_mlp.to(device)
R_mlp.load_state_dict(torch.load(R_mlp_path))
R_mlp.eval()

T_mlp = T_MLP()
T_mlp = T_mlp.to(device)
T_mlp.load_state_dict(torch.load(T_mlp_path))
T_mlp.eval()

Final_neigh = joblib.load(knn_path)

# %% start

test_camera = json.load(open(
    parent_dir + '/original data/lm/test/' +
    obj_id + '/scene_camera.json'))
test_Camera_IM = np.reshape(test_camera['0']['cam_K'], (3, 3))

MeshDir = parent_dir + '/original data/lm/models/'

view_width = 640
view_height = 480

test_camera = json.load(open(
    parent_dir + '/original data/lm/test/' +
    obj_id + '/scene_camera.json'))
test_Camera_IM = np.reshape(test_camera['0']['cam_K'], (3, 3))

fx,fy,cx,cy = test_Camera_IM[0,0], test_Camera_IM[1,1], test_Camera_IM[0,2], test_Camera_IM[1,2]

camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.05, zfar=5000.0)
camera_pose = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=5)

fuze_trimesh = trimesh.load(MeshDir + 'obj_' + obj_id + '.ply')

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frames = []

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

    bb = np.array([265.3092, 221.05261, 395.13208, 327.14752])

    x1, y1, x2, y2 = bb

    x1 = math.floor(x1)
    y1 = math.floor(y1)
    x2 = math.ceil(x2)
    y2 = math.ceil(y2)

    x = 265
    y = 221
    w = 131
    h = 107

    bbox_obj = [x, y, w, h]

    if w > h:
        new_x = math.floor(x)
        new_y = math.floor(y-(w-h)/2)
        new_w = math.floor(w)
        new_h = math.floor(w)
    else:
        new_x = math.floor(x-(h-w)/2)
        new_y = math.floor(y)
        new_w = math.floor(h)
        new_h = math.floor(h)

    left_trunc = np.maximum(new_x, 0)
    right_trunc = np.minimum(new_x + new_w, frame.shape[1])
    top_trunc = np.maximum(new_y, 0)
    bottom_trunc = np.minimum(new_y + new_h, frame.shape[0])

    ROI = frame[top_trunc:bottom_trunc, left_trunc:right_trunc]
    resized = cv2.resize(ROI, (128, 128), interpolation=cv2.INTER_CUBIC)

    s = max(new_w, new_h)
    scale_factor = 128/s

# %% vae
    Test_im = convert_tensor(resized)
    Test_im = Test_im.unsqueeze(0)

    with torch.no_grad():
        Test_im = Test_im.to(device)
        _, Test_Mean, _ = vae(Test_im)

# %% rotation mlp
    with torch.no_grad():
        PredictedRM = R_mlp(Test_Mean)

    PredictedRM = rotation_6d_to_matrix(PredictedRM)
    PredictedRM = PredictedRM.cpu().numpy()

# %% translation mlp
    Test_Mean = torch.squeeze(Test_Mean)
    Test_Mean = Test_Mean.cpu().numpy()

    LM_Test_bbox = np.array(bbox_obj)

    TestBx = np.expand_dims(LM_Test_bbox[0], axis=0)
    TestBy = np.expand_dims(LM_Test_bbox[1], axis=0)
    Test_width = np.expand_dims(LM_Test_bbox[2], axis=0)
    Test_height = np.expand_dims(LM_Test_bbox[3], axis=0)
    scale_factor = np.expand_dims(scale_factor, axis=0)

    XTest = np.concatenate((Test_Mean, TestBx, TestBy, Test_width,
                            Test_height, scale_factor), axis=0)
    XTest = torch.tensor(XTest, dtype=torch.float)
    XTest = torch.unsqueeze(XTest, 0)

    with torch.no_grad():
        XTest = XTest.to(device)
        Predict_Test_center = T_mlp(XTest)

    Predict_Test_center = Predict_Test_center.cpu().numpy()
    Predict_Test_OCX = Predict_Test_center[:, 0]
    Predict_Test_OCY = Predict_Test_center[:, 1]

# %% knn
    XTest = np.concatenate((Test_Mean, Test_width,
                            Test_height, scale_factor), axis=0)
    XTest = np.expand_dims(XTest, axis=0)
    PredictedTz = Final_neigh.predict(XTest)

    PredictTx = ((Predict_Test_OCX - test_Camera_IM[0, 2]) *
                 PredictedTz/test_Camera_IM[0, 0])
    PredictTy = ((Predict_Test_OCY - test_Camera_IM[1, 2]) *
                 PredictedTz/test_Camera_IM[1, 1])

    Predicted_T = np.array([PredictTx[0][0],
                            PredictTy[0][0],
                            PredictedTz[0][0]])
    Predicted_R = np.squeeze(PredictedRM)

# %% render
    TS = np.array([[Predicted_T[0]],
                   [Predicted_T[1]],
                   [Predicted_T[2]]])
    Ex_parameter = np.array([[0.0, 0.0, 0.0, 1.0]])
    inter_pose = np.concatenate((Predicted_R, TS), axis=1)
    objectpose = np.concatenate((inter_pose, Ex_parameter), axis=0)

    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh, poses=objectpose,
                                      smooth=True)
    scene = pyrender.Scene(bg_color=([0, 0, 0]))
    scene.add(mesh)

    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=view_width,
                                   viewport_height=view_height)
    rendered_image, _ = r.render(scene)

    r.delete()

# %% imshow
    detection = cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)

    numpy_horizontal = np.hstack((detection, rendered_image))
    numpy_horizontal_concat = np.concatenate((detection, rendered_image),
                                             axis=1)

    rgb_frame = cv2.cvtColor(numpy_horizontal_concat, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)

    cv2.imshow('CVML-Pose', numpy_horizontal_concat)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Saving GIF file")
with imageio.get_writer(parent_dir +
                        "/video/real_no_detect.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(frame)
