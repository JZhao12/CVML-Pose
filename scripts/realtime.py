#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:18:15 2022

@author: bogdan

This script is used for realtime implementation.

The detector model is based on:
https://github.com/ylabbe/cosypose

Some lines of code are also based on:
https://github.com/ylabbe/cosypose

The real-time rendering is based on:
https://github.com/mmatl/pyrender

"""

# %% import modules
import sys
# please change the below line of code to your cosypose path
sys.path.append('/home/jianyu/CVML-Pose/cosypose/')

import yaml
import torch
import cosypose
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
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector)
from cosypose.integrated.detector import Detector
from torchvision import transforms
from pathlib import Path
from model import CVML_18, R_MLP, T_MLP
from engine import rotation_6d_to_matrix

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


# %% settings for camera and models
detector = getModel()
cap = cv2.VideoCapture(0)
bgr = (0, 255, 0)

parser = argparse.ArgumentParser(description='cvml-pose real-time')
parser.add_argument('--object', type=int, help='object to test')
args = parser.parse_args()

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
light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]),
                                  intensity=5)

fuze_trimesh = trimesh.load(MeshDir + 'obj_' + obj_id + '.ply')

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frames = []
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

    pred = inference(detector, frame)
    if pred is None:
        cv2.imshow('CVML-Pose', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    else:
        info = pred.infos
        bboxes = pred.bboxes
        deter = info["label"] == indexstr

        if deter.any():
            Labels = info.label.tolist()
            Confidences = info.score.tolist()

            idx = info.index[info["label"] == indexstr].tolist()
            if len(idx) > 1:
                idx = [idx[0]]

            bb = np.squeeze(bboxes[idx])
            bb = bb.numpy()

            X1, Y1, X2, Y2 = bb
            X1 = math.floor(X1)
            Y1 = math.floor(Y1)
            X2 = math.ceil(X2)
            Y2 = math.ceil(Y2)

            c = cv2.waitKey(1)
            if c == 27:
                break

            bbox_obj = [X1, Y1, X2-X1, Y2-Y1]
            x, y, w, h = bbox_obj

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

            resized = cv2.resize(ROI, (128, 128),
                                 interpolation=cv2.INTER_CUBIC)

            s = max(new_w, new_h)
            scale_factor = 128/s

# %% latent var
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
            XTest = np.concatenate((Test_Mean, Test_width, Test_height,
                                    scale_factor), axis=0)
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
            detection = cv2.rectangle(frame, (X1, Y1), (X2, Y2), bgr, 2)
            rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)

            numpy_horizontal = np.hstack((detection, rendered_image))
            numpy_horizontal_concat = np.concatenate((detection,
                                                      rendered_image), axis=1)

            rgb_frame = cv2.cvtColor(numpy_horizontal_concat,
                                     cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

            cv2.imshow('CVML-Pose', numpy_horizontal_concat)
            c = cv2.waitKey(1)
            if c == 27:
                break

cap.release()
cv2.destroyAllWindows()

print("Saving GIF file")
with imageio.get_writer(parent_dir +
                        "/video/real_all_new.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        writer.append_data(frame)