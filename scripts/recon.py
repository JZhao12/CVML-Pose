# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 19:11:07 2022

@author: Eddie

This script is used to generate the ground truth reconstruction images.

The off-screen rendering is based on this example:
https://pyrender.readthedocs.io/en/latest/examples/offscreen.html

"""
# %% import modules
import os
import argparse
import numpy as np
import trimesh
import pyrender
import json
import imageio
import glob
import cv2
import math
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# %% path, argparse, model information
parser = argparse.ArgumentParser(description='gt reconstruction images')
parser.add_argument('--object', type=int, help='object to reconstruct')
args = parser.parse_args()

parent_dir = os.getcwd()
pbr_save_dir = parent_dir + '/processed data/pbr_train/'
recon_save = parent_dir + '/processed data/recon_train/'
MeshDir = parent_dir + '/original data/lm/models/'

Num_models = len(glob.glob(MeshDir + '*.ply'))

# %% create folders
for i in range(1, Num_models+1):

    folderid = str(i).zfill(6)
    PBR_re_dir = recon_save + folderid
    for k in range(50):
        directory = str(k).zfill(6)
        path = os.path.join(PBR_re_dir, directory)
        if os.path.exists(path) is False:
            os.makedirs(path)

# %% Configurations of the rendering

view_width = 6400
view_height = 4800

light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=5)
camera = pyrender.PerspectiveCamera(yfov=2)
camera_pose = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
Ex_parameter = np.array([[0.0, 0.0, 0.0, 1.0]])

# %% start render each object, and save images

object_number = args.object
ob_id = '%06i' % (object_number)

gt_info = json.load(open(pbr_save_dir + ob_id + '_all.json'))
fuze_trimesh = trimesh.load(MeshDir + 'obj_' + ob_id + '.ply')

for column in range(len(gt_info)):

    ori_name = gt_info[column]['filename']
    filename = 'Re_' + ori_name

    RM = gt_info[column]['RM']
    RM = np.reshape(RM, (3, 3))

    TS_gt = gt_info[column]['TV']

    TS = np.array([[TS_gt[0]], [TS_gt[1]], [TS_gt[2]]])

    inter_pose = np.concatenate((RM, TS), axis=1)
    objectpose = np.concatenate((inter_pose, Ex_parameter), axis=0)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh,
                                      poses=objectpose, smooth=True)

    scene = pyrender.Scene(bg_color=([0, 0, 0]))
    scene.add(mesh)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    r = pyrender.OffscreenRenderer(viewport_width=view_width,
                                   viewport_height=view_height)
    rendered_image, _ = r.render(scene)

    for pixel in range(view_height):
        a = rendered_image[pixel, :]
        if a.any():
            y = pixel
            break

    for pixel in reversed(range(view_height)):
        a = rendered_image[pixel, :]
        if a.any():
            y2 = pixel
            break

    for pixel in range(view_width):
        a = rendered_image[:, pixel]
        if a.any():
            x = pixel
            break

    for pixel in reversed(range(view_width)):
        a = rendered_image[:, pixel]
        if a.any():
            x2 = pixel
            break

    w = x2-x
    h = y2-y

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
    right_trunc = np.minimum(new_x + new_w, rendered_image.shape[1])
    top_trunc = np.maximum(new_y, 0)
    bottom_trunc = np.minimum(new_y + new_h, rendered_image.shape[0])

    ROI = rendered_image[top_trunc:bottom_trunc, left_trunc:right_trunc]
    resized = resized = cv2.resize(ROI, (128, 128),
                                   interpolation=cv2.INTER_CUBIC)

    objectid = filename[3:9]
    folderid = filename[10:16]
    path = os.path.join(recon_save, objectid, folderid, filename)
    imageio.imwrite(path, resized)
    print(path)
    r.delete()