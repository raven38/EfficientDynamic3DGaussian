#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import torch
import json
from PIL import Image
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, cameraList_from_camInfos_without_image, camera_to_JSON
from torch.utils import data
from utils.general_utils import PILtoTorch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):
    def __init__(self, cams, args):
        self.cams = cams
        self.args = args

    def __getitem__(self, index):
        cam_info = self.cams[index]
        # image = cam_info.image
        image = Image.open(cam_info.image_path)
        resized_image = torch.from_numpy(np.array(image)) / 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image.permute(2, 0, 1)
        else:
            resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=resized_image, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=cam_info.uid, time=cam_info.time, data_device=self.args.data_device)        

    def __len__(self):
        return len(self.cams)


class FlowDataset(data.Dataset):
    def __init__(self, cams, args):
        self.cams = cams
        self.args = args

    def __getitem__(self, index):
        cam_info = self.cams[index]
        # image = cam_info.image
        image = Image.open(cam_info.image_path)
        data_root = '/'.join(cam_info.image_path.split('/')[:-2])
        folder = cam_info.image_path.split('/')[-2]
        image_name =  cam_info.image_path.split('/')[-1]
        fwd_flow_path = os.path.join(data_root, f'{folder}_flow', f'{os.path.splitext(image_name)[0]}_fwd.npz')
        bwd_flow_path = os.path.join(data_root, f'{folder}_flow', f'{os.path.splitext(image_name)[0]}_bwd.npz')
        # print(fwd_flow_path, bwd_flow_path)
        if os.path.exists(fwd_flow_path):
            fwd_data = np.load(fwd_flow_path)
            fwd_flow = torch.from_numpy(fwd_data['flow'])
            fwd_flow_mask = torch.from_numpy(fwd_data['mask'])
        else:
            fwd_flow, fwd_flow_mask  = None, None
        if os.path.exists(bwd_flow_path):
            bwd_data = np.load(bwd_flow_path)
            bwd_flow = torch.from_numpy(bwd_data['flow'])
            bwd_flow_mask = torch.from_numpy(bwd_data['mask'])
        else:
            bwd_flow, bwd_flow_mask  = None, None
        
        # image = np.zeros((3, 128, 128))
        resized_image = torch.from_numpy(np.array(image)) / 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image.permute(2, 0, 1)
        else:
            resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                      image=resized_image, gt_alpha_mask=None,
                      image_name=cam_info.image_name, uid=cam_info.uid,
                      time=cam_info.time, data_device=self.args.data_device,
                      fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                      bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask)

    def __len__(self):
        return len(self.cams)


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.vis_cameras = {}
        self.use_loader = False

        if 'colmap_0' in args.source_path:
            scene_info = sceneLoadTypeCallbacks["Colmap2"](args.source_path, args.images, args.eval)
            self.use_loader = True
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found poses_bounds file, assuming DyNeRF data set!")
            scene_info = sceneLoadTypeCallbacks["DyNeRF"](args.source_path, args.eval)
            self.use_loader = True
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json, assuming HyperNeRF data set!")
            scene_info = sceneLoadTypeCallbacks["HyperNeRF"](args.source_path, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        self.time_delta = scene_info.time_delta

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if not self.use_loader:
            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        
        if self.use_loader:
            self.train_cameras[resolution_scales[0]] = FlowDataset(scene_info.train_cameras, args)
            self.test_cameras[resolution_scales[0]] = FlowDataset(scene_info.test_cameras, args)
            self.vis_cameras[resolution_scales[0]] = cameraList_from_camInfos_without_image(scene_info.vis_cameras, resolution_scales[0], args)
        else:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
                print("Loading Video Cameras")
                self.vis_cameras[resolution_scale] = cameraList_from_camInfos_without_image(scene_info.vis_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVisCameras(self, scale=1.0):
        return self.vis_cameras[scale]
    
