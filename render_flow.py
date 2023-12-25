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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_flow
import torchvision
from utils.general_utils import safe_state
from utils import flow_viz
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2 as cv
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, time_delta):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "flow")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if type(view) is list:
            view = view[0]
        rendering = render_flow(view, gaussians, pipeline, background, time_delta=time_delta)["render"]
        # print(view.world_view_transform)
        tx, ty, tz = view.world_view_transform[3, :3]
        # print(tx, ty, tz)
        # print(rendering.shape)     
        # flow = rendering.permute(1, 2, 0) @ torch.from_numpy(np.array([[1, 0, -tx/tz], [0, 1, -ty/tz]])).float().T.cuda()
        flow = rendering.permute(1, 2, 0) @ torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]])).float().T.cuda()
        flow = flow.cpu().numpy()
        # flow[..., 0] *= 1024
        # flow[..., 1] *= 1386
        # print(flow.mean(axis=(0, 1)), flow.std(axis=(0, 1)))
        # for i in range(20):
            # print(flow[40*i:40*(i+1)].mean(axis=(0, 1)))

        gt = view.original_image[0:3, :, :]
        # hsv = np.zeros((gt.shape[1], gt.shape[2], 3), dtype=np.uint8)
        # hsv[..., 1] = 255
        # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

        bgr = flow_viz.flow_to_image(flow)
        bgr = torch.from_numpy(bgr).permute(2, 0, 1).float()/255

        torchvision.utils.save_image(bgr, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.approx_l)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            if scene.use_loader:
                views = DataLoader(scene.getTrainCameras(), batch_size=1, shuffle=False, num_workers=16, collate_fn=list)
            else:
                views = scene.getTrainCameras()

            render_set(dataset.model_path, "train", scene.loaded_iter, views, gaussians, pipeline, background, scene.time_delta)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene.time_delta)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
