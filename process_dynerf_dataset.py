import concurrent.futures
import gc
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    while video_frames.isOpened():
        ret, video_frame = video_frames.read()
        if ret:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_frame = Image.fromarray(video_frame)
            if downsample != 1.0:
                img = video_frame.resize(img_wh, Image.LANCZOS)
            img = transform(img)
            video_data_save[count] = img.view(3, -1).permute(1, 0)
            count += 1
        else:
            break
    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] * img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs




class Neural3D_NDC_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=0.75,
        eval_step=1,
        eval_index=0,
        sphere_scale=1.0,
    ):
        self.img_wh = (
            int(1024 / downsample),
            int(768 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.split = split
        self.downsample = 2704 / self.img_wh[0]
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False

        self.load_meta()
        print("meta data loaded")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        assert len(videos) == poses_arr.shape[0]

        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        poses, pose_avg = center_poses(
            poses, self.blender2opencv
        )  # Re-center poses so that the average is near the center.

        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75
        self.near_fars /= (
            scale_factor  # rescale nearest plane so that it is at z = 4/3.
        )
        poses[..., 3] /= scale_factor

        # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 120
        self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)

        W, H = self.img_wh
        self.directions = torch.tensor(
            get_ray_directions_blender(H, W, self.focal)
        )  # (H, W, 3)

        if self.split == "train":
            # Loading all videos from this dataset requires around 50GB memory, and stack them into a tensor requires another 50GB.
            # To save memory, we allocate a large tensor and load videos into it instead of using torch.stack/cat operations.
            all_times = []
            all_rays = []
            count = 300

            for index in range(0, len(videos)):
                if (
                    index == self.eval_index
                ):  # the eval_index(0 as default) is the evaluation one. We skip evaluation cameras.
                    continue

                video_times = torch.tensor([i / (count - 1) for i in range(count)])
                all_times += [video_times]

                rays_o, rays_d = get_rays(
                    self.directions, torch.FloatTensor(poses[index])
                )  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
                all_rays += [torch.cat([rays_o, rays_d], 1)]
                print(f"video {index} is loaded")
                gc.collect()

            # load all video images
            all_imgs = process_videos(
                videos,
                self.eval_index,
                self.img_wh,
                self.downsample,
                self.transform,
                num_workers=8,
            )
            all_times = torch.stack(all_times, 0)
            all_rays = torch.stack(all_rays, 0)
            breakpoint()
            print("stack performed")
            N_cam, N_time, N_rays, C = all_imgs.shape
            self.image_stride = N_rays
            self.cam_number = N_cam
            self.time_number = N_time
            self.all_rgbs = all_imgs
            self.all_times = all_times.view(N_cam, N_time, 1)
            self.all_rays = all_rays.reshape(N_cam, N_rays, 6)
            self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)
            self.global_mean_rgb = torch.mean(all_imgs, dim=1)
        else:
            index = self.eval_index
            video_imgs = []
            video_frames = cv2.VideoCapture(videos[index])
            while video_frames.isOpened():
                ret, video_frame = video_frames.read()
                if ret:
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_frame = Image.fromarray(video_frame)
                    if self.downsample != 1.0:
                        img = video_frame.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    video_imgs += [img.view(3, -1).permute(1, 0)]
                else:
                    break
            video_imgs = torch.stack(video_imgs, 0)
            video_times = torch.tensor(
                [i / (len(video_imgs) - 1) for i in range(len(video_imgs))]
            )
            video_imgs = video_imgs[0 :: self.eval_step]
            video_times = video_times[0 :: self.eval_step]
            rays_o, rays_d = get_rays(
                self.directions, torch.FloatTensor(poses[index])
            )  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
            all_rays = torch.cat([rays_o, rays_d], 1)
            gc.collect()
            N_time, N_rays, C = video_imgs.shape
            self.image_stride = N_rays
            self.time_number = N_time
            self.all_rgbs = video_imgs.view(-1, N_rays, 3)
            self.all_rays = all_rays
            self.all_times = video_times
            self.all_rgbs = self.all_rgbs.view(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)

