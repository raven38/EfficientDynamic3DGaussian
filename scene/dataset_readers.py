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
import os
import sys
import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.hyper_camera import Camera as HyperNeRFCamera
import natsort

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    vis_cameras: list
    nerf_normalization: dict
    ply_path: str
    time_delta: float

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo



def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras2(cam_extrinsics, cam_intrinsics, images_folder, near=0.1, far=10, startime=0, duration=300):
    cam_infos = []
      
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)
    
    sortedtotalcamelist =  natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in  range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i # map each cam with a number
     

    for idx, key in enumerate(cam_extrinsics): # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height//2
        width = intr.width//2

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]//2
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]//2
            focal_length_y = intr.params[1]//2
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
       
        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_"+str(startime), "colmap_{}".format(j), 1)
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path) # .resize((width, height))
            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, time=(j-startime)/duration)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, time=(j-startime)/duration)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo2(path, images, eval, llffhold=8, multiview=False, duration=60):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    images_folder = os.path.join(path, reading_dir)
    # images_folder = images_folder.replace('/colmap_0/images', '')
    parentdir = os.path.dirname(path)

    near = 0.01
    far = 100

    starttime = os.path.basename(path).split("_")[1] # colmap_0, 
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)
    

    cam_infos_unsorted = readColmapCameras2(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_folder, near=near, far=far, startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     

    if eval:
        train_cam_infos =  cam_infos[duration:] 
        test_cam_infos = cam_infos[:duration]
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        # assert len(uniquecheck) == 1 
        
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        # for testname in uniquecheck:
            # assert testname not in sanitycheck
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:2] #dummy

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    totalply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")
    
    if not os.path.exists(totalply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        thisbin_path = os.path.join(path, "sparse/0/points3D.bin").replace("colmap_"+ str(starttime), "colmap_" + str(starttime), 1)
        xyz, rgb, _ = read_points3D_binary(thisbin_path)

        xyz = xyz[:, None, :]
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 10, 3))], axis=1)
        xyz = np.concatenate([xyz, xyz, xyz, xyz], axis=1)
        assert xyz.shape[0] == rgb.shape[0]
        storePly(totalply_path, xyz, rgb)
    try:
        pcd = fetchPly(totalply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=totalply_path,
                           vis_cameras=test_cam_infos,
                           time_delta=1/300)
    return scene_info


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    time_length = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=idx/time_length)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']

    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("x")]
    y_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("y")]
    z_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("z")]
    x_names = sorted(x_names, key = lambda x: int(x.replace('x', '')))
    y_names = sorted(y_names, key = lambda y: int(y.replace('y', '')))
    z_names = sorted(z_names, key = lambda z: int(z.replace('z', '')))
    assert len(x_names) == len(y_names) == len(z_names)
    x = np.zeros((colors.shape[0], len(x_names)))
    y = np.zeros((colors.shape[0], len(y_names)))
    z = np.zeros((colors.shape[0], len(z_names)))
    for idx, attr_name in enumerate(x_names):
        x[:, idx] = np.asarray(plydata.elements[0][attr_name])
    for idx, attr_name in enumerate(y_names):
        y[:, idx] = np.asarray(plydata.elements[0][attr_name])
    for idx, attr_name in enumerate(z_names):
        z[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions = np.stack([x, y, z], axis=-1)
    assert len(positions.shape) == 3
    assert positions.shape[-1] == 3
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = []
    for t in range(xyz.shape[1]):
        dtype.extend([(f'x{t}', 'f4'), (f'y{t}', 'f4'), (f'z{t}', 'f4')])
    dtype = dtype + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz[:, 0, :])

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz.reshape(xyz.shape[0], -1), normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D_ours.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)

        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        xyz = xyz[:, None, :]
        xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 10, 3))], axis=1)
        xyz = np.concatenate([xyz, xyz, xyz, xyz], axis=1)
        assert xyz.shape[0] == rgb.shape[0]  
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1./len(train_cam_infos))
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        # if 'time' in frames[0]:
        #     times = np.array([frame['time'] for idx, frame in enumerate(frames)])
        #     time_idx = times.argsort()
        # else:
        #     time_idx = [0 for f in frames]
        # print(times)
        # print(time_idx)
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            time = frame['time'] if 'time' in frame else 0

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], time=time))
            
    return cam_infos


# https://github.com/albertpumarola/D-NeRF/blob/main/load_blender.py
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    # https://github.com/albertpumarola/D-NeRF/blob/main/load_blender.py
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def generateCamerasFromTransforms(path, transformsfile, extension=".png"):
    cam_infos = []

    # https://github.com/albertpumarola/D-NeRF/blob/main/load_blender.py
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])    

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
    cam_name = os.path.join(path, frames[0]["file_path"] + extension)        
    image_path = os.path.join(path, cam_name)
    image_name = Path(cam_name).stem
    image = Image.open(image_path)
    width = image.size[0]
    height = image.size[1]

    for idx, (c2w, time) in enumerate(zip(render_poses, render_times)):
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        fovy = focal2fov(fov2focal(fovx, width), height)
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                    image=None, image_path=None, image_name=None,
                                    width=width, height=height, time=time))
            
    return cam_infos


def init_random_points(ply_path):
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # time_length = max([c.time for c in train_cam_infos]) + 1
        # time_length = 2
        # xyz = np.random.random((num_pts, 1, 3)) * 2.6 - 1.3
        # xyz = np.tile(xyz, (1, time_length, 1))
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3)), np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)
        xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 16, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 3, 3)), np.ones((num_pts, 1, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3)), np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    pcd = fetchPly(ply_path)
    # except:
        # pcd = None
    return pcd

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    vis_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d_ours.ply")
    pcd = init_random_points(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/len(train_cam_infos))
    return scene_info


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses



def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


def readDynerfSceneInfo(path, eval):
    blender2opencv = np.eye(4)
    downsample = 2

    poses_arr = np.load(os.path.join(path, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    videos = glob.glob(os.path.join(path, "cam??"))
    videos = sorted(videos)
    assert len(videos) == poses_arr.shape[0]

    H, W, focal = poses[0, :, -1]
    focal = focal / downsample
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    poses, pose_avg = center_poses(
        poses, blender2opencv
    )  # Re-center poses so that the average is near the center.
    
    near_original = near_fars.min()
    scale_factor = near_original * 0.75
    near_fars /= (
        scale_factor  # rescale nearest plane so that it is at z = 4/3.
    )
    # print(scale_factor)
    poses[..., 3] /= scale_factor
    
    image_dirs = [video.replace('.mp4', '') for video in videos]
    val_index = [0]
    images = [sorted(glob.glob(os.path.join(d, "*.png")), key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))[:300] for d in image_dirs]
    train_cam_infos = []
    for idx, image_paths in enumerate(images):
        if idx in val_index:
            continue
        p = poses[idx]
        for image_path in image_paths:
            image_name = os.path.basename(image_path).split(".")[0]
            time  = float(image_name) / 300
            image = Image.open(image_path)
            uid = idx * 1000 + int(image_name)
            pose = np.eye(4)
            pose[:3, :] = p[:3, :]
            R = -pose[:3, :3]
            R[:, 0] = -R[:, 0]
            T = -pose[:3, 3].dot(R)
            height = image.height
            width = image.width
            FovY = focal2fov(focal, height)
            FovX = focal2fov(focal, width)
            # R = pose[:3, :3]
            # T  = pose[:3, 3]

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=width, height=height, time=time)
            train_cam_infos.append(cam_info)

    test_cam_infos = []
    for idx, image_paths in enumerate(images):
        if idx not in val_index:
            continue
        p = poses[idx]
        for image_path in image_paths:
            image_name = os.path.basename(image_path).split(".")[0]
            time  = float(image_name) / 300
            image = Image.open(image_path)
            uid = idx * 1000 + int(image_name)
            pose = np.eye(4)
            pose[:3, :] = p[:3, :]
            R = -pose[:3, :3]
            R[:, 0] = -R[:, 0]
            T = -pose[:3, 3].dot(R)
            # R = pose[:3, :3]
            # T  = pose[:3, 3]
            
            height = image.height
            width = image.width
            FovY = focal2fov(focal, height)
            FovX = focal2fov(focal, width)

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=width, height=height, time=time)
            test_cam_infos.append(cam_info)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    widht, height = train_cam_infos[0].width, train_cam_infos[0].height
    # Sample N_views poses for validation - NeRF-like camera trajectory.
    N_views = 120
    val_poses = get_spiral(poses, near_fars, N_views=N_views)
    val_times = torch.linspace(0.0, 1.0, val_poses.shape[0])
    vis_cam_infos = []
    for idx, (pose, time) in enumerate(zip(val_poses, val_times)):
        p = pose
        uid = idx
        pose = np.eye(4)
        pose[:3, :] = p[:3, :]
        R = -pose[:3, :3]
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)
        # R = pose[:3, :3]
        # T  = pose[:3, 3]
            
        FovY = focal2fov(focal, height)
        FovX = focal2fov(focal, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=None, image_name=None, width=width, height=height, time=time)
        vis_cam_infos.append(cam_info)


    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d_ours.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2_000 # 100_000
        print(f"Generating random point cloud ({num_pts})...")
        threshold = 3
        xyz_max = np.array([1.5*threshold, 1.5*threshold, -0*threshold])
        xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
        xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 16, 3))], axis=1)
        # xyz = np.concatenate([np.random.random((num_pts, 1, 3)) * 2.6 - 1.3, np.zeros((num_pts, 2, 3))], axis=1)

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras =vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/300)
    return scene_info



def readHypernerfCamera(uid, camera, image_path, time):
    height, width = int(camera.image_shape[0]), int(camera.image_shape[1])
    image_name = os.path.basename(image_path).split(".")[0]
    R = camera.orientation.T
    # T = camera.translation.T
    T = - camera.position @ R
    image = Image.open(image_path)    
    FovY = focal2fov(camera.focal_length, height)
    FovX = focal2fov(camera.focal_length, width)
    return CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                      image_path=image_path, image_name=image_name, width=width, height=height, time=time)    


def readHypernerfSceneInfo(path, eval):
    # borrow code from https://github.com/hustvl/TiNeuVox/blob/main/lib/load_hyper.py
    use_bg_points = False
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)    
    
    near = scene_json['near']
    far = scene_json['far']
    coord_scale = scene_json['scale']
    scene_center = scene_json['center']
    
    all_imgs = dataset_json['ids']
    val_ids  = dataset_json['val_ids']
    add_cam = False
    if len(val_ids) == 0:
        i_train = np.array([i for i in np.arange(len(all_imgs)) if (i%4 == 0)])
        i_test = i_train+2
        i_test = i_test[:-1,]
    else:
        add_cam = True
        train_ids = dataset_json['train_ids']
        i_test = []
        i_train = []
        for i in range(len(all_imgs)):
            id = all_imgs[i]
            if id in val_ids:
                i_test.append(i)
            if id in train_ids:
                i_train.append(i)

    print('i_train',i_train)
    print('i_test',i_test)
    all_cams = [meta_json[i]['camera_id'] for i in all_imgs]
    all_times = [meta_json[i]['time_id'] for i in all_imgs]
    max_time = max(all_times)
    all_times = [meta_json[i]['time_id']/max_time for i in all_imgs]
    selected_time = set(all_times)
    ratio = 0.5

    all_cam_params = []
    for im in all_imgs:
        camera = HyperNeRFCamera.from_json(f'{path}/camera/{im}.json')
        camera = camera.scale(ratio)
        camera.position = camera.position - scene_center
        camera.position = camera.position * coord_scale
        all_cam_params.append(camera)

    all_imgs = [f'{path}/rgb/{int(1/ratio)}x/{i}.png' for i in all_imgs]
    h, w = all_cam_params[0].image_shape
    if use_bg_points:
        with open(f'{path}/points.npy', 'rb') as f:
            points = np.load(f)
        bg_points = (points - scene_center) * coord_scale
        bg_points = torch.tensor(bg_points).float()    

    train_cam_infos = [readHypernerfCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_train]
    test_cam_infos = [readHypernerfCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in i_test]

    vis_cam_infos = [readHypernerfCamera(i, all_cam_params[i], all_imgs[i], all_times[i]) for i in np.argsort(all_cams, kind='stable')]

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "points3d_ours.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000
    #     print(f"Generating random point cloud ({num_pts})...")
    #     threshold = 3
    #     xyz_max = np.array([1.5*threshold, 1.5*threshold, 1.5*threshold])
    #     xyz_min = np.array([-1.5*threshold, -1.5*threshold, -1.5*threshold])        
    #     xyz = np.concatenate([(np.random.random((num_pts, 1, 3)))* (xyz_max-xyz_min) + xyz_min, np.zeros((num_pts, 10, 3))], axis=1)

    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)

    # pcd = fetchPly(ply_path)

    ply_path = os.path.join(path, "points.npy")
    xyz = np.load(ply_path, allow_pickle=True)
    xyz = (xyz - scene_center) * coord_scale
    xyz = xyz.astype(np.float32)[:, None, :]
    xyz = np.concatenate([xyz, np.zeros((xyz.shape[0], 12, 3))], axis=1)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))   

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           vis_cameras=vis_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, time_delta=1/max_time)
    return scene_info



sceneLoadTypeCallbacks = {
    "Colmap2": readColmapSceneInfo2,
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DyNeRF": readDynerfSceneInfo,
    "HyperNeRF": readHypernerfSceneInfo,
}
