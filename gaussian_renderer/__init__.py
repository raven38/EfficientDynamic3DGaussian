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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    time = viewpoint_camera.time
    idx1, idx2, idx3 = 0, 1, 2
    # mask = torch.logical_and(pc.get_xyz[:, 3, 0] <= time, time <= pc.get_xyz[:, 4, 0])
    # print(pc.get_xyz[:, 3, 0], pc.get_xyz[:, 4, 0])
    # if time >= 0.5:
        # idx1, idx2, idx3 = 3, 4, 5        
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz[:, 0, :], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # print(viewpoint_camera.R, viewpoint_camera.T, viewpoint_camera.FoVx, viewpoint_camera.FoVy)

    static_util_iter = 3000
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    # print(pc.get_xyz[:, 0, :].mean(axis=0).data, pc.get_opacity.mean().item(), pc.get_rotation[:, 0, :].mean(axis=0).data, pc.get_scaling.mean().item()) # , pc.get_features.mean(axis=0))
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    L = pc.L

    basis = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time
    basis[::2] = torch.sin(basis[::2])
    basis[1::2] = torch.cos(basis[1::2])
    if itr != -1 and itr <= static_util_iter:
        means3D = pc.get_xyz[:, 0, :]
    else:
        means3D = pc.get_xyz[:, 0, :] + (pc.get_xyz[:, 1:2*L+1, :]*basis.unsqueeze(-1)).sum(1)

    means2D = screenspace_points[:]
    opacity = pc.get_opacity[:]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[:]

        if L == 0:
            rontations = pc.get_rotation[:, 0, :]
        else:
            if itr != -1 and itr <= static_util_iter:
                rotations = pc.get_rotation[:, idx1, :]
            else:
                rotations = pc.get_rotation[:, idx1, :] + pc.get_rotation[:, idx2, :]*time

        # rotations = pc.get_rotation[:, 0, :] + pc.get_rotation[:, 1, :]*torch.sin(torch.tensor(math.pi*time)) + pc.get_rotation[:, 2, :]*torch.cos(torch.tensor(math.pi*time)) + pc.get_rotation[:, 3, :]*torch.sin(2*torch.tensor(math.pi*time)) + pc.get_rotation[:, 4, :]*torch.cos(2*torch.tensor(math.pi*time)) + pc.get_rotation[:, 5, :]*torch.sin(4*torch.tensor(math.pi*time)) + pc.get_rotation[:, 6, :]*torch.cos(4*torch.tensor(math.pi*time)) + pc.get_rotation[:, 7, :]*torch.sin(8*torch.tensor(math.pi*time)) + pc.get_rotation[:, 8, :]*torch.cos(4*torch.tensor(math.pi*time))

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            if itr != -1 and itr <= static_util_iter:
                dir_pp = (pc.get_xyz[:, 0, :] - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            else:
                dir_pp = (pc.get_xyz[:, idx1, :] + (pc.get_xyz[:, 1:2*L+1, :]*basis.unsqueeze(-1)).sum(1) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))

            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # if itr % 400 == 0:
        # print(sum(radii > 0)/len(radii), len(radii))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_flow(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, itr=-1, time_delta=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    time = viewpoint_camera.time

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz[:, 0, :], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    static_util_iter = 3000
    # print(viewpoint_camera.R, viewpoint_camera.T, viewpoint_camera.FoVx, viewpoint_camera.FoVy)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    # print(pc.get_xyz[:, 0, :].mean(axis=0).data, pc.get_opacity.mean().item(), pc.get_rotation[:, 0, :].mean(axis=0).data, pc.get_scaling.mean().item()) # , pc.get_features.mean(axis=0))
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    L = pc.L
    basis = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time
    basis[::2] = torch.sin(basis[::2])
    basis[1::2] = torch.cos(basis[1::2])
    if itr != -1 and itr <= static_util_iter:
        means3D = pc.get_xyz[:, 0, :]
    else:
        means3D = pc.get_xyz[:, 0, :] + (pc.get_xyz[:, 1:2*L+1, :]*basis.unsqueeze(-1)).sum(1)

    # means3D = pc.get_xyz[:, 0, :] + pc.get_xyz[:, 1, :]*time
    means2D = screenspace_points[:]
    opacity = pc.get_opacity[:]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[:]

        if L == 0:
            rontations = pc.get_rotation[:, 0, :]
        else:
            if itr != -1 and itr <= static_util_iter:
                rotations = pc.get_rotation[:, 0, :]
            else:
                rotations = pc.get_rotation[:, 0, :] + pc.get_rotation[:, 1, :]*time


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    basis = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time
    basis[::2] = torch.sin(basis[::2])
    basis[1::2] = torch.cos(basis[1::2])
    t1 = (pc.get_xyz[:, 1:2*L+1, :]*basis.unsqueeze(-1)).sum(1)
    time2 = time + time_delta    
    basis2 = 2**torch.arange(0, L, device='cuda').repeat_interleave(2)*math.pi*time2
    basis2[::2] = torch.sin(basis[::2])
    basis2[1::2] = torch.cos(basis[1::2])
    t2 = (pc.get_xyz[:, 1:2*L+1, :]*basis2.unsqueeze(-1)).sum(1)
    flow = t2 - t1.detach()

    # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    # print(flow.mean(0), flow.stod(0), (flow<0).sum(0))


    focal_y = int(viewpoint_camera.image_height) / (2.0 * tanfovy)
    focal_x = int(viewpoint_camera.image_width) / (2.0 * tanfovx)
    tx, ty, tz = viewpoint_camera.world_view_transform[3, :3]
    viewmatrix = viewpoint_camera.world_view_transform.cuda()
    # amended by MobiusLqm
    t = torch.matmul(means3D, viewmatrix[:3, :3]) + viewmatrix[3, :3]    
    t = t.detach()
    flow[:, 0] = flow[:, 0] * focal_x / t[:, 2]  + flow[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
    flow[:, 1] = flow[:, 1] * focal_y / t[:, 2]  + flow[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])

    colors_precomp = flow
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D.detach(),
        means2D = means2D.detach(),
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity.detach(),
        scales = scales.detach(),
        rotations = rotations.detach(),
        cov3D_precomp = cov3D_precomp)
    # if itr % 400 == 0:
        # print(sum(radii > 0)/len(radii), len(radii))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(rendered_image.std(dim=(1,2)))
    # print(rendered_image.mean(dim=(1,2)))
    # print(rendered_image[0, 50, 100:110])
    # print(rendered_image[1, 50, 100:110])
    # print(rendered_image.std(dim=(1,2)))
    # print(rendered_image.mean(dim=(1,2)))    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
