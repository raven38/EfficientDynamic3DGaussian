import os
import torch
from torch import nn

import sys
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
import numpy as np
import quaternion

def save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, path):
    mkdir_p(os.path.dirname(path))

    l = []
    for t in range(xyz.shape[1]):
        l.extend([f'x{t:03}', f'y{t:03}', f'z{t:03}'])

    l.extend(['nx', 'ny', 'nz'])
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]*features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]*features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[-1]):
        for t in range(rotation.shape[-2]):
            l.append(f'rot_{t:03}_{i}')
    dtype_full = [(attribute, 'f4') for attribute in l]
    xyz = xyz.detach().flatten(start_dim=1).cpu().numpy()
    normals = np.zeros((xyz.shape[0], 3))
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().flatten(start_dim=1).cpu().numpy()
    print(xyz.shape, f_dc.shape, f_rest.shape, opacities.shape, scaling.shape, rotation.shape)

    print(len(l))
    print(xyz.shape[0], len(dtype_full))
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(path):
    plydata = PlyData.read(path)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("x")]
    y_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("y")]
    z_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("z")]
    x_names = sorted(x_names, key = lambda x: int(x.replace('x', '')))
    y_names = sorted(y_names, key = lambda y: int(y.replace('y', '')))
    z_names = sorted(z_names, key = lambda z: int(z.replace('z', '')))
    assert len(x_names) == len(y_names) == len(z_names)
    x = np.zeros((opacities.shape[0], len(x_names)))
    y = np.zeros((opacities.shape[0], len(y_names)))
    z = np.zeros((opacities.shape[0], len(z_names)))
    for idx, attr_name in enumerate(x_names):
        x[:, idx] = np.asarray(plydata.elements[0][attr_name])
    for idx, attr_name in enumerate(y_names):
        y[:, idx] = np.asarray(plydata.elements[0][attr_name])
    for idx, attr_name in enumerate(z_names):
        z[:, idx] = np.asarray(plydata.elements[0][attr_name])
    xyz = np.stack((x, y, z),  axis=-1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: (int(x.split('_')[-1]), int(x.split('_')[-2])))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = rots.reshape(xyz.shape[0], -1, 4)

    xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
    features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
    scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
    rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
    return xyz, features_dc, features_rest, opacity, scaling, rotation


ply_path1 = sys.argv[1]
ply_path2 = sys.argv[2]
new_ply_path = sys.argv[3]
offset_x, offset_y, offset_z = sys.argv[4:7]
offset_x, offset_y, offset_z = float(offset_x), float(offset_y), float(offset_z)

xyz1, features_dc1, features_rest1, opacity1, scaling1, rotation1 = load_ply(ply_path1)
xyz2, features_dc2, features_rest2, opacity2, scaling2, rotation2 = load_ply(ply_path2)


xyz2 = torch.cat([xyz2, torch.zeros((xyz2.shape[0], xyz1.shape[1] - xyz2.shape[1], xyz2.shape[2])).to('cuda')], dim=1)
rotation2 = torch.cat([rotation2, torch.zeros((rotation2.shape[0], rotation1.shape[1] - rotation2.shape[1], rotation2.shape[2])).to('cuda')], dim=1)

c2w = np.array([
                [
                    0.8865219950675964,
                    -0.30261921882629395,
                    0.3500004708766937,
                    1.4108970165252686
                ],
                [
                    0.4626864194869995,
                    0.579828143119812,
                    -0.670612096786499,
                    -2.7033238410949707
                ],
                [
                    -1.4901161193847656e-08,
                    0.756452739238739,
                    0.6540482640266418,
                    2.6365528106689453
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])
c2w = np.array([[ 0.888224  , -0.28239794,  0.36236657, 1.4108970165252686],
       [ 0.4588151 ,  0.50513121, -0.7309796, -2.7033238410949707 ],
             [ 0.02338447,  0.81553287,  0.5782381, 2.6365528106689453 ],
             [0, 0, 0, 1]])
c2w[:3, :3] = np.array([[ 0.83427332, -0.19049942,  0.51739541],
       [ 0.53440564,  0.51025918, -0.67382949],
       [-0.13564163,  0.83865698,  0.52749958]])
c2w[:3, :3] = np.array([[ 0.99391392,  0.01907107,  0.10849614],
       [ 0.09653589,  0.32365205, -0.94123864],
       [-0.05306543,  0.94598396,  0.31984124]])
c2w[:3, 1:3] *= -1
# get the world-to-camera transform and set R, T

w2c = np.linalg.inv(c2w)
print(xyz2[:, 0, :].mean(0), xyz2[:, 0, :].var(0))
xyz2 = xyz2 @ torch.tensor(w2c[:3, :3]).to('cuda').float()
xyz2 = xyz2 / 2.4

# quaternions = quaternion.from_rotation_matrix(w2c, nonorthogonal=True)
# rotation2[:, 0, :] = torch.tensor(quaternion.as_float_array(quaternion.from_rotation_matrix(quaternion.as_rotation_matrix(quaternion.as_quat_array(rotation2[:, 0, :].cpu().numpy())) @ w2c[:3, :3], nonorthogonal=True))).to('cuda').float()
# rotation2[:, 0, :] = torch.tensor(quaternion.as_float_array(quaternion.as_quat_array(rotation2[:, 0, :].cpu().numpy()) * quaternions)).to('cuda')
print(scaling2[:, :].mean(0), scaling2[:, :].var(0), scaling2[:, :].max(0)[0], scaling2[:, :].min(0)[0])
# scaling2 = scaling2 @ torch.tensor(np.linalg.inv(w2c[:3, :3])).to('cuda').float()
# rotation2[:, 1, :] /= 2
# scaling2[:, 0] = scaling2[:, :] @ torch.tensor(w2c[:3, 0]).to('cuda').float()
# scaling2[:, 1], scaling2[:, 0] = scaling2[:, 0], scaling2[:, 1]
scaling2 *= 1.4


# R = quaternion.as_rotation_matrix(quaternion.as_quat_array(rotation2[:, 0, :].cpu().numpy())) @ w2c[:3, :3]
# print(R.shape)
# inv_R = np.linalg.inv(R)

# print(np.diag(scaling2.cpu().numpy()).shape)
# print((torch.diag_embed(scaling2).cpu().numpy() @ R @ w2c[:3, :3]).shape, np.linalg.inv(R @ w2c[:3, :3]).shape)
# scaling2 = np.matmul(torch.diag_embed(scaling2).cpu().numpy()@ R @ w2c[:3, :3], np.linalg.inv(R @ w2c[:3, :3]))
# print(scaling2.shape)
# print(scaling2[:, :].mean(0), scaling2[:, :].var(0), scaling2[:, :].max(0)[0], scaling2[:, :].min(0)[0])
# scaling2 = torch.diagonal(torch.tensor(scaling2).to('cuda'), dim1=-2, dim2=-1)


offset = torch.tensor(np.array([offset_x, offset_y, offset_z])).to('cuda')
xyz2[:, 0, :] = xyz2[:, 0, :] + offset
print(xyz2[:, 0, :].mean(0), xyz2[:, 0, :].var(0))

xyz = torch.cat([xyz1, xyz2], dim=0)
features_dc = torch.cat([features_dc1, features_dc2], dim=0)
features_rest = torch.cat([features_rest1, features_rest2], dim=0)
opacity = torch.cat([opacity1, opacity2], dim=0)
scaling = torch.cat([scaling1, scaling2], dim=0)
rotation  = torch.cat([rotation1, rotation2], dim=0)




save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, new_ply_path)

