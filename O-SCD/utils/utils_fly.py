#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
import math
import time
import cv2
import torch.nn.functional as F
import os

def parse_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def get_image_names(in_folder, image_extensions=[".jpg", ".png", ".jpeg"]):
    return [
        f
        for f in os.listdir(in_folder)
        if os.path.splitext(f)[-1].lower() in image_extensions
    ]

def psnr(img1, img2):
    return 10 * torch.log10(1 / F.mse_loss(img1, img2)).item()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_lapla_norm(img, kernel):
    laplacian_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda", dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(laplacian, ord=1, dim=1, keepdim=True)
    laplacian_norm[..., :, 0] = 0
    laplacian_norm[..., :, -1] = 0
    laplacian_norm[..., 0, :] = 0
    laplacian_norm[..., -1, :] = 0
    return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)



def increment_runtime(runtime, start_time):
    # torch.cuda.synchronize()
    runtime[0] += time.time() - start_time
    runtime[1] += 1


C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


## Camera/triangulation/projection functions
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def depth2points(uv, depth, f, centre):
    xyz = torch.cat([(uv[..., :2] - centre) / f, torch.ones_like(uv[..., 0:1])], dim=-1)
    return depth * xyz


def reproject(uv, depth, f, centre, relR, relt):
    xyz = depth2points(uv, depth, f, centre)
    xyz = xyz @ relR.T + relt
    return pts2px(xyz, f, centre)


def make_torch_sampler(uv, width, height):
    """
    Converts OpenCV UV coordinates to a sampler for torch's grid_sample.
    To be used with align_corners=True
    """
    sampler = uv.clone()  # + 0.5
    sampler[..., 0] = sampler[..., 0] * (2.0 / (width - 1)) - 1.0
    sampler[..., 1] = sampler[..., 1] * (2.0 / (height - 1)) - 1.0
    return sampler


def sample(map, uv, width, height):
    sampler = make_torch_sampler(uv, width, height)
    return F.grid_sample(map, sampler, mode="bilinear", align_corners=True)


def pts2px(xyz, f, centre):
    return f * xyz[..., :2] / xyz[..., 2:3] + centre


def sixD2mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def mtx2sixD(R):
    return R[..., :2].clone()


## Visualization functions
def display_matches(mkpts1, mkpts2, img1, img2, scale=1, match_step=1, indices=None):
    image1 = img1.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    image2 = img2.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    if indices is not None:
        mkpts1 = mkpts1[indices]
        mkpts2 = mkpts2[indices]
    matched_mkptsi_np = mkpts1[::match_step].cpu().float().numpy()
    matched_mkptsj_np = mkpts2[::match_step].cpu().float().numpy()
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in matched_mkptsi_np]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in matched_mkptsj_np]
    mask_np = (
        ((mkpts1 != -1).all(dim=-1) * (mkpts2 != -1).all(dim=-1))[::match_step]
        .cpu()
        .numpy()
    )
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask_np)) if mask_np[i]]
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    if scale != 1:
        img_matches = cv2.resize(img_matches, (0, 0), fx=scale, fy=scale)
    cv2.imshow("matches_img", img_matches[..., ::-1])
    cv2.waitKey()


@torch.no_grad()
def draw_poses(image, view_matrix, view_fovx, scale, cam_width, cam_height, Rts, cam_f, color):
    """
    Overlay the camera frustums on the np image

    Args:
       image (np.ndarray): The image to draw on
       view_matrix (torch.Tensor): The point of view to render from
       view_fov (float): The field of view to render with
       scale (float): The scale of the drawn poses
       cam_width (int): The width of the image to draw the frustums
       cam_height (int): The height of the image to draw the frustums
       Rts (torch.Tensor): The camera poses to draw (camera to world)
       cam_f (float): The focal length of the poses to draw
    Returns:
       image (np.ndarray): The image with the frustums drawn on
    """
    if len(Rts) > 0:
        # Rendering options
        width, height = image.shape[1], image.shape[0]
        f = fov2focal(view_fovx, width)
        centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')

        # Camera intrinsics to draw
        cam_centre = torch.tensor([(cam_width - 1) / 2, (cam_height - 1) / 2], device='cuda')
        # Make a 3D frustum using intrinsics
        origin = torch.tensor([0, 0, 0], device='cuda')
        corners2d = torch.tensor([[0, 0], [cam_width, 0], [cam_width, cam_height], [0, cam_height]], device='cuda')
        corners3d = depth2points(corners2d, scale, cam_f, cam_centre)
        # Duplicate and transform frustums for each pose
        cams_verts = torch.cat([origin.unsqueeze(0), corners3d], dim=0)
        n_cams = Rts.shape[0]
        cams_verts = torch.bmm((cams_verts - Rts[:n_cams, None, :3, 3]), Rts[:n_cams, :3, :3])
        cams_verts_view = (cams_verts @ view_matrix[:3, :3] + view_matrix[3:4, :3])
        cams_verts_2d = pts2px(cams_verts_view, f, centre).view(n_cams, -1, 2)
        # Out of view check
        valid_cams = (cams_verts_view[..., 2] > 0).all(dim=-1)
        cams_verts_2d = cams_verts_2d[valid_cams]

        # Draw frustums on the image
        draw_order = torch.tensor([1, 2, 0, 3, 4, 0, 1, 4, 3, 2], device="cuda")
        cams_verts_2d = cams_verts_2d[..., draw_order, :]
        image = cv2.polylines(
            image,
            cams_verts_2d.detach().cpu().numpy().astype(int),
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return image


@torch.no_grad()
def draw_anchors(image, view_matrix, view_fovx, scale, anchors, anchor_weights=[]):
    coords = [
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
    ]
    draw_order = [0,4,6,2,0,1,5,7,3,1,5,4,6,7,3,2,0]
    centred_cube_verts = scale * torch.tensor([coords[i] for i in draw_order], device='cuda')

    # Rendering options
    width, height = image.shape[1], image.shape[0]
    f = fov2focal(view_fovx, width)
    centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')

    if len(anchors) != len(anchor_weights):
        anchor_weights = np.zeros(len(anchors))

    for anchor_weight, anchor in zip(anchor_weights, anchors):
        cube_verts = centred_cube_verts + anchor.position
        cube_vert_view = cube_verts @ view_matrix[:3, :3] + view_matrix[3:4, :3]
        if cube_vert_view[..., 2].min() > 0:
            cube_verts_2d = pts2px(cube_vert_view, f, centre)
            verts_2d = cube_verts_2d.cpu().numpy().astype(int)[None]
            cv2.polylines(image, verts_2d, isClosed=False, color=(anchor_weight * 255, 0, (1-anchor_weight)*255), thickness=2, lineType=cv2.LINE_AA)
    return image


def get_transform_mean_up_fwd(input, target, w_scale):
    """
    Get the transform that aligns input poses to target mean position, up and forward vectors.
    This appears more stable than Procrustes analysis.

    The input and target are both [N,4,4] transforms from world to camera.
    We want to:
      - match the mean position (camera center) of 'input' to that of 'target'
      - align the average "up" direction from 'input' to the average "up" direction of 'target'
      - align the average "forward" direction from 'input' to the average "forward" direction of 'target'

    """
    inv_input = torch.linalg.inv(input)
    inv_target = torch.linalg.inv(target)
    center_input = inv_input[:, :3, 3]
    center_target = inv_target[:, :3, 3]

    # Compute average up and forward vectors in world coords
    up_input_avg = inv_input[:, :3, 1].mean(dim=0)
    up_target_avg = inv_target[:, :3, 1].mean(dim=0)
    fwd_input_avg = inv_input[:, :3, 2].mean(dim=0)
    fwd_target_avg = inv_target[:, :3, 2].mean(dim=0)

    # Normalize these average directions to get unit vectors
    up_input_avg = up_input_avg / up_input_avg.norm()
    up_target_avg = up_target_avg / up_target_avg.norm()
    fwd_input_avg = fwd_input_avg / fwd_input_avg.norm()
    fwd_target_avg = fwd_target_avg / fwd_target_avg.norm()

    # Input basis
    right_input = torch.cross(up_input_avg, fwd_input_avg)
    right_input = right_input / right_input.norm()

    R_in = torch.stack([right_input, up_input_avg, fwd_input_avg], dim=1)

    # Target basis
    right_target = torch.cross(up_target_avg, fwd_target_avg)
    right_target = right_target / right_target.norm()

    R_tgt = torch.stack([right_target, up_target_avg, fwd_target_avg], dim=1)

    # This rotation aligns the input basis to target basis
    R = R_tgt @ R_in.transpose(0, 1)

    # This scale aligns the input center to target center
    center_input_mean = center_input.mean(dim=0)
    center_target_mean = center_target.mean(dim=0)
    if w_scale:
        s_input = ((center_input - center_input_mean)**2).sum(dim=-1).mean().sqrt()
        s_target = ((center_target - center_target_mean)**2).sum(dim=-1).mean().sqrt()
        s = s_target / s_input
    else:
        s = 1.0

    # This translation aligns the input center to target center
    t = center_target_mean - R @ center_input_mean * s

    return R, t, s


def align_mean_up_fwd(input, target, w_scale=False):
    """
    Align input poses to target mean position, up and forward vectors.

    Returns:
      A set of [N,4,4] transforms, which are the aligned poses of 'input'.
    """

    R, t, s = get_transform_mean_up_fwd(input, target, w_scale)
    inv_input = torch.linalg.inv(input)
    inv_input[:, :3, :3] = R @ inv_input[:, :3, :3]
    inv_input[:, :3, 3] = (R @ inv_input[:, :3, 3:4]).squeeze(-1) * s + t[None]

    return torch.linalg.inv(inv_input)

## Pose alignment and evaluation functions
def align_poses(input, target, w_scale=True):
    """Align input poses to target using Procrustes analysis on camera centers"""
    center_input = torch.linalg.inv(input)[:, :3, 3]
    center_target = torch.linalg.inv(target)[:, :3, 3]
    t0, t1, s0, s1, R = procrustes_analysis(center_target, center_input, w_scale)
    center_aligned = (center_input - t1) / s1 @ R.t() * s0 + t0
    R_aligned = input[:, :3, :3] @ R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    aligned = torch.eye(4, device=input.device).repeat(input.shape[0], 1, 1)
    aligned[:, :3, :3] = R_aligned[:, :3, :3]
    aligned[:, :3, 3] = t_aligned
    return aligned


## From https://github.com/chenhsuanlin/bundle-adjusting-NeRF
# BARF: Bundle-Adjusting Neural Radiance Fields
# Copyright (c) 2021 Chen-Hsuan Lin
# Under the MIT License.
# Modified to interface with our pose format 
def rotation_distance(R1, R2, eps=1e-9):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = (
        ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    )  # numerical stability near -1/+1
    return angle


def procrustes_analysis(X0, X1, w_scale=True):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    if w_scale:
        s0 = (X0c**2).sum(dim=-1).mean().sqrt()
        s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    else:
        s0, s1 = 1, 1
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    return t0[0], t1[0], s0, s1, R


