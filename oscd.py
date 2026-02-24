import os
import cv2
import numpy as np
import torch
torch.backends.cuda.benchmark = True
import random
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render, render_change
from scene import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import Namespace
from arguments.config_args import parse_args
from transformers import Sam2Model
from dataloaders.image_dataset import ImageDataset
# from dataloaders.stream_dataset import StreamDataset
from poses.feature_detector import Detector
from poses.matcher import Matcher
from poses.pose_initializer import PoseInitializer, get_reference_keyframes
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.keyframe import Keyframe
from scene.cameras import Camera
import json
from utils.camera_utils import camera_to_JSON
import warnings 
warnings.filterwarnings("ignore")

def generate_candidate_map(original_image, image_rgb, model, patch_size, height, width):
    with torch.no_grad():
        ssim_map = ssim(image_rgb, original_image, window_size=11, map=True).mean(dim=0)
        l1_image = torch.abs(image_rgb - original_image).mean(dim=0)
        delta_im = 0.8 * l1_image + 0.2 * (1 - ssim_map)
        vmin, vmax = delta_im.min(), delta_im.max()
        delta_im = (delta_im - vmin) / (vmax - vmin + 1e-8)
    
    image_rgb_1024 = torch.nn.functional.interpolate(
        image_rgb.unsqueeze(0), size=(1024, 1024), mode='nearest'
    ).squeeze()
    original_image_1024 = torch.nn.functional.interpolate(
        original_image.unsqueeze(0), size=(1024, 1024), mode='nearest'
    ).squeeze()

    input_stack = torch.stack([image_rgb_1024, original_image_1024]).half()
    
    with torch.inference_mode():
        outputs = model.get_image_embeddings(input_stack)[-1]  # (2, 256, 64, 64)

    patch_features_flat = outputs
    
    diff_map = (patch_features_flat[0] - patch_features_flat[1]).abs().mean(dim=0)
    
    diff_vmin, diff_vmax = diff_map.min(), diff_map.max()
    diff_map = (diff_map - diff_vmin) / (diff_vmax - diff_vmin + 1e-8)

    diff_map = torch.nn.functional.interpolate(
        diff_map.unsqueeze(0).unsqueeze(0), size=(height, width), 
        mode='bilinear', align_corners=False
    ).squeeze()
    
    diff_map = (diff_map - vmin) / (vmax - vmin + 1e-8)
    diff_map = diff_map.clamp(0, 1)

    delta = (delta_im + diff_map).unsqueeze(0)

    return delta

def main(dataset: Namespace, opt : Namespace, pipe: Namespace, args: Namespace):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    gaussians_rgb = GaussianModel(dataset.sh_degree, dataset.sh_degree)
    gaussians_rgb.load_ply(os.path.join(args.source_path, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
    gaussians_rgb.training_setup(opt)

    gaussians_change = GaussianModel(dataset.sh_degree, 0)
    gaussians_change.load_ply_change(os.path.join(args.source_path, "reference_reconstruction", "point_cloud", "iteration_30000", "point_cloud.ply"))
    gaussians_change.training_setup_change(opt)

    # dataset = StreamDataset(args.source_path, args.downsampling)
    dataset = ImageDataset(args, instance='inf')

    height, width = dataset.get_image_size()

    reference_dataset = ImageDataset(args, instance='ref')

    max_error = max(args.match_max_error * width, 1.5)
    matcher = Matcher(args.fundmat_samples, max_error)
    triangulator = Triangulator(
        args.num_kpts, args.num_keyframes_for_triangulation, max_error
    )
    pose_initializer = PoseInitializer(
        width, height, triangulator, matcher, max_error, args
    )
    dense_extractor = DenseExtractor(width, height)
    detector = Detector(args.num_kpts, width, height)

    renders_path = os.path.join(args.model_path, "renders")
    os.makedirs(renders_path, exist_ok=True)
    os.makedirs(os.path.join(renders_path, "change_mask"), exist_ok=True)
    if args.refine:
            os.makedirs(os.path.join(renders_path, "change_mask_refined"), exist_ok=True)


    reference_keyframes = []

    pbar_ref = tqdm(range(0, len(reference_dataset)), desc="Loading reference keyframes", disable=True)
    for frameID in pbar_ref:
        image, info = reference_dataset.getnext()
        desc_kpts = detector(image)

        Rt = info["Rt"]
        f = info["focal"]
        Fovx = info["FovX"]
        Fovy = info["FovY"]
        
        keyframe = Keyframe(
            image,
            info,
            desc_kpts,
            Rt,
            frameID,
            f,
            dense_extractor,
            triangulator,
            args,
        )
        reference_keyframes.append(keyframe)

    n_ref_cams = len(reference_keyframes)
    pose_initializer.init_focal(f)
    for i in range(n_ref_cams):
            for j in range(i + 1, n_ref_cams):
                _ = matcher(
                    reference_keyframes[i].desc_kpts,
                    reference_keyframes[j].desc_kpts,
                    remove_outliers= True,
                    update_kpts_flag="inliers", kID=i, kID_other=j)

    for frameID in pbar_ref:
        keyframe = reference_keyframes[frameID]
        keyframe.update_3dpts(reference_keyframes)

    cam_centers = [v.get_centre() for v in reference_keyframes]
    pbar_ref.close()

    viewpoints = []
    change_masks = {}

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").half().to("cuda")   
    model.get_image_embeddings = torch.compile(model.get_image_embeddings, mode='max-autotune')
    # model.get_image_embeddings = torch.compile(model.get_image_embeddings, mode='reduce-overhead')
    # model.get_image_embeddings = torch.compile(model.get_image_embeddings, mode='default')


    
    patch_size = 14

    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
    total_iterations = 0
    ema_loss_for_log = 0.0

    dummy_image = torch.zeros((3, height, width), device="cuda").half()
    for dummy_id in range(min(10, len(dataset))):
        with torch.no_grad():
            candidate_map = generate_candidate_map(dummy_image, dummy_image, model, patch_size, height, width)


    pbar_inf = tqdm(range(0, len(dataset)), desc="Running OSCD")
    
    for frameID in pbar_inf:

        image, info = dataset.getnext()
        desc_kpts = detector(image)
        prev_keyframes = get_reference_keyframes(
            n=args.num_reference_keyframes, keyframes=reference_keyframes, matcher=matcher, desc_kpts=desc_kpts
        )
        Rt, _ = pose_initializer.init_inference_pose(
            prev_keyframes, desc_kpts, frameID, False, image
        )

        R = Rt[:3, :3].cpu().numpy()
        t = Rt[:3, 3].cpu().numpy()
        uid = f"{frameID:04d}"
        image_name = info["name"].split(".")[0]

        view = Camera(colmap_id=uid, R=np.transpose(R), T=t, FoVx=Fovx, FoVy=Fovy, 
                     image=image[:3, ...], gt_alpha_mask=None,
                     image_name=image_name, uid=uid)

        cam_centers.append(view.camera_center)


        with torch.no_grad():
            render_pkg = render(view, gaussians_rgb, pipe, background)
            image_rgb = render_pkg["render"]
            original_image = view.original_image[:3, ...]
            candidate_map = generate_candidate_map(original_image, image_rgb, model, patch_size, height, width)
            view.candidate_map = candidate_map.detach().clone()
            viewpoints.append(view)


        for iteration in range(16):
            total_iterations += 1
            if np.random.rand() > 0.33:
                keyframe_idx = randint(0, len(viewpoints)-1)
            else:
                keyframe_idx = -1

            viewpoint = viewpoints[keyframe_idx]
            gaussians_change.update_learning_rate(total_iterations)

            render_pkg_change = render_change(viewpoint, gaussians_change, pipe, background)
            change_mask, viewspace_point_tensor, visibility_filter, radii = render_pkg_change["render"], render_pkg_change["viewspace_points"], render_pkg_change["visibility_filter"], render_pkg_change["radii"]

            gt_change = viewpoint.candidate_map
            change_mask = torch.sigmoid(change_mask.mean(dim=0, keepdim=True))

            d_loss = (gt_change*(1.0-change_mask)).mean()
            d_reg = torch.log(change_mask.mean()**2 + 1.0)

            loss = d_loss + d_reg
            loss.backward()

            gaussians_change.optimizer.step()
            gaussians_change.optimizer.zero_grad(set_to_none = True)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if total_iterations % 5 == 0:
                    pbar_inf.set_postfix({
                        "it": total_iterations,
                        "loss": f"{ema_loss_for_log:.4f}",
                    })

                gaussians_change.max_radii2D[visibility_filter] = torch.max(gaussians_change.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians_change.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration in [4]:
                    grads = gaussians_change.xyz_gradient_accum / gaussians_change.denom
                    grads[grads.isnan()] = 0.0

                    gaussians_change.tmp_radii = radii
                    if iteration == 4:
                        scene_center = torch.stack(cam_centers, dim=0).mean(dim=0)
                        extent = torch.max(torch.linalg.norm(torch.stack([v for v in cam_centers], dim=0) - scene_center.unsqueeze(0), dim=-1)).item() * 1.1

                    gaussians_change.densify_and_clone(grads, opt.densify_grad_threshold*5, extent)
                    gaussians_change.densify_and_split(grads, opt.densify_grad_threshold*5, extent)

                    gaussians_change.tmp_radii = None
                    torch.cuda.empty_cache()
                          
     
        with torch.no_grad():
            render_pkg_change = render_change(view, gaussians_change, pipe, background)
            change_mask, viewspace_point_tensor, visibility_filter, radii = render_pkg_change["render"], render_pkg_change["viewspace_points"], render_pkg_change["visibility_filter"], render_pkg_change["radii"]
            change_mask = change_mask.mean(dim=0)
            change_mask = (change_mask > 0.5).float()
            change_masks[view.image_name] = change_mask

    pbar_inf.close()
    
    if args.refine:
        pbar_finetuning = tqdm(range(total_iterations, 3000), desc="Refining Change Mask")
    
        for iteration in pbar_finetuning:
            total_iterations += 1
            keyframe_idx = randint(0, len(viewpoints)-1)
            viewpoint = viewpoints[keyframe_idx]
            gaussians_change.update_learning_rate(total_iterations)
            render_pkg_change = render_change(viewpoint, gaussians_change, pipe, background)
            change_mask, viewspace_point_tensor, visibility_filter, radii = render_pkg_change["render"], render_pkg_change["viewspace_points"], render_pkg_change["visibility_filter"], render_pkg_change["radii"]
            gt_change = viewpoint.candidate_map

            change_mask = torch.sigmoid(change_mask.mean(dim=0, keepdim=True))
            d_loss = (gt_change*(1.0-change_mask)).mean()
            d_reg = torch.log(change_mask.mean()**2 + 1.000000001)
            loss = d_loss + d_reg
            loss.backward()

            with torch.no_grad():

                gaussians_change.optimizer.step()
                gaussians_change.optimizer.zero_grad(set_to_none = True)

                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if total_iterations % 5 == 0:
                    pbar_finetuning.set_postfix({
                        "it": total_iterations,
                        "loss": f"{ema_loss_for_log:.4f}",
                    })

            if total_iterations < opt.densify_until_iter:
                gaussians_change.max_radii2D[visibility_filter] = torch.max(gaussians_change.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians_change.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if total_iterations % opt.densification_interval == 0:
                    size_threshold = 20 if total_iterations > opt.opacity_reset_interval else None
                    gaussians_change.densify_and_prune(opt.densify_grad_threshold, 0.4, extent, size_threshold)

                if total_iterations % 2000 == 0 or (False and total_iterations == opt.densify_from_iter):
                    gaussians_change.reset_opacity()

        pbar_finetuning.close()

    for key in change_masks:
        cv2.imwrite(os.path.join(renders_path, "change_mask", f"{key}.png"), (change_masks[key].cpu().numpy() * 255).astype(np.uint8))

    if args.refine:
        with torch.no_grad():
            for view in viewpoints:
                render_pkg = render_change(view, gaussians_change, pipe, background)
                change_mask, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                change_mask = change_mask.mean(dim=0)
                change_mask = (change_mask > 0.5).float()
                cv2.imwrite(os.path.join(renders_path, "change_mask_refined", f"{view.image_name}.png"), (change_mask.cpu().numpy() * 255).astype(np.uint8))

    # Save inference cameras to JSON for viewer
    cameras_json = []
    for idx, view in enumerate(viewpoints):
        cameras_json.append(camera_to_JSON(idx, view))
    
    with open(os.path.join(args.model_path, "cameras.json"), 'w') as f:
        json.dump(cameras_json, f, indent=4)

if __name__ == "__main__":
    # Set up command line argument parser
    args, lp, op, pp = parse_args()
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    main(lp.extract(args), op.extract(args), pp.extract(args), args)
