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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_l_val
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.flow_utils import compute_overlap_mask
from torch.cuda.amp import autocast, GradScaler
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def poses_to_vec(poses):
    """Convert camera poses to vector format [V, 12]
    
    Returns raw 12-element vectors (no normalization) containing:
    - Elements 0-8: Rotation matrix R (3x3) flattened in row-major order
    - Elements 9-11: Translation vector t (3x1)
    
    VACM module expects these raw pose values for internal processing.
    Camera intrinsics are handled separately if needed by VACM.
    """
    pose_vecs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for pose in poses:
        if hasattr(pose, 'world_view_transform') and pose.world_view_transform is not None:
            # Extract 3x4 transform from 4x4 matrix
            world_view = pose.world_view_transform
            if world_view.shape[0] == 4:  # 4x4 matrix
                transform = world_view[:3, :]  # [3, 4]
            else:  # Already 3x4
                transform = world_view
            
            # Ensure correct device and dtype, flatten in row-major order
            # IMPORTANT: VACM expects raw pose vector, not normalized
            pose_vec = transform.to(device=device, dtype=torch.float32).flatten()  # [12]
        else:
            # Fallback: create identity 3x4 transform [R|t] where R=I, t=0
            # IMPORTANT: VACM expects raw pose vector, not normalized
            identity_transform = torch.zeros(3, 4, device=device, dtype=torch.float32)
            identity_transform[:3, :3] = torch.eye(3, device=device, dtype=torch.float32)
            pose_vec = identity_transform.flatten()  # [12]
        
        pose_vecs.append(pose_vec)
    
    return torch.stack(pose_vecs, dim=0)  # [V, 12]

def should_apply_vacm(iteration, overlap_ratio, flags, total_iters):
    """Determine if VACM should be applied at current iteration
    
    Args:
        iteration: Current training iteration
        overlap_ratio: Ratio of overlapping points (0-1), or None if unavailable
        flags: Object with densify_just_happened, sh_just_upgraded attributes
        total_iters: Total number of training iterations
        
    Returns:
        bool: True if VACM should be applied
    """
    # Warm-up & disable window
    warmup = int(0.05 * total_iters)
    disable_after = int(0.7 * total_iters)
    if iteration < warmup or iteration >= disable_after:
        return False

    # 错峰保护：densify / SH 升级时跳过
    if getattr(flags, "densify_just_happened", False) or getattr(flags, "sh_just_upgraded", False):
        return False

    # 重叠密度阈值
    if overlap_ratio is None or overlap_ratio <= 0.3:
        return False

    # 自适应频率
    if iteration < 10_000:
        freq = 300
    elif iteration < 20_000:
        freq = 200
    else:
        freq = 100

    return (iteration % freq) == 0

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup_reconstruction(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Reconstruction training", bar_format='{l_bar}{r_bar}')
    first_iter += 1
    
    # Initialize GradScaler for mixed precision training (enabled by default, disable with --no_amp)
    use_amp = not opt.no_amp
    scaler = GradScaler(enabled=use_amp)
    
    # VACM scheduling state
    class VACMFlags:
        def __init__(self):
            self.densify_just_happened = False
            self.sh_just_upgraded = False
            self.densify_delay_counter = 0
    
    vacm_flags = VACMFlags()
    vacm_apply_count = 0
    vacm_skip_count = 0
    overlap_smooth = 0.0  # Smoothed overlap ratio for monitoring
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        
        # Clear gradients from previous iteration at the beginning
        gaussians.optimizer.zero_grad(set_to_none=True)

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            old_sh_degree = gaussians.active_sh_degree
            gaussians.oneupSHdegree()
            if gaussians.active_sh_degree > old_sh_degree:
                vacm_flags.sh_just_upgraded = True

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Forward pass with unified autocast for rendering and loss computation
        with autocast(enabled=use_amp):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss computation
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # Add VACM regularization if enabled
            if hasattr(gaussians, "vacm_scale") and gaussians.vacm_scale.requires_grad:
                loss += 1e-6 * (gaussians.vacm_scale ** 2).sum()
        
        # Apply VACM if enabled
        view_loss = None
        if args.use_vacm:
            # Get all camera poses and ensure device/dtype compatibility
            all_cameras = scene.getTrainCameras()
            poses_vec = poses_to_vec(all_cameras)  # [V, 12]
            
            # Ensure poses_vec is on the same device as gaussians and has correct dtype
            model_device = gaussians.get_xyz.device
            poses_vec = poses_vec.to(device=model_device, dtype=torch.float32)
            
            # Compute overlap mask and ensure device compatibility
            overlap_mask = compute_overlap_mask(all_cameras, gaussians.get_xyz)
            
            # Debug output every 500 iterations
            if iteration % 500 == 0:
                overlap_exists = overlap_mask is not None
                overlap_ratio_debug = overlap_mask.float().mean().item() if overlap_exists and overlap_mask.any() else 0.0
                print(f"[DEBUG] Iter {iteration}: overlap_mask_exists={overlap_exists}, overlap_ratio={overlap_ratio_debug:.3f}")
            
            # Handle None case when overlap computation fails
            if overlap_mask is None or not overlap_mask.any():
                overlap_ratio = 0.0
                if iteration % 500 == 0 and overlap_mask is None:
                    print(f"[VACM] Iteration {iteration}: overlap mask unavailable, skipping VACM")
                vacm_skip_count += 1
            else:
                overlap_mask = overlap_mask.to(device=model_device, dtype=torch.bool)
                overlap_ratio = overlap_mask.float().mean().item()
                
                # Check if VACM should be applied using scheduling function
                if should_apply_vacm(iteration, overlap_ratio, vacm_flags, opt.iterations):
                    # Use unified autocast for VACM operations
                    with autocast(enabled=use_amp):
                        # Apply VACM with device-compatible tensors
                        gaussians.apply_vacm(iteration, poses_vec, overlap_mask)
                        vacm_apply_count += 1
                        
                        # Compute view consistency loss with adjacent views (only when applying VACM)
                        if len(all_cameras) > 1:
                            # Find current camera index using id comparison for safety
                            current_idx = None
                            for idx, cam in enumerate(all_cameras):
                                if id(cam) == id(viewpoint_cam):
                                    current_idx = idx
                                    break
                            
                            # If not found, use a random adjacent pair instead
                            if current_idx is None:
                                current_idx = 0
                                
                            next_idx = (current_idx + 1) % len(all_cameras)
                            next_cam = all_cameras[next_idx]
                            
                            # Render next view
                            next_render_pkg = render(next_cam, gaussians, pipe, bg)
                            next_image = next_render_pkg["render"]
                            
                            # Extract features for view consistency with proper batch dimension
                            # Reshape to [1, C, H*W] format for compute_l_val
                            view_feats_i = image.unsqueeze(0).flatten(2)  # [3, H, W] -> [1, 3, H*W]
                            view_feats_j = next_image.unsqueeze(0).flatten(2)  # [3, H, W] -> [1, 3, H*W]
                            
                            # Compute view consistency loss
                            view_loss = compute_l_val(view_feats_i, view_feats_j)
                            loss += args.lambda_view * view_loss
                else:
                    vacm_skip_count += 1
        
        # Backward pass with proper AMP handling
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Print VRAM usage every 100 iterations
            if iteration % 100 == 0:
                print(f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                
            # VACM monitoring and decay
            if args.use_vacm:
                # Update smoothed overlap ratio
                if iteration % 100 == 0 and 'overlap_ratio' in locals():
                    overlap_smooth = 0.9 * overlap_smooth + 0.1 * overlap_ratio
                
                # Print VACM status every 1000 iterations
                if iteration % 1000 == 0:
                    print(f"[VACM] overlap_smooth={overlap_smooth:.3f}, applied={vacm_apply_count}, skipped={vacm_skip_count}")
                    
                # Decay view consistency loss weight
                if iteration % 5000 == 0:
                    args.lambda_view *= 0.999
                
            # Check point cloud size and prune if needed
            if iteration % 100 == 0 and gaussians.get_xyz.shape[0] > 500000:
                prune_mask = compute_overlap_mask(scene.getTrainCameras(), gaussians.get_xyz)
                prune_mask = prune_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool)
                gaussians.prune_points(prune_mask)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), view_loss)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    # Set densify flag and delay counter
                    vacm_flags.densify_just_happened = True
                    vacm_flags.densify_delay_counter = 10
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            # Update densify delay counter
            if vacm_flags.densify_delay_counter > 0:
                vacm_flags.densify_delay_counter -= 1
                if vacm_flags.densify_delay_counter == 0:
                    vacm_flags.densify_just_happened = False
                    
            # Reset SH upgrade flag (only lasts one iteration)
            if vacm_flags.sh_just_upgraded:
                vacm_flags.sh_just_upgraded = False

            # Optimizer step with proper GradScaler handling
            if iteration < opt.iterations:
                if use_amp:
                    # AMP mode: use GradScaler
                    scaler.unscale_(gaussians.optimizer)
                    
                    # Check if any parameters have valid gradients
                    has_grads = any(p.grad is not None and p.grad.numel() > 0 
                                   for group in gaussians.optimizer.param_groups 
                                   for p in group['params'])
                    
                    if has_grads:
                        # Optional gradient clipping
                        if opt.grad_clip_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_([p for group in gaussians.optimizer.param_groups for p in group['params']], opt.grad_clip_norm)
                        
                        # Step optimizer with scaler
                        scaler.step(gaussians.optimizer)
                    
                    # Always update scaler after unscale
                    scaler.update()
                else:
                    # Non-AMP mode: direct optimizer step
                    # Optional gradient clipping
                    if opt.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_([p for group in gaussians.optimizer.param_groups for p in group['params']], opt.grad_clip_norm)
                    
                    # Direct optimizer step
                    gaussians.optimizer.step()

            # Clear gradients at the end of iteration to free memory
            gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, view_loss=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if view_loss is not None:
            tb_writer.add_scalar('train_loss_patches/view_loss', view_loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--exp_name", type=str, default='default')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]

    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "reconstruction", args.exp_name)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nReconstruction complete.")
