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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.functional import resize
import torchvision.transforms as T
import numpy as np
from scene.VGG import VGGEncoder, normalize_vgg
from utils.loss_utils import cal_adain_style_loss, cal_mse_content_loss, compute_l_scm
from utils.seg_utils import extract_sam_mask
from utils.flow_utils import light_flow_matches, compute_overlap_mask
from torch.cuda.amp import autocast, GradScaler
import pickle
import hashlib
import warnings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def getDataLoader(dataset_path, batch_size, sampler, image_side_length=256, num_workers=2):
    transform = T.Compose([
                T.Resize(size=(image_side_length*2, image_side_length*2)),
                T.RandomCrop(image_side_length),
                T.ToTensor(),
            ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler(len(train_dataset)), num_workers=num_workers)

    return dataloader

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def _get_disk_cache_path(args, cache_key):
    """Generate disk cache path for semantic masks"""
    cache_dir = os.path.join(args.model_path, "mask_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Use hash to avoid filesystem issues with long keys
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return os.path.join(cache_dir, f"mask_{key_hash}.pkl")

def _save_mask_to_disk(args, cache_key, mask_data):
    """Save mask to disk cache (numpy RGB uint8 format for ORB compatibility)"""
    try:
        cache_path = _get_disk_cache_path(args, cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(mask_data, f)
    except Exception as e:
        warnings.warn(f"Failed to save mask to disk cache: {e}")

def _load_mask_from_disk(args, cache_key):
    """Load mask from disk cache (returns numpy RGB uint8 format for ORB compatibility)"""
    try:
        cache_path = _get_disk_cache_path(args, cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load mask from disk cache: {e}")
    return None

def _validate_index_consistency(matches, H, W, max_check=5):
    """Validate that flattened indices from matches are consistent with tensor dimensions"""
    if not matches or len(matches) == 0:
        return True
    
    # Check a few matches to ensure indices are valid
    check_count = min(len(matches), max_check)
    for i in range(check_count):
        p_idx, q_idx = matches[i]
        # Validate indices are within bounds
        if not (0 <= p_idx < H * W) or not (0 <= q_idx < H * W):
            warnings.warn(f"Invalid match index: ({p_idx}, {q_idx}) for image size {H}x{W}")
            return False
    return True

def training(dataset, opt, pipe, ckpt_path, decoder_path, style_weight, content_preserve, args):
    opt.iterations = 100_000 if not decoder_path else 30_000
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    gaussians = GaussianModel(dataset.sh_degree)
    # load the feature reconstructed gaussians ckpt file
    scene = Scene(dataset, gaussians, load_path=ckpt_path)
    vgg_encoder = VGGEncoder().cuda()

    # compute the final vgg features for each point, and init pointnet decoder
    gaussians.training_setup_style(opt, decoder_path)

    # init wikiart dataset
    style_loader = getDataLoader(args.wikiartdir, batch_size=1, sampler=InfiniteSamplerWrapper, 
                    image_side_length=256, num_workers=4)
    style_iter = iter(style_loader)

    bg_color = [1]*3 if dataset.white_background else [0]*3
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Artistic training", bar_format='{l_bar}{r_bar}')
    first_iter += 1
    
    # Initialize GradScaler for mixed precision training (enabled by default, disable with --no_amp)
    use_amp = not opt.no_amp
    scaler = GradScaler(enabled=use_amp)
    
    # Cache for semantic masks to avoid recomputation
    original_masks_cache = {}  # Cache RGB masks (numpy uint8 for OpenCV)
    semantic_mask_cache = {}  # Cache semantic masks across iterations (torch tensors stored on CPU)
    max_cache_size = 50  # Limit cache size to prevent memory issues in long training
    
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        
        # Clear gradients from previous iteration at the beginning
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # content preserve training
        if content_preserve and iteration % 7 == 0:
            decoded_rgb = gaussians.decoder(gaussians.final_vgg_features.detach()) # [N, 3]
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=decoded_rgb)
            rendered_rgb = render_pkg["render"] # [3, H, W]
            gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]
            loss = l1_loss(gt_image, rendered_rgb)
            
            # Backward pass with AMP support
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            iter_end.record()
            if iteration % 10 == 0:
                progress_bar.update(10)
            tb_writer.add_scalar('train_loss/content_preserve', loss.item(), iteration)
            # Optimizer step with unified AMP logic
            if iteration < opt.iterations:
                if use_amp:
                    scaler.unscale_(gaussians.optimizer)
                    if opt.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_([p for group in gaussians.optimizer.param_groups for p in group['params']], opt.grad_clip_norm)
                    scaler.step(gaussians.optimizer)
                    scaler.update()
                else:
                    if opt.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_([p for group in gaussians.optimizer.param_groups for p in group['params']], opt.grad_clip_norm)
                    gaussians.optimizer.step()
            continue


        # get style_img, this style_img has NOT been normalized according to the pretrained VGGmodel
        style_img = next(style_iter)[0].cuda()
        gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]

        # Get or compute semantic mask (cached across iterations and per-view)
        semantic_mask = None
        if args.use_sgm:
            # Use (camera_id, block_id) as cache key to better reuse masks across iterations
            # Block id groups iterations in chunks (e.g., 100 iters per block)
            block_id = iteration // 100
            # Use a stable camera identifier if available
            try:
                cam_uid = getattr(viewpoint_cam, 'uid', None) or getattr(viewpoint_cam, 'image_name', None) or f"cam_{id(viewpoint_cam)}"
            except Exception:
                cam_uid = f"cam_{id(viewpoint_cam)}"
            cache_key = f"{cam_uid}_block_{block_id}"

            if cache_key not in semantic_mask_cache:
                try:
                    # First render without SGM to get initial image for mask extraction
                    with torch.no_grad():
                        style_img_features = vgg_encoder(normalize_vgg(style_img)) # [1, C, H, W]
                        gt_image_features = vgg_encoder(normalize_vgg(gt_image.unsqueeze(0)))
                        
                        # Initial render without SGM
                        initial_features = gaussians.style_transfer.forward(
                            gaussians.final_vgg_features.detach(),
                            style_img_features.relu3_1,
                            trans=True
                        )
                        initial_rgb = gaussians.decoder(initial_features)
                        initial_render = render(viewpoint_cam, gaussians, pipe, background, override_color=initial_rgb)
                        initial_rendered_rgb = initial_render["render"]
                        
                    # Run SAM on GPU if available (faster), results cached on CPU
                    # NOTE: extract_sam_mask internally handles numpy uint8 conversion
                    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
                    mask_gpu = extract_sam_mask(
                        initial_rendered_rgb.permute(1, 2, 0),
                        cache=True,
                        view_id=f"view_{cache_key}",
                        sam_device=sam_device,
                    )
                    # Store mask on CPU to avoid VRAM accumulation
                    semantic_mask_cache[cache_key] = mask_gpu.cpu()

                    # Limit cache size to prevent memory issues
                    if len(semantic_mask_cache) > max_cache_size:
                        # Remove oldest cache entries (FIFO)
                        oldest_key = min(semantic_mask_cache.keys())
                        del semantic_mask_cache[oldest_key]
                except Exception as e:
                    warnings.warn(f"[SGM] Failed to build semantic mask for key '{cache_key}': {e}. Skip SGM for this iteration.")

            # Transfer cached mask back to GPU when needed（再次检查 key 是否存在，避免 KeyError）
            if cache_key in semantic_mask_cache:
                semantic_mask = semantic_mask_cache[cache_key].to(device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                semantic_mask = None
        
        # Render with proper features computation
        with torch.no_grad():
            style_img_features = vgg_encoder(normalize_vgg(style_img)) # [1, C, H, W]
            gt_image_features = vgg_encoder(normalize_vgg(gt_image.unsqueeze(0)))

        # decoder the features of points to rgb
        if args.use_sgm and semantic_mask is not None:
            tranfered_features = gaussians.style_transfer.forward(
                gaussians.final_vgg_features.detach(), # point cloud features [N, C]
                style_img_features.relu3_1,
                trans=True,
                use_sgm=True,
                semantic_mask=semantic_mask,
                iter=iteration
            )
        else:
            tranfered_features = gaussians.style_transfer.forward(
                gaussians.final_vgg_features.detach(), # point cloud features [N, C]
                style_img_features.relu3_1,
                trans=True
            )

        decoded_rgb = gaussians.decoder(tranfered_features) # [N, 3]

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=decoded_rgb)
        rendered_rgb = render_pkg["render"] # [3, H, W]

        # style loss and content loss
        rendered_rgb_features = vgg_encoder(normalize_vgg(rendered_rgb.unsqueeze(0))) 

        content_loss = cal_mse_content_loss(gt_image_features.relu4_1, rendered_rgb_features.relu4_1)
        style_loss = 0.
        for style_feature, image_feature in zip(style_img_features, rendered_rgb_features):
            style_loss += cal_adain_style_loss(style_feature, image_feature)
            
        # Compute semantic correspondence loss if enabled
        sem_loss = 0.
        if args.use_sgm and semantic_mask is not None:
            # Convert stylized mask format - handle both single [1,1,H,W] and batch [B,1,H,W]
            # INPUT: torch tensor float32 [0,1] range
            # OUTPUT: numpy array uint8 [0,255] RGB format for OpenCV ORB feature matching
            def convert_mask_to_rgb(mask_tensor):
                """Convert torch tensor mask to numpy RGB format for OpenCV operations
                Args:
                    mask_tensor: torch.Tensor, float32 [0,1] range, shape [B,1,H,W] or [1,1,H,W]
                Returns:
                    numpy.ndarray: uint8 [0,255] RGB format, shape [H,W,3] for OpenCV compatibility
                """
                if mask_tensor.dim() == 4:  # [B,1,H,W] or [1,1,H,W]
                    if mask_tensor.shape[0] > 1:  # Batch case
                        masks = []
                        for b in range(mask_tensor.shape[0]):
                            single_mask = mask_tensor[b].squeeze().cpu().numpy()  # [H,W] float32 [0,1]
                            single_mask = (single_mask * 255).astype(np.uint8)  # Convert to uint8 [0-255]
                            rgb_mask = np.stack([single_mask, single_mask, single_mask], axis=-1)  # [H,W,3] uint8
                            masks.append(rgb_mask)
                        return masks
                    else:  # Single image case
                        single_mask = mask_tensor.squeeze().cpu().numpy()  # [H,W] float32 [0,1]
                        single_mask = (single_mask * 255).astype(np.uint8)  # Convert to uint8 [0-255] for ORB
                        return np.stack([single_mask, single_mask, single_mask], axis=-1)  # [H,W,3] uint8 RGB for OpenCV
                else:
                    raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")
            
            # Convert semantic mask to RGB format for ORB feature matching
            # INPUT: torch tensor float32 [0,1], OUTPUT: numpy uint8 RGB [0,255] for OpenCV
            stylized_mask = convert_mask_to_rgb(semantic_mask)
            
            # Get or compute original mask (cached to avoid recomputation)
            # Use a more robust cache key generation
            try:
                view_key = f"view_{viewpoint_cam.uid}"
            except AttributeError:
                # Fallback if uid is not available
                view_key = f"view_{hash(viewpoint_cam.image_name)}_{iteration//100}"
            
            # Try memory cache first, then disk cache, then compute
            if view_key not in original_masks_cache:
                # Try loading from disk cache (numpy RGB uint8 format for ORB compatibility)
                original_mask = _load_mask_from_disk(args, f"original_{view_key}")
                if original_mask is None:
                    # Compute and save to disk cache, use same SAM device policy as for stylized mask (prefer CUDA)
                    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
                    original_mask_tensor = extract_sam_mask(
                        gt_image.permute(1, 2, 0),
                        cache=True,
                        view_id=view_key,
                        sam_device=sam_device,
                    )
                    original_mask = convert_mask_to_rgb(original_mask_tensor)  # Convert to numpy RGB uint8
                    _save_mask_to_disk(args, f"original_{view_key}", original_mask)
                
                # Store in memory cache for quick access
                original_masks_cache[view_key] = original_mask
            else:
                original_mask = original_masks_cache[view_key]
            
            # Feature matching with ORB on numpy uint8 RGB images
            # INPUT: numpy uint8 [H,W,3] RGB [0,255] for OpenCV ORB
            # OUTPUT: list of (p_idx, q_idx) where indices are flattened (y*W + x) coordinates
            matches = light_flow_matches(stylized_mask, original_mask)
            
            # Validate index consistency (optional debug check)
            H, W = stylized_mask.shape[:2]
            if not _validate_index_consistency(matches, H, W):
                matches = []  # Use empty matches if validation fails
            
            # Convert numpy arrays back to tensors for PyTorch loss computation
            # INPUT: numpy uint8 RGB [0,255], OUTPUT: torch tensor float32 [0,1] for F.mse_loss
            device = "cuda" if torch.cuda.is_available() else "cpu"
            stylized_mask_tensor = torch.from_numpy(stylized_mask).float().to(device) / 255.0  # Normalize back to [0,1]
            original_mask_tensor = torch.from_numpy(original_mask).float().to(device) / 255.0  # Normalize back to [0,1]
            
            # Reshape tensors to match compute_l_scm expected format [C, H*W]
            # Convert from [H, W, C] to [C, H*W] for indexing with flattened coordinates
            stylized_mask_flat = stylized_mask_tensor.permute(2, 0, 1).reshape(3, -1)  # [3, H*W] float32 for torch ops
            original_mask_flat = original_mask_tensor.permute(2, 0, 1).reshape(3, -1)  # [3, H*W] float32 for torch ops
            
            # Compute semantic consistency loss using matched keypoint indices
            # matches: list of (p_idx, q_idx) where idx = y*W + x (row-major flattened coordinates)
            # stylized_mask_flat/original_mask_flat: [3, H*W] tensors for indexing with [..., idx]
            # This ensures coordinate system consistency between ORB (x,y) -> (y*W + x) and PyTorch flatten
            sem_loss = compute_l_scm(stylized_mask_flat, original_mask_flat, matches)

        loss = content_loss + style_loss * style_weight
        if args.use_sgm:
            loss += sem_loss * args.lambda_sem
        
        # Backward pass with AMP support
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

            # Log
            tb_writer.add_scalar('train_loss/content_loss', content_loss.item(), iteration)
            tb_writer.add_scalar('train_loss/style_loss', style_loss.item(), iteration)
            if args.use_sgm:
                # sem_loss 可能是 tensor 也可能是 float，这里统一转为 Python float 以避免 .item() 报错
                sem_loss_value = sem_loss.item() if hasattr(sem_loss, 'item') else float(sem_loss)
                tb_writer.add_scalar('train_loss/semantic_loss', sem_loss_value, iteration)
            
            # Print VRAM usage every 100 iterations
            if iteration % 100 == 0:
                print(f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
            
            # Decay semantic loss weight
            if args.use_sgm and iteration % 1000 == 0:
                args.lambda_sem *= 0.999
                
            # Check point cloud size and prune if needed
            if iteration % 100 == 0 and gaussians.get_xyz.shape[0] > 500000:
                mask = compute_overlap_mask(scene.getTrainCameras(), gaussians.get_xyz)
                gaussians.prune_points(mask)
                
            if iteration % 500 == 0:
                style_img = resize(style_img, (128, 128))
                rendered_rgb.clamp_(0, 1)
                rendered_rgb[:, -128:, -128:] = style_img.squeeze(0)
                tb_writer.add_image('stylized_img', rendered_rgb.clamp(0,1), iteration, dataformats='CHW')

            # Optimizer step with unified AMP logic
            if iteration < opt.iterations:
                if use_amp:
                    scaler.unscale_(gaussians.optimizer)
                    if opt.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_([p for group in gaussians.optimizer.param_groups for p in group['params']], opt.grad_clip_norm)
                    scaler.step(gaussians.optimizer)
                    scaler.update()
                else:
                    if opt.grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_([p for group in gaussians.optimizer.param_groups for p in group['params']], opt.grad_clip_norm)
                    gaussians.optimizer.step()
    # Save model
    os.makedirs(args.model_path + "/chkpnt", exist_ok = True)
    torch.save(gaussians.capture(is_style_model=True), args.model_path + "/chkpnt" + "/gaussians.pth")

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
    tb_writer = SummaryWriter(args.model_path)

    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--rendering_mode", type=str, default="rgb", choices=["rgb", "feature"])
    parser.add_argument("--wikiartdir", type=str, default="datasets/wikiart")
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--style_weight", type=float, default=10.)
    parser.add_argument("--content_preserve", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]
    
    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "artistic", args.exp_name)
    print("Optimizing " + args.model_path + (' with content_preserve' if args.content_preserve else ''))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.ckpt_path, args.decoder_path, args.style_weight, args.content_preserve, args)

    # All done
    print("\nArtistic training complete.")