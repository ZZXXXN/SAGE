import torch
import numpy as np
import warnings
from typing import List, Tuple

"""
Flow and correspondence utilities.
For feature matching, OpenCV is required:
    pip install opencv-python
"""

# Global variable to track OpenCV availability
_CV2_AVAILABLE = False

# Global variable to track world_to_ndc availability
_WORLD_TO_NDC_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    warnings.warn(
        "OpenCV not installed, will use random matches as fallback. "
        "To enable feature matching, install: pip install opencv-python"
    )

# Try to import world_to_ndc function
try:
    from utils.graphics_utils import world_to_ndc, fov2focal
    _WORLD_TO_NDC_AVAILABLE = True
    print("[INFO] Successfully imported world_to_ndc from utils.graphics_utils")
except ImportError as e:
    warnings.warn(f"[WARNING] Could not import world_to_ndc: {e}. compute_overlap_mask will return None.")
    _WORLD_TO_NDC_AVAILABLE = False

def _camera_to_intrinsics(camera):
    """Extract intrinsic matrix from camera object
    
    Args:
        camera: Camera object with FoVx, FoVy, image_width, image_height
        
    Returns:
        intrinsics: torch.Tensor [3, 3] - camera intrinsic matrix
    """
    width = camera.image_width
    height = camera.image_height
    
    # Compute focal lengths from FoV
    fx = fov2focal(camera.FoVx, width)
    fy = fov2focal(camera.FoVy, height)
    
    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0
    
    # Construct intrinsic matrix
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    return K

def _to_numpy_uint8(img):
    """Convert tensor/array to uint8 numpy array"""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    return img

def light_flow_matches(img1, img2, max_matches=100) -> List[Tuple[int, int]]:
    """Find feature matches between two images using ORB+BFMatcher
    
    Args:
        img1: First image, tensor or array of shape [H, W, 3]
        img2: Second image, tensor or array of shape [H, W, 3]
        max_matches: Maximum number of matches to return
        
    Returns:
        List of (p,q) index tuples, where p is pixel index in img1
        and q is the corresponding pixel index in img2.
        Indices are flattened (i.e., for use with [..., H*W] tensors)
    """
    # Convert inputs to uint8 numpy arrays
    img1 = _to_numpy_uint8(img1)
    img2 = _to_numpy_uint8(img2)
    H, W = img1.shape[:2]
    
    if _CV2_AVAILABLE:
        try:
            # Initialize ORB detector
            orb = cv2.ORB_create()
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
            kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
            
            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                raise ValueError("Not enough features detected")
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:max_matches]
            
            # Convert keypoint coordinates to flattened indices
            match_pairs = []
            for m in matches:
                p1 = kp1[m.queryIdx].pt
                p2 = kp2[m.trainIdx].pt
                
                # Round to nearest pixel and convert to flattened index
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                x2, y2 = int(round(p2[0])), int(round(p2[1]))
                
                # Ensure coordinates are within image bounds
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    idx1 = y1 * W + x1
                    idx2 = y2 * W + x2
                    match_pairs.append((idx1, idx2))
            
            return match_pairs
            
        except Exception as e:
            warnings.warn(f"Feature matching failed: {e}. Using random matches as fallback.")
    
    # Fallback: generate random matches
    num_pixels = H * W
    num_matches = min(max_matches, num_pixels // 100)  # Use 1% of pixels or max_matches
    
    p_indices = np.random.choice(num_pixels, num_matches, replace=False)
    q_indices = np.random.choice(num_pixels, num_matches, replace=False)
    
    return list(zip(p_indices.tolist(), q_indices.tolist()))

def compute_overlap_mask(poses, xyz, threshold=0.1) -> torch.Tensor:
    """Compute mask for points visible in multiple views
    
    Args:
        poses: List of Camera objects with world_view_transform, FoVx, FoVy, image_width, image_height
        xyz: Point cloud coordinates of shape [P, 3]
        threshold: Minimum overlap ratio (default 0.3)
        
    Returns:
        Boolean mask of shape [P] indicating points visible in multiple views,
        or None if computation failed or insufficient overlap
    """
    # Global counter for throttled warnings (one-time per 500 iterations)
    if not hasattr(compute_overlap_mask, "_warn_count"):
        compute_overlap_mask._warn_count = 0
    
    # Check if world_to_ndc is available
    if not _WORLD_TO_NDC_AVAILABLE:
        if compute_overlap_mask._warn_count % 500 == 0:
            warnings.warn("compute_overlap_mask: world_to_ndc not available, returning None")
        compute_overlap_mask._warn_count += 1
        return None
    
    try:
        # Validate input parameters
        if xyz is None or len(xyz) == 0:
            if compute_overlap_mask._warn_count % 500 == 0:
                warnings.warn("compute_overlap_mask: xyz is None or empty, returning None")
            compute_overlap_mask._warn_count += 1
            return None
            
        if poses is None or len(poses) < 2:
            if compute_overlap_mask._warn_count % 500 == 0:
                warnings.warn("compute_overlap_mask: need at least 2 camera poses, returning None")
            compute_overlap_mask._warn_count += 1
            return None
        
        if not torch.is_tensor(xyz):
            xyz = torch.tensor(xyz, dtype=torch.float32)
            
        # Track points visible in each view
        num_points = len(xyz)
        num_cameras = len(poses)
        visibility_map = torch.zeros((num_cameras, num_points), dtype=torch.bool, device=xyz.device)
        vis_ratios = []
        
        for i, camera in enumerate(poses):
            # Extract camera intrinsics
            K = _camera_to_intrinsics(camera).to(xyz.device)
            
            # Project to NDC and get visibility mask
            ndc_xy, visible = world_to_ndc(
                xyz, 
                camera.world_view_transform,
                K,
                camera.image_width,
                camera.image_height
            )
            
            # Check for invalid projections (NaN, Inf)
            if torch.isnan(ndc_xy).any() or torch.isinf(ndc_xy).any():
                if compute_overlap_mask._warn_count % 500 == 0:
                    warnings.warn("compute_overlap_mask: NDC projection contains NaN/Inf, returning None")
                compute_overlap_mask._warn_count += 1
                return None
            
            visibility_map[i] = visible
            vis_ratio = visible.float().mean().item()
            vis_ratios.append(vis_ratio)
            
            # Check if any camera has too few visible points
            if vis_ratio < 0.01:  # Less than 1% visible
                if compute_overlap_mask._warn_count % 500 == 0:
                    warnings.warn(f"compute_overlap_mask: camera {i} has only {vis_ratio*100:.1f}% visible points, returning None")
                compute_overlap_mask._warn_count += 1
                return None
        
        # Compute overlap: points visible in at least 2 cameras
        view_counts = visibility_map.sum(dim=0)  # [P]
        overlap_mask = view_counts >= 2
        
        # Compute overlap ratio
        overlap_ratio = overlap_mask.float().mean().item()
        
        # Check overlap threshold (use max of parameter and minimum 0.3)
        min_overlap = max(threshold, 0.3)
        if overlap_ratio < min_overlap:
            if compute_overlap_mask._warn_count % 500 == 0:
                warnings.warn(f"compute_overlap_mask: overlap ratio {overlap_ratio:.3f} < {min_overlap:.3f}, returning None")
            compute_overlap_mask._warn_count += 1
            return None
        
        return overlap_mask.bool()
        
    except Exception as e:
        if compute_overlap_mask._warn_count % 500 == 0:
            warnings.warn(f"compute_overlap_mask: unexpected error ({e}), returning None")
        compute_overlap_mask._warn_count += 1
        return None