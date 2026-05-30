import os
import torch
import numpy as np
from pathlib import Path
import tempfile
from torch.cuda.amp import autocast

"""
Segment Anything Model (SAM) utilities.
To use SAM features, install required packages:
    pip install segment-anything
    pip install opencv-python
    
Then download the model checkpoint:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""

# Global variables to track SAM availability and device
_SAM_AVAILABLE = False
_SAM_MODEL = None
_SAM_CURRENT_DEVICE = None

try:
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment-anything not installed. Will use fallback masks.")
    print("To enable SAM features, install: pip install segment-anything opencv-python")

def _init_sam_model(sam_device="cpu"):
    """Initialize SAM model with ViT-B variant if available
    
    Args:
        sam_device: Device to load SAM model on ('cpu' or 'cuda'). Defaults to 'cpu' to save VRAM.
    """
    global _SAM_MODEL, _SAM_AVAILABLE, _SAM_CURRENT_DEVICE
    
    if not _SAM_AVAILABLE:
        return
    
    # If model is already loaded on the correct device, no need to reinitialize
    if _SAM_MODEL is not None and _SAM_CURRENT_DEVICE == sam_device:
        return
    
    # If model is loaded on different device, need to move it
    if _SAM_MODEL is not None and _SAM_CURRENT_DEVICE != sam_device:
        print(f"Moving SAM model from {_SAM_CURRENT_DEVICE} to {sam_device}")
        _SAM_MODEL.model.to(sam_device)
        _SAM_CURRENT_DEVICE = sam_device
        return

    try:
        # Try to find SAM checkpoint in common locations
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        potential_paths = [
            sam_checkpoint,
            os.path.join("weights", sam_checkpoint),
            os.path.expanduser(f"~/.cache/segment_anything/{sam_checkpoint}")
        ]
        
        checkpoint_path = None
        for path in potential_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
                
        if checkpoint_path is None:
            print(f"Warning: SAM checkpoint not found in: {potential_paths}")
            _SAM_AVAILABLE = False
            return
            
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(sam_device)
        _SAM_MODEL = SamPredictor(sam)
        _SAM_CURRENT_DEVICE = sam_device
        print(f"SAM model initialized successfully on {sam_device}")
        
    except Exception as e:
        print(f"Warning: Failed to initialize SAM model: {e}")
        _SAM_AVAILABLE = False

def _get_cache_path(view_id):
    """Get cache file path for mask"""
    cache_dir = Path(tempfile.gettempdir()) / "sam_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"sam_mask_{view_id}.pt"

def set_sam_device(device):
    """Set SAM model device explicitly
    
    Args:
        device: 'cpu' or 'cuda'
    """
    if _SAM_AVAILABLE:
        _init_sam_model(device)

def extract_sam_mask(img, cache=True, view_id=None, sam_device="cpu"):
    """Extract segmentation mask using Segment Anything Model (SAM)
    
    Args:
        img: Input image tensor or numpy array
             - torch.Tensor: [3, H, W] or [H, W, 3] float32 [0-1]
             - numpy.ndarray: [H, W, 3] uint8 [0-255] or float [0-1]
        cache: Whether to use cache for mask storage/loading
        view_id: Optional view identifier for cache filename
        sam_device: Device to run SAM model on ('cpu' or 'cuda'). Defaults to 'cpu' to save VRAM.
        
    Returns:
        mask: Binary segmentation mask tensor of shape [1, 1, H, W]
    """
    # Convert input to numpy uint8 [H, W, 3] format for SAM
    if torch.is_tensor(img):
        img = img.detach().cpu()
        
        # Handle different tensor shapes
        if img.ndim == 3:
            if img.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
                img = img.permute(1, 2, 0)
            # else: already [H, W, 3]
        elif img.ndim == 4:
            raise ValueError("Batch input not supported, process images one by one")
        
        # CRITICAL: Apply .contiguous() AFTER permute, BEFORE dtype conversion
        # This ensures memory layout is correct before changing dtype
        img = img.contiguous()
        
        # Clamp to [0, 1] and convert to uint8 [0-255]
        img = torch.clamp(img, 0.0, 1.0)
        img = (img * 255.0).to(torch.uint8).numpy()
        
        # CRITICAL: Force numpy copy to ensure C-contiguous and avoid OpenCV assertion failures
        # Some GPU environments (e.g., A100) require explicit memory copy for cv2 operations
        img = img.copy()
        
    elif isinstance(img, np.ndarray):
        # Ensure correct shape [H, W, 3]
        if img.ndim == 3 and img.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            img = np.transpose(img, (1, 2, 0))
        
        # Ensure uint8 [0-255]
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # CRITICAL: Ensure C-contiguous memory layout and force copy for SAM/OpenCV compatibility
        if not img.flags['C_CONTIGUOUS'] or not img.flags['OWNDATA']:
            img = np.ascontiguousarray(img).copy()
        else:
            # Even if contiguous, make a copy to avoid OpenCV issues
            img = img.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")
    
    # Validate final shape
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] shape, got {img.shape}")
    
    H, W = img.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try loading from cache first
    if cache and view_id is not None:
        cache_path = _get_cache_path(view_id)
        if cache_path.exists():
            try:
                mask = torch.load(cache_path, map_location=device)
                if mask.shape[-2:] == (H, W):
                    return mask.float()
            except:
                pass

    # Initialize SAM if not already done or if device changed
    if _SAM_AVAILABLE:
        _init_sam_model(sam_device)
    
    # Generate mask using SAM
    if _SAM_AVAILABLE and _SAM_MODEL is not None:
        try:
            # CRITICAL: Do NOT use autocast() here!
            # SAM expects numpy uint8 input and handles dtype conversion internally.
            # autocast() will interfere with SAM's internal operations and cause dtype errors.
            _SAM_MODEL.set_image(img)
            # Get mask for center point prompt
            center_point = np.array([[W//2, H//2]])
            masks, _, _ = _SAM_MODEL.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                multimask_output=False
            )
            mask = torch.from_numpy(masks[0]).to(device)[None, None].float()
        except Exception as e:
            print(f"Warning: SAM inference failed: {e}")
            mask = torch.ones((1, 1, H, W), device=device)
    else:
        # Fallback: return all-ones mask
        mask = torch.ones((1, 1, H, W), device=device)
    
    # Cache the result
    if cache and view_id is not None:
        try:
            torch.save(mask.cpu(), _get_cache_path(view_id))
        except:
            pass
            
    return mask.float()