"""
Splatt3R Model Integration for SLAM
This module provides an interface to load and use Splatt3R models.
"""

import torch
import os
import sys
from huggingface_hub import hf_hub_download

# Add paths for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_current_dir, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(_current_dir, "src", "mast3r_src", "dust3r"))
sys.path.insert(0, os.path.join(_current_dir, "src", "pixelsplat_src"))
sys.path.insert(0, _current_dir)

from splatt3r_core.main import MAST3RGaussians


def load_splatt3r_model(checkpoint_path=None, device="cuda"):
    """
    Load Splatt3R model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file. If None, downloads from HuggingFace
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded MAST3RGaussians model
    """
    if checkpoint_path is None:
        # Download from HuggingFace
        model_name = "brandonsmart/splatt3r_v1.0"
        filename = "epoch=19-step=1200.ckpt"
        print(f"Downloading Splatt3R checkpoint from {model_name}")
        checkpoint_path = hf_hub_download(repo_id=model_name, filename=filename)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading Splatt3R model from {checkpoint_path}")
    model = MAST3RGaussians.load_from_checkpoint(checkpoint_path, device)
    model.eval()

    return model


class Splatt3RInference:
    """
    Wrapper class for Splatt3R inference in SLAM context.
    """

    def __init__(self, model, device="cuda"):
        """
        Initialize Splatt3R inference wrapper.

        Args:
            model: MAST3RGaussians model
            device: Device for inference
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_gaussians(self, view1, view2):
        """
        Predict 3D Gaussians from two views.

        Args:
            view1: First view dict with 'img', 'original_img', 'true_shape'
            view2: Second view dict with 'img', 'original_img', 'true_shape'

        Returns:
            Tuple of (pred1, pred2) containing Gaussian parameters
        """
        # Ensure inputs are on correct device
        for key in ["img", "original_img"]:
            if key in view1:
                view1[key] = view1[key].to(self.device)
            if key in view2:
                view2[key] = view2[key].to(self.device)

        # Run inference
        pred1, pred2 = self.model(view1, view2)

        return pred1, pred2

    @torch.no_grad()
    def encode_image(self, img, img_shape):
        """
        Encode a single image using the model's encoder.

        Args:
            img: Input image tensor
            img_shape: Shape of the image

        Returns:
            Encoded features, positions, and shape
        """
        img = img.to(self.device)
        feat, pos, shape = self.model.encoder._encode_image(img, img_shape)
        return feat, pos, shape
