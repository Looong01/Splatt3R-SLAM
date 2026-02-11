# Splatt3R Core Module
# This module contains the Splatt3R model and utilities for 3D Gaussian Splatting

import sys
import os

# Add the mast3r_src to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_current_dir, 'src', 'mast3r_src'))
sys.path.insert(0, os.path.join(_current_dir, 'src', 'mast3r_src', 'dust3r'))
sys.path.insert(0, os.path.join(_current_dir, 'src', 'pixelsplat_src'))

from .splatt3r_model import load_splatt3r_model, Splatt3RInference

__all__ = ['load_splatt3r_model', 'Splatt3RInference']
