"""
Path setup for splatt3r_slam package.
Adds splatt3r_core's bundled mast3r and dust3r source to sys.path
so that `import mast3r` and `import dust3r` resolve to the local copies
without needing `pip install -e thirdparty/mast3r`.
"""

import sys
import os

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mast3r_src = os.path.join(_root_dir, "splatt3r_core", "src", "mast3r_src")
_dust3r_src = os.path.join(_root_dir, "splatt3r_core", "src", "mast3r_src", "dust3r")

# Insert at the beginning so our bundled copies take priority
for _p in [_mast3r_src, _dust3r_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
