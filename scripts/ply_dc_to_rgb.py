"""Convert a 3DGS .ply (f_dc_* SH-DC colour) into a MeshLab-friendly .ply
with uchar red/green/blue properties.

3DGS stores colour as spherical-harmonic DC coefficients:
    rgb = f_dc * C0 + 0.5   (C0 = 0.28209479177387814)
MeshLab does not read f_dc and falls back to a single default colour. This
adds standard uchar RGB columns alongside all original properties, so the
file stays a valid 3DGS ply AND displays coloured points in MeshLab.

Usage:
    python3 scripts/ply_dc_to_rgb.py in.ply [out.ply]
    # out defaults to <in>_rgb.ply
"""
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".ply", "_rgb.ply")
    ply = PlyData.read(src)
    v = ply["vertex"]
    names = list(v.data.dtype.names)
    for need in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if need not in names:
            raise SystemExit(f"{src}: no {need} property -- not a 3DGS ply?")
    rgb = np.clip(v["f_dc_0"] * C0 + 0.5, 0, 1)
    g = np.clip(v["f_dc_1"] * C0 + 0.5, 0, 1)
    b = np.clip(v["f_dc_2"] * C0 + 0.5, 0, 1)
    dtype = [(n, "f4") for n in names] + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    out = np.empty(len(v.data), dtype=dtype)
    for n in names:
        out[n] = v.data[n]
    out["red"] = (rgb * 255).round().astype(np.uint8)
    out["green"] = (g * 255).round().astype(np.uint8)
    out["blue"] = (b * 255).round().astype(np.uint8)
    PlyData([PlyElement.describe(out, "vertex")], text=ply.text).write(dst)
    print(f"wrote {dst} ({len(out)} vertices, +red/green/blue)")


if __name__ == "__main__":
    main()
