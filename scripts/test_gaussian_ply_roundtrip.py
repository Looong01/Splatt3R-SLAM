#!/usr/bin/env python
"""Round-trip test: covariance -> PLY -> decode -> covariance.

Synthesises N random Gaussians, encodes them with the exact same code path
used by splatt3r_slam/evaluate.py::save_gaussian_map (via
splatt3r_slam/gaussian_ply_codec.py::encode_gaussians_for_ply), writes them
to a temporary .ply with plyfile, reads the file back, inverts every
transform, and asserts the reconstructed parameters match the originals.

Run:  python scripts/test_gaussian_ply_roundtrip.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splatt3r_slam.gaussian_ply_codec import C0, encode_gaussians_for_ply

N = 10_000
SEED = 0


def make_random_gaussians(n, seed):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * 5.0
    # Random PSD covariances: A @ A.T + eps*I. Scale A down so typical
    # scales stay in a sane range; keep a modest condition number.
    A = torch.randn(n, 3, 3, generator=g) * 0.2
    cov = A @ A.transpose(-1, -2) + 1e-4 * torch.eye(3).expand(n, 3, 3)
    rgb = torch.rand(n, 3, generator=g)
    # Keep opacity away from the 1e-6 / 1-1e-6 logit clamps so the round
    # trip is lossless for every sample.
    opa = torch.rand(n, generator=g) * (1.0 - 2e-4) + 1e-4
    return means, cov, rgb, opa


def write_ply(path, attributes):
    # Mirrors save_gaussian_map's attribute layout (evaluate.py).
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    elements = np.empty(attributes.shape[0], dtype=[(n_, "f4") for n_ in names])
    for i, name in enumerate(names):
        elements[name] = attributes[:, i]
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def read_and_decode(path):
    """Invert the encode path: exp scales, sigmoid opacity, SH2RGB colour,
    quaternion -> R, Sigma = R diag(s^2) R^T."""
    v = PlyData.read(str(path))["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64)
    logit_opa = v["opacity"].astype(np.float64)
    log_scales = np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    quat_wxyz = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)

    scales = np.exp(log_scales)
    opa = 1.0 / (1.0 + np.exp(-logit_opa))
    rgb = f_dc * C0 + 0.5

    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    cov = R @ (scales[..., None] ** 2 * np.eye(3)) @ R.transpose(0, 2, 1)
    return means, cov, rgb, opa


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    means, cov, rgb, opa = make_random_gaussians(N, SEED)

    # save_gaussian_map consumes the covariance as its (G,6) upper triangle.
    row, col = torch.triu_indices(3, 3)
    cov_tri = cov[:, row, col].float()

    attributes = encode_gaussians_for_ply(
        means.float(), cov_tri, rgb.float(), opa.float()
    )

    with tempfile.TemporaryDirectory() as tmp:
        ply_path = Path(tmp) / "roundtrip.ply"
        write_ply(ply_path, attributes)
        means2, cov2, rgb2, opa2 = read_and_decode(ply_path)

    cov_err = rel_err(cov2, cov.numpy().astype(np.float64))
    means_err = np.abs(means2 - means.numpy()).max()
    rgb_err = np.abs(rgb2 - rgb.numpy()).max()
    opa_err = np.abs(opa2 - opa.numpy()).max()

    print(f"N = {N} Gaussians")
    print(f"covariance relative error : {cov_err:.3e}  (assert < 1e-5)")
    print(f"means max abs error       : {means_err:.3e}  (float32)")
    print(f"rgb max abs error         : {rgb_err:.3e}  (float32)")
    print(f"opacity max abs error     : {opa_err:.3e}  (float32)")

    assert cov_err < 1e-5, f"covariance round-trip error too large: {cov_err}"
    # Everything is stored as float32 in the .ply; eps for O(1..10) values
    # is ~1e-6. Opacity goes through sigmoid(logit(.)) on top of that.
    assert means_err < 1e-4, f"means round-trip error too large: {means_err}"
    assert rgb_err < 1e-5, f"rgb round-trip error too large: {rgb_err}"
    assert opa_err < 1e-5, f"opacity round-trip error too large: {opa_err}"
    print("PASS")


if __name__ == "__main__":
    main()
