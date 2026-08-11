"""Encoding of Gaussian parameters into the 3DGS .ply attribute layout.

Extracted from ``evaluate.save_gaussian_map`` so that the round-trip test
(``scripts/test_gaussian_ply_roundtrip.py``) exercises the exact same encode
path instead of a copy. Kept dependency-light (numpy/torch/scipy only) so the
test can import it without pulling in lietorch / the SLAM backends.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

C0 = 0.28209479177387814  # SH band-0 constant, matches sh_utils.SH2RGB

# float64 batched SVD chunk size (number of Gaussians per batch).
SVD_CHUNK = 1_000_000


def svd_rotations_scales(cov):
    """Decompose symmetric PSD covariances (G,3,3) into rotation matrices and
    per-axis scales.

    Sigma = R S^2 R^T is symmetric PSD, so its SVD is U diag(S) Vh with
    U == R and Vh == R^T. The rotation is therefore U alone -- NOT
    ``U @ Vh``, which is R R^T = I and carries no rotation at all.
    (utils/export.py's covariance_to_quaternion_and_scale() has this same
    mistake; this was caught by a round-trip test -- rebuilding the
    covariance from the written quaternion+scale reproduced the input to
    8e-16 with U, but was off by 1.7e-1 with U @ Vh.)

    On CPU, and chunked: a full-density map is tens of millions of
    Gaussians, and a float64 batched SVD over all of them at once is a
    multi-GB spike on whichever device holds them. Each chunk below is
    SVD'd independently, so the peak transient is SVD_CHUNK*3*3 float64
    (~216 MB at 1M) rather than N*3*3.

    Returns (rot_mat, scales): rot_mat is a float64 numpy (G,3,3) array of
    proper rotations (det=+1; SVD reflections are sign-flipped), scales is a
    float64 torch (G,3) tensor of sqrt(eigenvalues).
    """
    cov = cov.cpu()
    Us, Ss = [], []
    for start in range(0, cov.shape[0], SVD_CHUNK):
        U_c, S_c, _Vh_c = torch.linalg.svd(cov[start : start + SVD_CHUNK].double())
        Us.append(U_c)
        Ss.append(S_c)
    U = torch.cat(Us)
    S = torch.cat(Ss)
    scales = torch.sqrt(S.clamp_min(1e-12))
    rot_mat = U.numpy().copy()
    # SVD can return a reflection (det=-1); a rotation must have det=+1,
    # and scipy rejects reflections outright.
    dets = np.linalg.det(rot_mat)
    rot_mat[dets < 0] *= -1.0
    return rot_mat, scales


def encode_gaussians_for_ply(means, cov_tri, rgb, opa, f_rest=None):
    """Encode Gaussian parameters as the float32 attribute matrix of a
    3DGS-style .ply: columns [x,y,z, nx,ny,nz, f_dc_0..2, opacity,
    scale_0..2, rot_0..3], where scales are log-scales, opacity is a logit,
    f_dc inverts SH2RGB, and rot is a (w,x,y,z) quaternion.

    ONLY THE DC BAND IS WRITTEN. There are no ``f_rest_*`` columns, so any
    higher spherical-harmonic bands are dropped: a map optimized at
    ``--sh-degree 3`` silently degrades to view-independent colour on export
    and cannot be round-tripped. Harmless today -- the Splatt3R head emits one
    coefficient per channel (``sh_degree=1`` in the head means one COEFFICIENT,
    not degree 1; see mast3r/catmlp_dpt_head.py:150) and the live renderer
    passes ``colors_precomp`` with ``sh_degree=0`` -- but it is a trap the
    moment SH >= 1 reaches the online pipeline, because nothing downstream can
    tell that the colour became view-independent. ``f_rest`` is accepted purely
    so that case raises instead of silently losing data.

    means:   (G,3) torch float
    cov_tri: (G,6) torch float, upper-triangle of the 3x3 covariance
    rgb:     (G,3) torch float in [0,1]
    opa:     (G,)  torch float in (0,1)
    f_rest:  must be None or empty -- see above
    """
    if f_rest is not None and getattr(f_rest, "numel", lambda: 0)() > 0:
        raise NotImplementedError(
            "encode_gaussians_for_ply writes f_dc only; passing higher SH bands "
            "would silently drop them. Add f_rest_* columns (and the matching "
            "reader in decode_gaussians_from_ply) before exporting an SH>=1 map."
        )
    # (G,6) upper triangle -> full symmetric (G,3,3)
    cov = torch.zeros((cov_tri.shape[0], 3, 3), device=cov_tri.device, dtype=cov_tri.dtype)
    row, col = torch.triu_indices(3, 3)
    cov[:, row, col] = cov_tri
    cov[:, col, row] = cov_tri

    rot_mat, scales = svd_rotations_scales(cov)
    quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)

    means_np = means.cpu().numpy()
    log_scales = np.log(scales.cpu().numpy().clip(min=1e-12))
    # Invert sigmoid; clamp away from the asymptotes so logit stays finite.
    opa_np = opa.cpu().numpy().clip(1e-6, 1 - 1e-6)
    logit_opa = np.log(opa_np / (1.0 - opa_np))[:, None]
    # Invert SH2RGB: rgb = sh0 * C0 + 0.5
    f_dc = (rgb.cpu().numpy() - 0.5) / C0
    normals = np.zeros_like(means_np)

    return np.concatenate(
        [means_np, normals, f_dc, logit_opa, log_scales, quat_wxyz], axis=1
    ).astype(np.float32)


def gaussians_to_ply_element(means, cov_tri, rgb, opa, f_rest=None):
    """encode_gaussians_for_ply + wrap as a PlyElement, with uchar
    red/green/blue columns appended.

    The 3DGS convention stores colour as SH DC coefficients (f_dc_*), which
    generic point-cloud tools (MeshLab, CloudCompare) do not read -- they
    fall back to a single default colour. Appending standard uchar RGB keeps
    the file a valid 3DGS ply (all original properties intact, readers that
    select columns by name are unaffected) while displaying coloured points
    everywhere. RGB is computed from the same [0,1] colour encode inverts,
    so the two columns are consistent by construction.
    """
    from plyfile import PlyElement

    attributes = encode_gaussians_for_ply(means, cov_tri, rgb, opa, f_rest)
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    dtype = [(n_, "f4") for n_ in names] + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(attributes.shape[0], dtype=dtype)
    for i, name in enumerate(names):
        elements[name] = attributes[:, i]
    rgb_u8 = (np.clip(rgb.cpu().numpy(), 0, 1) * 255).round().astype(np.uint8)
    elements["red"] = rgb_u8[:, 0]
    elements["green"] = rgb_u8[:, 1]
    elements["blue"] = rgb_u8[:, 2]
    return PlyElement.describe(elements, "vertex")


def decode_gaussians_from_ply(path, device="cpu", dtype=torch.float32):
    """Inverse of ``encode_gaussians_for_ply``: read a 3DGS .ply back into
    Gaussian parameters.

    Returned in BOTH forms, because the two consumers want different ones:

      * the *stored* pre-activation parameters (``log_scales``, ``quat_wxyz``,
        ``logit_opacity``, ``f_dc``) -- these are what per-scene optimization
        should treat as free variables, exactly as INRIA 3DGS does, so that
        scales stay positive and opacity stays in (0,1) without constraints;
      * the *activated* values (``scales``, ``opacity``, ``rgb``,
        ``covariances``) -- what the rasterizer consumes directly.

    Previously this inverse existed only inline inside
    ``scripts/test_gaussian_ply_roundtrip.py``, so nothing outside the test
    could load a map back. Anything that reads a written map should use this,
    so encode and decode stay a matched pair.

    Returns a dict of torch tensors on ``device``.
    """
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    n = len(v["x"])

    def col(*names):
        return np.stack([np.asarray(v[nm]) for nm in names], axis=1).astype(np.float64)

    means = col("x", "y", "z")
    f_dc = col("f_dc_0", "f_dc_1", "f_dc_2")
    log_scales = col("scale_0", "scale_1", "scale_2")
    quat_wxyz = col("rot_0", "rot_1", "rot_2", "rot_3")
    logit_opa = np.asarray(v["opacity"]).astype(np.float64)

    scales = np.exp(log_scales)
    opacity = 1.0 / (1.0 + np.exp(-logit_opa))
    rgb = f_dc * C0 + 0.5

    # scipy wants (x,y,z,w); the file stores (w,x,y,z).
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    cov = R @ (scales[..., None] ** 2 * np.eye(3)) @ R.transpose(0, 2, 1)

    def t(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    return {
        "n": n,
        # pre-activation (optimize these)
        "means": t(means),
        "log_scales": t(log_scales),
        "quat_wxyz": t(quat_wxyz),
        "logit_opacity": t(logit_opa),
        "f_dc": t(f_dc),
        # activated (render these)
        "scales": t(scales),
        "opacity": t(opacity),
        "rgb": t(rgb),
        "covariances": t(cov),
    }
