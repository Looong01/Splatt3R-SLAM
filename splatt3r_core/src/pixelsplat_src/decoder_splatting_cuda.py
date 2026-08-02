import torch
from einops import rearrange, repeat

try:
    from .cuda_splatting import render_cuda
except ImportError:
    from cuda_splatting import render_cuda
try:
    from utils.geometry import normalize_intrinsics
except ImportError:

    def normalize_intrinsics(intrinsics, image_shape):
        """Fallback: normalize intrinsics matrix by image dimensions."""
        intrinsics = intrinsics.clone()
        intrinsics[..., 0, :] /= image_shape[1]
        intrinsics[..., 1, :] /= image_shape[0]
        return intrinsics


class DecoderSplattingCUDA(torch.nn.Module):

    def __init__(self, background_color, spatial_stride=1):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
        # See the splatt3r-lora-finetuning skill: with debug=True enabled
        # on the rasterizer (cuda_splatting.py), a real training crash's
        # illegal-memory-access was pinned to cuda_rasterizer/
        # rasterizer_impl.cu:398 -- the BACKWARD::render kernel, operating
        # on tile/Gaussian-index state saved from the forward pass. This
        # extension (thirdparty/diff-gaussian-rasterization-modified) is
        # the classic 3D Gaussian Splatting renderer, built assuming
        # sparse, SfM-initialized scenes (thousands of Gaussians total).
        # This pixel-aligned prediction scheme instead renders one
        # Gaussian per pixel -- at this decoder's un-strided 384x384,
        # that is 147,456 Gaussians for a single view, likely overflowing
        # some fixed-capacity internal per-tile counting/indexing the
        # kernel was never designed for at that density. Sanitizing
        # scales/rotations/means/opacities/sh (lora.py) only ever delayed
        # this, never fixed it, because the problem is Gaussian COUNT
        # density, not any individual value. spatial_stride subsamples the
        # H,W prediction grid before rendering -- stride=2 cuts the count
        # by 4x -- mirroring splatt3r_slam/splatt3r_utils.py's
        # spatial_stride, which exists for the same reason on the SLAM
        # inference side.
        self.spatial_stride = spatial_stride

    def forward(self, batch, pred1, pred2, image_shape):

        base_pose = batch["context"][0]["camera_pose"]  # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack(
            [target_view["camera_pose"] for target_view in batch["target"]], dim=1
        )
        intrinsics = torch.stack(
            [target_view["camera_intrinsics"] for target_view in batch["target"]], dim=1
        )
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # Rotate the ground truth extrinsics into the coordinate system used by MAST3R
        # --i.e. in the coordinate system of the first context view, normalized by the scene scale
        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics

        means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)
        opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

        if self.spatial_stride > 1:
            s = self.spatial_stride
            means = means[:, :, ::s, ::s]
            covariances = covariances[:, :, ::s, ::s]
            harmonics = harmonics[:, :, ::s, ::s]
            opacities = opacities[:, :, ::s, ::s]

        b, v, _, _ = extrinsics.shape
        near = torch.full((b, v), 0.1, device=means.device)
        far = torch.full((b, v), 1000.0, device=means.device)

        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(
                rearrange(means, "b v h w xyz -> b (v h w) xyz"),
                "b g xyz -> (b v) g xyz",
                v=v,
            ),
            repeat(
                rearrange(covariances, "b v h w i j -> b (v h w) i j"),
                "b g i j -> (b v) g i j",
                v=v,
            ),
            repeat(
                rearrange(harmonics, "b v h w c d_sh -> b (v h w) c d_sh"),
                "b g c d_sh -> (b v) g c d_sh",
                v=v,
            ),
            repeat(
                rearrange(opacities, "b v h w 1 -> b (v h w)"), "b g -> (b v) g", v=v
            ),
            # scale_invariant=False: `near` above is a hardcoded constant
            # (0.1), never derived from actual scene content, so
            # render_cuda's default scale_invariant=True was applying a
            # fixed, unconditional 10x magnification (scale=1/near) to
            # every camera position and Gaussian, every single sample --
            # not adaptive to anything, just a constant global zoom.
            # Recurring illegal-memory-access crashes (same
            # rasterizer_impl.cu:326, cuda_rasterizer/auxiliary.h's
            # in_frustum()) survived both input sanitization and 4x fewer
            # Gaussians (spatial_stride), which pointed away from "a value
            # is bad" and "there are too many Gaussians" and toward
            # something depth/projection-related: in_frustum() culls
            # anything with camera-space z <= 0.2f, a HARDCODED absolute
            # threshold -- and perspective-projected screen footprint
            # scales with focal_length/z, so a Gaussian sitting just past
            # that cutoff (z slightly > 0.2) can still cover the entire
            # frame. The unconditional 10x stretch this scale_invariant
            # block applied shifts every Gaussian's z by the same 10x,
            # which changes which Gaussians land near that fixed 0.2
            # cutoff in a way that has nothing to do with the actual
            # scene -- since `near` is a constant here, not real
            # scale-adaptive behavior, removing it entirely is a
            # simplification, not a loss of functionality.
            scale_invariant=False,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        return color, None
