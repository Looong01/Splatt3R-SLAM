import einops
import torch

from utils.geometry import unproject_depth, world_space_to_camera_space, camera_space_to_pixel_space


@torch.no_grad()
def calculate_in_frustum_mask(depth_1, intrinsics_1, c2w_1, depth_2, intrinsics_2, c2w_2):
    """
    A function that takes in the depth, intrinsics and c2w matrices of two sets
    of views, and then works out which of the pixels in the first set of views
    has a direct corresponding pixel in any of views in the second set

    Args:
        depth_1: (b, v1, h, w)
        intrinsics_1: (b, v1, 3, 3)
        c2w_1: (b, v1, 4, 4)
        depth_2: (b, v2, h, w)
        intrinsics_2: (b, v2, 3, 3)
        c2w_2: (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    # Same autocast-under-matmul trap as src/pixelsplat_src/cuda_splatting.py:
    # render_cuda() (see the splatt3r-lora-finetuning skill for the full
    # writeup and a standalone repro) -- unproject_depth/world_space_to_
    # camera_space/camera_space_to_pixel_space below are matmul/einsum-heavy,
    # so under an active autocast region (Lightning's precision="bf16-mixed"
    # wraps the whole train/validation step) `camera_points` -- and hence
    # `rendered_depth` -- comes out BFloat16 even though depth_1/depth_2
    # arrive here as plain float32 from the dataloader. Meanwhile
    # `sampled_depths` (from grid_sample below) stays float32, so
    # torch.isclose(rendered_depth, sampled_depths) hits `RuntimeError:
    # BFloat16 did not match Float`. This whole function is already
    # @torch.no_grad() -- it only ever produces a boolean mask, nothing
    # here is part of the differentiable loss -- so there's no tradeoff to
    # disabling autocast for all of it, unlike render_cuda() where fp32 was
    # scoped tightly to avoid undoing the backbone's memory savings.
    with torch.autocast(device_type="cuda", enabled=False):
        _, v1, h, w = depth_1.shape
        _, v2, _, _ = depth_2.shape

        # Unproject the depth to get the 3D points in world space
        points_3d = unproject_depth(depth_1[..., None], intrinsics_1, c2w_1)  # (b, v1, h, w, 3)

        # Project the 3D points into the pixel space of all the second views simultaneously
        camera_points = world_space_to_camera_space(points_3d, c2w_2)  # (b, v1, v2, h, w, 3)
        points_2d = camera_space_to_pixel_space(camera_points, intrinsics_2)  # (b, v1, v2, h, w, 2)

        # Calculate the depth of each point
        rendered_depth = camera_points[..., 2]  # (b, v1, v2, h, w)

        # We use three conditions to determine if a point should be masked

        # Condition 1: Check if the points are in the frustum of any of the v2 views
        in_frustum_mask = (
            (points_2d[..., 0] > 0) &
            (points_2d[..., 0] < w) &
            (points_2d[..., 1] > 0) &
            (points_2d[..., 1] < h)
        )  # (b, v1, v2, h, w)
        in_frustum_mask = in_frustum_mask.any(dim=-3)  # (b, v1, h, w)

        # Condition 2: Check if the points have non-zero (i.e. valid) depth in the input view
        non_zero_depth = depth_1 > 1e-6

        # Condition 3: Check if the points have matching depth to any of the v2
        # views torch.nn.functional.grid_sample expects the input coordinates to
        # be normalized to the range [-1, 1], so we normalize first
        points_2d[..., 0] /= w
        points_2d[..., 1] /= h
        points_2d = points_2d * 2 - 1
        matching_depth = torch.ones_like(rendered_depth, dtype=torch.bool)
        for b in range(depth_1.shape[0]):
            for i in range(v1):
                for j in range(v2):
                    depth = einops.rearrange(depth_2[b, j], 'h w -> 1 1 h w')
                    coords = einops.rearrange(points_2d[b, i, j], 'h w c -> 1 h w c')
                    sampled_depths = torch.nn.functional.grid_sample(depth, coords, align_corners=False)[0, 0]
                    matching_depth[b, i, j] = torch.isclose(rendered_depth[b, i, j], sampled_depths, atol=1e-1)

        matching_depth = matching_depth.any(dim=-3)  # (..., v1, h, w)

        mask = in_frustum_mask & non_zero_depth & matching_depth
        return mask


@torch.no_grad()
def calculate_loss_mask(batch):
    '''Calcuate the loss mask for the target views in the batch'''

    target_depth = torch.stack([target_view['depthmap'] for target_view in batch['target']], dim=1)
    target_intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
    target_c2w = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
    context_depth = torch.stack([context_view['depthmap'] for context_view in batch['context']], dim=1)
    context_intrinsics = torch.stack([context_view['camera_intrinsics'] for context_view in batch['context']], dim=1)
    context_c2w = torch.stack([context_view['camera_pose'] for context_view in batch['context']], dim=1)

    target_intrinsics = target_intrinsics[..., :3, :3]
    context_intrinsics = context_intrinsics[..., :3, :3]

    mask = calculate_in_frustum_mask(
        target_depth, target_intrinsics, target_c2w,
        context_depth, context_intrinsics, context_c2w
    )
    return mask
