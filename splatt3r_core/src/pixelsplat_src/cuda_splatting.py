from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from torch import Tensor

try:
    from .projection import get_fov, homogenize_points
except ImportError:
    from projection import get_fov, homogenize_points


def get_projection_matrix(
    near,
    far,
    fov_x,
    fov_y,
):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    scale_invariant: bool = True,
    use_sh: bool = True,
    return_extras: bool = False,
):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # The compiled CUDA rasterizer extension (thirdparty/diff-gaussian-
    # rasterization-modified) is hardcoded to torch::kFloat32 throughout --
    # rasterize_points.cu calls .data<float>()/.data_ptr<float>() on every
    # tensor argument, with no AT_DISPATCH/templating for other dtypes
    # (unlike the RoPE kernel in croco/models/curope, which was patched to
    # support Half/BFloat16). Under precision="bf16-mixed"/"16-mixed"
    # training these tensors would otherwise arrive here as bf16/fp16
    # (produced by autocast-covered ops upstream in the Gaussian head),
    # which hits a dtype-mismatch error inside the extension.
    #
    # `.float()` alone is NOT enough: an explicit cast to float32 does not
    # survive the next autocast-covered op. `full_projection = view_matrix
    # @ projection_matrix` below is a matmul, and PyTorch's autocast
    # intercepts matmul calls and casts them to bf16 *regardless of the
    # input tensors' actual dtype* while an autocast region is active
    # (Lightning's precision="bf16-mixed" wraps the whole training/
    # validation step in one) -- confirmed by hitting exactly this in
    # practice: `.float()`-only got "expected scalar type Float but found
    # BFloat16" right at the rasterizer call, from `projmatrix` silently
    # turning back into BFloat16 after the matmul despite both of its
    # inputs having been cast to float32 immediately before. Disabling
    # autocast for this whole block is what actually pins it: no op inside
    # gets reduced-precision treatment, matmul included, independent of
    # the ambient training precision. Autograd still casts gradients back
    # to the original (bf16) dtype automatically on the way out, so this
    # doesn't block the mixed-precision memory savings upstream in the
    # backbone -- it only keeps this one call's own math in fp32, which is
    # all the extension can accept anyway.
    with torch.autocast(device_type="cuda", enabled=False):
        extrinsics = extrinsics.float()
        intrinsics = intrinsics.float()
        near = near.float()
        far = far.float()
        background_color = background_color.float()
        gaussian_means = gaussian_means.float()
        gaussian_covariances = gaussian_covariances.float()
        gaussian_sh_coefficients = gaussian_sh_coefficients.float()
        gaussian_opacities = gaussian_opacities.float()

        # Make sure everything is in a range where numerical issues don't appear.
        if scale_invariant:
            scale = 1 / near
            extrinsics = extrinsics.clone()
            extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
            gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
            gaussian_means = gaussian_means * scale[:, None, None]
            near = near * scale
            far = far * scale

        _, _, _, n = gaussian_sh_coefficients.shape
        degree = isqrt(n) - 1
        shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

        b, _, _ = extrinsics.shape
        h, w = image_shape

        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        tan_fov_x = (0.5 * fov_x).tan()
        tan_fov_y = (0.5 * fov_y).tan()

        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        full_projection = view_matrix @ projection_matrix

        all_images = []
        all_radii = []
        all_mean_grads = []
        for i in range(b):
            # Set up a tensor for the gradients of the screen-space means.
            mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
            try:
                mean_gradients.retain_grad()
            except Exception:
                pass

            settings = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=tan_fov_x[i].item(),
                tanfovy=tan_fov_y[i].item(),
                bg=background_color[i],
                scale_modifier=1.0,
                viewmatrix=view_matrix[i],
                projmatrix=full_projection[i],
                sh_degree=degree,
                campos=extrinsics[i, :3, 3],
                prefiltered=False,  # This matches the original usage.
                # TEMPORARY DIAGNOSTIC (see the splatt3r-lora-finetuning
                # skill): flip back to False once the real crash site is
                # found. Even with CUDA_LAUNCH_BLOCKING=1, the recurring
                # illegal-memory-access crash keeps surfacing at whatever
                # innocuous line happens to run next (an .item() call, a
                # zeros_like -- neither should be capable of causing this
                # on their own), which means the actual bad write is
                # inside the extension's own CUDA kernels and doesn't
                # trip an invalid-address fault until some later,
                # unrelated allocation touches the same corrupted memory.
                # debug=True makes CHECK_CUDA (thirdparty/diff-gaussian-
                # rasterization-modified/cuda_rasterizer/auxiliary.h:188)
                # synchronize and check for an error after every internal
                # kernel launch, printing the exact __FILE__/__LINE__
                # inside the extension's own .cu source where it actually
                # happened, instead of deferring. This is what should
                # finally answer "which kernel, and why" instead of
                # continuing to guess from Python-side symptoms.
                debug=True,
            )
            rasterizer = GaussianRasterizer(settings)

            row, col = torch.triu_indices(3, 3)

            image, radii = rasterizer(
                means3D=gaussian_means[i],
                means2D=mean_gradients,
                shs=shs[i] if use_sh else None,
                colors_precomp=None if use_sh else shs[i, :, 0, :],
                opacities=gaussian_opacities[i, ..., None],
                cov3D_precomp=gaussian_covariances[i, :, row, col],
            )
            all_images.append(image)
            all_radii.append(radii)
            all_mean_grads.append(mean_gradients)
        if return_extras:
            # INRIA's adaptive density control keys off the screen-space
            # position gradient and the projected radius, both of which are
            # produced here and were otherwise discarded. Returned only on
            # request so the training path's signature and behaviour are
            # untouched -- scripts/refine_gaussian_map.py is the only caller
            # that needs them.
            return torch.stack(all_images), torch.stack(all_radii), all_mean_grads
        return torch.stack(all_images)


def render_cuda_orthographic(
    extrinsics,
    width,
    height,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    fov_degrees,
    use_sh: bool = True,
    dump: dict | None = None,
):
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    gaussian_means,
    gaussian_covariances,
    gaussian_opacities,
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
):
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
    )
    return result.mean(dim=1)
