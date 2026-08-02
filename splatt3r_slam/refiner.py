"""Online Gaussian-map refinement that keeps the map a FUNCTION OF THE TRAJECTORY.

Why the parameterization is the whole design
--------------------------------------------
Every offline refinement in this project optimizes a world-space map that was
baked from the SLAM run's estimated poses. That bake is where co-adaptation is
created, and it is not a small effect. Same synthetic perturbation, same frozen
map, same protocol, differing only in which poses the map was built from
(splatt3r-finetuning-experiments, 13.12b):

    map built at ground-truth poses   +51.8% of the error recovered
    map built at SLAM poses           -22.2%   -- pushed FURTHER from truth

A map baked at estimated poses has its photometric optimum *at* those poses, so
it actively drags any pose correction back toward the error it was built with.

So nothing here is ever baked. Gaussians are parameters in their owning
keyframe's camera frame; world placement is computed at render time from that
keyframe's *current* pose. When the pose-graph backend corrects a keyframe, the
map re-deforms with the trajectory and every learned value is carried along
rigidly -- there is nothing to un-learn.

The SLAM system already does this for *display* (`SharedKeyframes.gs_*` holds
camera-space Gaussians; `visualization.py` bakes per keyframe with a cache keyed
on `T_WC`). This module extends the same treatment to the optimized map.

Storage, and why it is not in SharedKeyframes
---------------------------------------------
`SharedKeyframes.gs_means` is `(buffer, h*w, 3)` -- exactly one Gaussian per
pixel, fixed. Densification is worth +1.4 dB (19.07 -> 20.51 held-out over 30k
iterations, 13.12c) and would immediately break that. So the refiner owns its
own variable-length storage and never writes the shared Gaussian buffers, which
also keeps exactly one writer per buffer.
"""
import torch

C0 = 0.28209479177387814


def sim3_to_mat(T_WC_data):
    """(...,8) lietorch Sim3 data -> (...,4,4) with the scale folded into sR.

    Mirrors splatt3r_utils._sim3_to_4x4; kept here so the refiner can convert a
    whole batch of keyframe poses at once without a Python loop.
    """
    import lietorch as _lt

    data = T_WC_data.detach()
    if data.dim() == 1:
        data = data.unsqueeze(0)
    t, q, s = data.split([3, 4, 1], dim=-1)
    mat = _lt.SE3(torch.cat([t, q], dim=-1)).matrix()
    mat[..., :3, :3] = mat[..., :3, :3] * s.unsqueeze(-1)
    return mat.to(device=T_WC_data.device, dtype=torch.float32)


class LocalGaussianMap(torch.nn.Module):
    """Every keyframe's Gaussians as one flat parameter set, tagged by keyframe.

    Free variables are pre-activation, as INRIA parameterizes them, so scales
    stay positive and opacity stays in (0,1) with no constraints. They live in
    the OWNING KEYFRAME's camera frame; `world()` composes them through the
    current poses.

    `kf_id` is a buffer, not a parameter: which keyframe a Gaussian belongs to
    is structure, not something to optimize.
    """

    def __init__(self, means, scales, rotations, rgb, opacity, kf_id):
        super().__init__()
        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(scales.clamp_min(1e-8).log())
        # Quaternions arrive in whatever order the Splatt3R head emits and are
        # handed to build_covariance unchanged -- the same path
        # bake_gaussians_world takes. Do NOT "convert" them: the 3DGS .ply
        # convention is (w,x,y,z) and build_covariance wants (x,y,z,w), and
        # mixing the two is exactly the bug that cost 1.32 dB in the offline
        # pipeline (13.12a).
        self.quat = torch.nn.Parameter(rotations)
        self.logit_opacity = torch.nn.Parameter(
            torch.logit(opacity.clamp(1e-6, 1 - 1e-6)))
        self.f_dc = torch.nn.Parameter((rgb - 0.5) / C0)
        self.register_buffer("kf_id", kf_id)

    @property
    def n(self):
        return self.means.shape[0]

    def covariances_local(self):
        from utils.geometry import build_covariance
        q = self.quat / self.quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return build_covariance(self.log_scales.exp(), q)

    def opacity(self):
        return torch.sigmoid(self.logit_opacity)

    def rgb(self):
        return (self.f_dc * C0 + 0.5).clamp(0, 1)

    def world(self, kf_mats):
        """Place every Gaussian using its keyframe's CURRENT pose.

        kf_mats: (K,4,4) from sim3_to_mat, the live keyframe poses. Read them
        fresh on every call -- that is what makes a loop closure free.
        """
        A = kf_mats[self.kf_id, :3, :3]          # (M,3,3), scale folded in
        b = kf_mats[self.kf_id, :3, 3]           # (M,3)
        means_w = torch.einsum("mij,mj->mi", A, self.means) + b
        cov_w = A @ self.covariances_local() @ A.transpose(1, 2)
        return means_w, cov_w

    def param_groups(self, extent, lr_means=1.6e-4, lr_f_dc=2.5e-3,
                     lr_opacity=5e-2, lr_scale=5e-3, lr_rot=1e-3):
        # INRIA's reference rates; the positional one scales with scene extent
        # there and does so here too, even though the means are camera-space --
        # what the rate has to match is the metric size of the scene, which is
        # the same quantity either way.
        return [
            {"params": [self.means], "lr": lr_means * extent, "name": "means"},
            {"params": [self.f_dc], "lr": lr_f_dc, "name": "f_dc"},
            {"params": [self.logit_opacity], "lr": lr_opacity, "name": "opacity"},
            {"params": [self.log_scales], "lr": lr_scale, "name": "scale"},
            {"params": [self.quat], "lr": lr_rot, "name": "rotation"},
        ]


def gaussians_from_keyframe(local, img_tensor, h, w, kf_idx, device,
                            spatial_stride=1, depth_max_percentile=0.98,
                            max_scale=0.5, min_confidence=1.5, min_opacity=0.3):
    """Camera-space Gaussians for one keyframe, ready to become parameters.

    Delegates to splatt3r_utils.prepare_gaussians_local so the refiner starts
    from byte-identical inputs to what the live renderer draws. Note
    inflate_scales_for_stride=False: that inflation compensates for *display*
    subsampling and has no business in an optimized map.
    """
    from splatt3r_slam.splatt3r_utils import prepare_gaussians_local

    prepared = prepare_gaussians_local(
        local, img_tensor, h, w,
        spatial_stride=spatial_stride,
        depth_max_percentile=depth_max_percentile,
        max_scale=max_scale,
        min_confidence=min_confidence,
        min_opacity=min_opacity,
        inflate_scales_for_stride=False,
    )
    if prepared is None:
        return None
    means, scales, rotations, rgb, opas = prepared
    kf_id = torch.full((means.shape[0],), kf_idx, dtype=torch.long, device=device)
    return means, scales, rotations, rgb, opas, kf_id


class SupervisionFrames:
    """Bounded shared store of frames the refiner may supervise on.

    Lives on the CPU (shared-memory uint8), not in CUDA IPC: the tracker
    process writes every tracked frame here and the refiner process samples
    from it, and CPU shared tensors are the only kind that cross process
    boundaries without tying the two processes to the same GPU.

    Poses are stored ANCHOR-RELATIVE, not in world frame: each frame is
    recorded against the keyframe current at its tracking time (the
    FramePoseLog rule, evaluate.py), as (anchor_idx, T_anchor_frame). The
    world pose is composed at SAMPLE time through the anchor's CURRENT pose,
    so when the backend corrects a keyframe, the supervision follows the map
    -- the property the (b') block test measured as the difference between
    refinement preserving a loop closure and undoing it
    (splatt3r-finetuning-experiments, 15.5).

    Why both a recent ring and a reservoir: Gaussians injected with a new
    keyframe are only visible in frames near it in time (a recent window is
    mandatory), while regions the camera has left are never revisited by a
    recent-only sampler (a history sample is mandatory). The (e) sampling
    ablation (skill 15.6) measured the recent-heavy mix as pure downside at
    desk scale -- uniform-over-history won every cell -- so `recent_frac`
    defaults to 0.3 (reservoir-dominant); the ring still exists because a
    long sequence's reservoir evicts, and then the mix question reopens.
    """

    def __init__(self, manager, h, w, n_recent=64, n_reservoir=200,
                 recent_frac=0.3):
        self.n_recent = n_recent
        self.n_reservoir = n_reservoir
        self.recent_frac = recent_frac
        self.lock = manager.RLock()
        self.seen = manager.Value("l", 0)      # frames offered, for reservoir
        self.n_rec = manager.Value("i", 0)
        self.n_res = manager.Value("i", 0)
        self.rec_head = manager.Value("i", 0)
        # uint8 images on CPU shared memory; poses (9,) = [anchor_idx, sim3(8)]
        self.rec_img = torch.zeros(n_recent, h, w, 3, dtype=torch.uint8).share_memory_()
        self.rec_pose = torch.zeros(n_recent, 9).share_memory_()
        self.res_img = torch.zeros(n_reservoir, h, w, 3, dtype=torch.uint8).share_memory_()
        self.res_pose = torch.zeros(n_reservoir, 9).share_memory_()

    def offer(self, img_u8, anchor, rel_data, rng):
        """Add a tracked frame. img_u8: (h,w,3) uint8 CPU; anchor: keyframe
        index the frame was tracked against; rel_data: (8,) T_anchor_frame."""
        row = torch.empty(9)
        row[0] = float(anchor)
        row[1:] = rel_data.reshape(-1).cpu().float()
        with self.lock:
            i = self.rec_head.value
            self.rec_img[i] = img_u8
            self.rec_pose[i] = row
            self.rec_head.value = (i + 1) % self.n_recent
            self.n_rec.value = min(self.n_rec.value + 1, self.n_recent)

            # Algorithm R: keep a uniform sample of everything seen so far.
            n = self.seen.value
            if self.n_res.value < self.n_reservoir:
                j = self.n_res.value
                self.n_res.value += 1
            else:
                j = int(rng.integers(n + 1))
                if j >= self.n_reservoir:
                    j = -1
            if j >= 0:
                self.res_img[j] = img_u8
                self.res_pose[j] = row
            self.seen.value = n + 1

    def sample(self, rng, kf_data, device):
        """One (image float [0,1] (1,3,h,w) on device, world c2w (4,4)) pair,
        or None if empty / the anchor was rolled back. kf_data: (K,8) current
        keyframe Sim3 data, cloned under the keyframes lock by the caller --
        the anchor's correction is picked up HERE, on every sample."""
        with self.lock:
            nr, ns = self.n_rec.value, self.n_res.value
            if nr == 0 and ns == 0:
                return None
            use_recent = (nr > 0) and (ns == 0 or rng.random() < self.recent_frac)
            buf_i, buf_p, n = ((self.rec_img, self.rec_pose, nr) if use_recent
                               else (self.res_img, self.res_pose, ns))
            k = int(rng.integers(n))
            img = buf_i[k].float().permute(2, 0, 1) / 255.0
            p = buf_p[k].clone()
        anchor = int(p[0].item())
        if anchor >= kf_data.shape[0]:
            return None  # anchor keyframe rolled back; drop this sample
        import lietorch
        T = lietorch.Sim3(kf_data[anchor:anchor + 1]) * lietorch.Sim3(
            p[1:9].unsqueeze(0).to(kf_data.device))
        c2w = sim3_to_mat(T.data)[0]
        return img[None].to(device), c2w


def _optimizer_for(model, extent, old_opt=None):
    """Adam over the map, carrying existing moments across a size change.

    Appending a keyframe changes the parameter count, and rebuilding the
    optimizer from scratch would zero every moment -- a mistake this project
    has already made once (the offline densification path re-created Adam on
    every split/clone and wiped its state). Old moments are copied into the
    prefix of the new buffers; new Gaussians simply start at zero, which is what
    a fresh parameter should do.
    """
    opt = torch.optim.Adam(model.param_groups(extent), eps=1e-15)
    if old_opt is None:
        return opt
    old_by_name = {g["name"]: g["params"][0] for g in old_opt.param_groups}
    for g in opt.param_groups:
        p_old = old_by_name.get(g["name"])
        st = old_opt.state.get(p_old) if p_old is not None else None
        if st is None or "exp_avg" not in st:
            continue
        p_new = g["params"][0]
        n = min(st["exp_avg"].shape[0], p_new.shape[0])
        ea = torch.zeros_like(p_new)
        eas = torch.zeros_like(p_new)
        ea[:n] = st["exp_avg"][:n]
        eas[:n] = st["exp_avg_sq"][:n]
        opt.state[p_new] = {"step": st["step"].clone() if torch.is_tensor(st["step"])
                            else torch.tensor(float(st["step"])),
                            "exp_avg": ea, "exp_avg_sq": eas}
    return opt


def _optimizer_subset(model, extent, old_opt, keep):
    """Adam after REMOVING Gaussians (dedup): moments are subset by the same
    boolean mask, so surviving Gaussians keep their optimization state."""
    opt = torch.optim.Adam(model.param_groups(extent), eps=1e-15)
    old_by_name = {g["name"]: g["params"][0] for g in old_opt.param_groups}
    for g in opt.param_groups:
        p_old = old_by_name.get(g["name"])
        st = old_opt.state.get(p_old) if p_old is not None else None
        if st is None or "exp_avg" not in st:
            continue
        opt.state[g["params"][0]] = {
            "step": st["step"].clone() if torch.is_tensor(st["step"])
                    else torch.tensor(float(st["step"])),
            "exp_avg": st["exp_avg"][keep].clone(),
            "exp_avg_sq": st["exp_avg_sq"][keep].clone(),
        }
    return opt


def dedup_by_voxel(model, kf_mats, voxel):
    """One de-clustering lifecycle event (skill 15.7): where several
    keyframes' clusters share a voxel, keep the EARLIEST owner's Gaussians.

    Returns (new_model, keep_mask). The caller rebuilds the optimizer with
    _optimizer_subset so surviving Gaussians keep their Adam moments.
    """
    with torch.no_grad():
        means_w, _ = model.world(kf_mats)
        vox = torch.floor(means_w / voxel).long()
        vox = vox - vox.min(0).values
        B = int(vox.max().item()) + 2
        vox_key = (vox[:, 0] * B + vox[:, 1]) * B + vox[:, 2]
        n_kf_ = int(model.kf_id.max().item()) + 1
        order = torch.argsort(vox_key * n_kf_ + model.kf_id)
        vk_s, kf_s = vox_key[order], model.kf_id[order]
        bounds = torch.ones_like(vk_s, dtype=torch.bool)
        bounds[1:] = vk_s[1:] != vk_s[:-1]
        group_id = torch.cumsum(bounds, 0) - 1
        min_kf = kf_s[bounds][group_id]
        keep_sorted = kf_s == min_kf
        keep = torch.empty_like(keep_sorted)
        keep[order] = keep_sorted
        new_model = LocalGaussianMap(
            model.means.detach()[keep].clone(),
            model.log_scales.detach()[keep].exp().clone(),
            model.quat.detach()[keep].clone(),
            model.rgb().detach()[keep].clone(),
            model.opacity().detach()[keep].clone(),
            model.kf_id[keep].clone()).to(model.means.device)
    return new_model, keep


class RefinedMapSnapshot:
    """Double-buffered world-space snapshot the refiner publishes for viewers.

    13 float32 per Gaussian (mean 3, cov upper-triangle 6, rgb 3, opacity 1)
    on CPU shared memory, because it is the only channel that does not pin
    producer and consumer to one GPU. Reader protocol: read `version`, read
    buf[1 - write_idx][:count], read `version` again; retry if it changed.
    (Consumer side is NOT implemented: the headless test box cannot run the
    imgui viewer, so consumption could not be verified -- see the skill.)
    """

    def __init__(self, manager, max_gaussians):
        self.lock = manager.RLock()
        self.version = manager.Value("l", 0)
        self.write_idx = manager.Value("i", 0)
        self.count = manager.Value("l", 0)
        self.buf = [torch.zeros(max_gaussians, 13).share_memory_(),
                    torch.zeros(max_gaussians, 13).share_memory_()]

    def publish(self, model, kf_mats):
        with torch.no_grad():
            means_w, cov_w = model.world(kf_mats)
            row, col = torch.triu_indices(3, 3)
            data = torch.cat(
                [means_w, cov_w[:, row, col], model.rgb(),
                 model.opacity().reshape(-1, 1)], dim=1).cpu().float()
        n = data.shape[0]
        with self.lock:
            w = self.write_idx.value
            self.buf[w][:n] = data
            self.count.value = n
            self.write_idx.value = 1 - w
            self.version.value += 1


def render_map(means_w, cov_w, rgb, opacity, c2w, K, hw, device,
               near=0.01, far=100.0):
    """Render the composed world-space map from one camera pose."""
    from src.pixelsplat_src.cuda_splatting import render_cuda

    h, w = hw
    ext = torch.as_tensor(c2w, dtype=torch.float32, device=device)[None]
    intr = torch.as_tensor(K, dtype=torch.float32, device=device)[None].clone()
    intr[:, 0, :] /= w
    intr[:, 1, :] /= h
    img = render_cuda(
        ext, intr,
        torch.full((1,), near, device=device), torch.full((1,), far, device=device),
        (h, w), torch.zeros((1, 3), device=device),
        means_w[None], cov_w[None],
        ((rgb - 0.5) / C0)[:, :, None][None],
        opacity.reshape(-1)[None],
        use_sh=True)
    return img.reshape(1, 3, h, w)


def _gaussian_window(size=11, sigma=1.5, device="cuda"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(3, 1, size, size).contiguous()


def _ssim(a, b, win):
    import torch.nn.functional as F
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    sb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_ab + c1) * (2 * sab + c2)) /
            ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))).mean()


def save_refined_map(path, model, kf_mats):
    """Write the refined map as a standard 3DGS .ply, composed through the
    keyframes' FINAL poses. Same encode path as evaluate.save_gaussian_map."""
    import numpy as np
    from plyfile import PlyData, PlyElement
    from splatt3r_slam.gaussian_ply_codec import encode_gaussians_for_ply

    with torch.no_grad():
        means_w, cov_w = model.world(kf_mats)
        row, col = torch.triu_indices(3, 3)
        cov_tri = cov_w[:, row, col].float()
        attributes = encode_gaussians_for_ply(
            means_w.float(), cov_tri, model.rgb().float(),
            model.opacity().float().reshape(-1))
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
    print(f"[refiner] wrote {attributes.shape[0]} refined Gaussians -> {path}",
          flush=True)


def run_refiner(cfg, states, keyframes, sup_frames, K, save_path=None,
                iters_per_cycle=32, log_every=200, duty_cycle=0.25,
                device=None, dedup_voxel=0.0, max_gaussians=4_000_000,
                snapshot=None):
    """Refiner process: optimize the map while SLAM keeps running.

    Holds the map in keyframe-LOCAL parameters and re-reads keyframe poses on
    every step, so a pose-graph correction re-deforms the map for free instead
    of invalidating it -- see the module docstring for the measurement that
    makes this the whole design. Supervision poses follow their anchors for
    the same reason (skill 15.5).

    Never writes SharedKeyframes.gs_*: exactly one writer per buffer.

    duty_cycle caps the refiner's share of the GPU: (g) measured an
    unthrottled refiner DOUBLING tracker latency on one card (skill 15.2), so
    after every iteration the process sleeps long enough to keep its
    sustained rate at ~duty_cycle of its own unthrottled rate. Steps are
    dropped, never frames.

    device: where the refiner COMPUTES. The shared buffers live on
    keyframes.device; passing a different device (e.g. "cuda:1" on a two-card
    box, with both visible to this process) moves the steady-state render
    loop off the tracking card entirely -- (g) measured cross-GPU contention
    as noise. Only small per-keyframe copies still touch the tracking card.

    dedup_voxel/max_gaussians: de-clustering lifecycle (skill 15.7). When the
    map exceeds max_gaussians after a keyframe injection, one voxel dedup
    runs and Adam moments are subset to the survivors.
    """
    import time

    import numpy as np

    from splatt3r_slam.config import set_global_config
    from splatt3r_slam.frame import Mode

    set_global_config(cfg)
    data_device = keyframes.device
    device = device or data_device
    rng = np.random.default_rng(0)
    torch.set_grad_enabled(True)

    model = None
    opt = None
    known_kf = 0
    extent = 1.0
    step = 0
    ema_iter = None
    win = _gaussian_window(device=device)
    DSSIM_WEIGHT = 0.2

    while states.get_mode() is not Mode.TERMINATED:
        if states.is_paused():
            time.sleep(0.01)
            continue

        # --- absorb new keyframes (the non-stationarity) ---
        n_kf = len(keyframes)
        if n_kf > known_kf:
            parts = []
            for k in range(known_kf, n_kf):
                local = keyframes.get_gaussians_local(k)
                if local is None:
                    continue
                kf = keyframes[k]
                h, w = kf.img.shape[-2:]
                got = gaussians_from_keyframe(local, kf.img, h, w, k, data_device)
                if got is not None:
                    parts.append(tuple(t.to(device) for t in got))
            if parts:
                new = [torch.cat([p[i] for p in parts]) for i in range(6)]
                if model is None:
                    model = LocalGaussianMap(*new).to(device)
                    extent = float(
                        (new[0].max(0).values - new[0].min(0).values).norm() / 2)
                else:
                    with torch.no_grad():
                        merged = [
                            torch.cat([model.means.detach(), new[0]]),
                            torch.cat([model.log_scales.detach().exp(), new[1]]),
                            torch.cat([model.quat.detach(), new[2]]),
                            torch.cat([model.rgb().detach(), new[3]]),
                            torch.cat([model.opacity().detach(), new[4]]),
                            torch.cat([model.kf_id, new[5]]),
                        ]
                    model = LocalGaussianMap(*merged).to(device)
                opt = _optimizer_for(model, extent, opt)
                print(f"[refiner] +{new[0].shape[0]:,} gaussians "
                      f"(kf {known_kf}..{n_kf - 1}) -> {model.n:,} total",
                      flush=True)
                if dedup_voxel > 0 and model.n > max_gaussians:
                    with keyframes.lock:
                        kf_now = keyframes.T_WC[:n_kf, 0].clone()
                    model, keep_mask = dedup_by_voxel(
                        model, sim3_to_mat(kf_now).to(device), dedup_voxel)
                    opt = _optimizer_subset(model, extent, opt, keep_mask)
                    print(f"[refiner] dedup @{dedup_voxel} m -> {model.n:,} "
                          f"gaussians", flush=True)
            known_kf = n_kf

        if model is None or opt is None:
            time.sleep(0.01)
            continue

        for _ in range(iters_per_cycle):
            # Poses are read FRESH every step. This single line is what makes a
            # loop closure free: nothing in `model` holds world coordinates, so
            # a corrected keyframe simply re-places its own cluster -- and the
            # supervision sample below composes through the same poses, so
            # anchor-carried frames move with their keyframes (skill 15.5).
            with keyframes.lock:
                # Pose math (lietorch compose / sim3_to_mat) MUST run on the
                # shared buffers' own device: lietorch's group ops silently
                # compute garbage on any other device (measured 2026-08-02:
                # Sim3 compose on cuda:1 returns zero translation). Only the
                # resulting 4x4 matrices cross to the compute device.
                kf_data = keyframes.T_WC[:known_kf, 0].clone()
            got = sup_frames.sample(rng, kf_data, device)
            if got is None:
                break
            tgt, c2w = got
            kf_mats = sim3_to_mat(kf_data).to(device)

            t0 = time.perf_counter()
            means_w, cov_w = model.world(kf_mats)
            pred = render_map(means_w, cov_w, model.rgb(), model.opacity(),
                              c2w, K, tgt.shape[-2:], device)
            loss = ((1 - DSSIM_WEIGHT) * (pred - tgt).abs().mean()
                    + DSSIM_WEIGHT * (1 - _ssim(pred.clamp(0, 1), tgt, win)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

            # Duty cycle: sleep enough to hold the sustained rate at
            # ~duty_cycle of the unthrottled rate (EMA of iteration time).
            dt = time.perf_counter() - t0
            ema_iter = dt if ema_iter is None else 0.95 * ema_iter + 0.05 * dt
            if duty_cycle < 1.0 and ema_iter > 0:
                time.sleep(ema_iter * (1.0 - duty_cycle) / max(duty_cycle, 1e-3))

            if step % log_every == 0:
                if snapshot is not None:
                    snapshot.publish(model, kf_mats)
                print(f"[refiner] step {step}  loss {loss.item():.4f}  "
                      f"{model.n:,} gaussians  iter {ema_iter * 1e3:.0f} ms",
                      flush=True)

    if model is not None:
        with keyframes.lock:
            kf_data = keyframes.T_WC[:known_kf, 0].clone()
        kf_mats = sim3_to_mat(kf_data).to(device)
        if snapshot is not None:
            snapshot.publish(model, kf_mats)
        if save_path:
            save_refined_map(save_path, model, kf_mats)
    print(f"[refiner] terminated after {step} steps", flush=True)
