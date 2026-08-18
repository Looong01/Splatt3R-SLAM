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

    def __init__(self, means, scales, rotations, rgb, opacity, kf_id,
                 scale_floor=None, pitch=None):
        super().__init__()
        # Positions, in one of two parameterizations.
        #
        # Default: the mean itself is the parameter, as INRIA has it, with one
        # learning rate for the whole map scaled by scene extent.
        #
        # pitch given: the parameter is a DIMENSIONLESS residual and the mean is
        # `base + pitch * delta`, so one Adam step of size `lr` displaces a
        # Gaussian by `lr * pitch_i` -- a per-Gaussian rate proportional to its
        # own lattice spacing. This cannot be done by scaling gradients: Adam's
        # update is lr * m / (sqrt(v) + eps), which is scale-invariant in the
        # gradient, so only a reparameterization changes the step size.
        #
        # Why it might matter: `lr_means = 1.6e-4 * extent` is 4.8e-4 m/step on
        # 360 against a ~2 mm lattice pitch -- 23% of the spacing per step, on a
        # lattice nothing re-seeds. Freezing (17.13) fixes that by giving up the
        # coverage repair the positional updates also do (base improves
        # hp_alpha 0.041 -> 0.020 over 3000 steps). Scaling by pitch is the
        # hypothesis that both are available at once.
        self._depth_free = False
        self.pitch_lr = pitch is not None
        if self.pitch_lr:
            self.register_buffer("means_base", means)
            self.register_buffer("pitch", pitch)
            self.means_delta = torch.nn.Parameter(torch.zeros_like(means))
        else:
            self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(scales.clamp_min(1e-8).log())
        # Band limit held SEPARATELY from the learned scale, added in
        # quadrature at every forward: s_eff = sqrt(exp(log_scales)^2 +
        # floor^2). Baking the filter into `scales` instead would let the
        # optimizer walk it straight back off -- and it would, because under
        # point-sampled rasterization (scale down, opacity up) is very nearly
        # loss-preserving at the supervision views, so the scale has a free
        # direction for Adam to wander down over thousands of steps, while the
        # gaps that opens are invisible at native rate. Held here it cannot.
        # Identical to no floor when floor=0, gradients included.
        self.register_buffer(
            "scale_floor",
            torch.zeros_like(opacity) if scale_floor is None else scale_floor)
        # Cached as a Python bool: testing it per call would sync the device on
        # the hot path.
        self._has_floor = bool((self.scale_floor > 0).any())
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
        # One learnable log-depth-scale per keyframe, applied to that cluster's
        # LOCAL means -- i.e. it slides the whole cluster along its own view
        # rays, nearer or further.
        #
        # This is aimed at exactly one artifact. The straight-edged seams in the
        # GUI captures are not a colour step (17.4): they are a semi-transparent
        # VEIL whose silhouette is a projected image rectangle, which is what a
        # keyframe cluster placed at slightly wrong depth looks like floating in
        # front of the true surface. Splatt3R predicts metric depth per PAIR,
        # so a per-pair scale error is the natural explanation, and one scalar
        # per keyframe is the smallest parameter that can correct it.
        #
        # Deliberately NOT a per-keyframe pose. The pose gate is closed at
        # fusion time (13.12d: re-anchoring is worth +0.13 dB and re-optimizing
        # poses against a co-adapted map pushes the trajectory further from
        # truth). Scale along the view ray is a different quantity from the
        # trajectory and does not feed back into it.
        self.kf_log_depth = torch.nn.Parameter(
            torch.zeros(int(kf_id.max().item()) + 1 if kf_id.numel() else 1,
                        device=kf_id.device))

    @property
    def n(self):
        return self.means_base.shape[0] if self.pitch_lr else self.means.shape[0]

    @property
    def positions(self):
        """The means as used everywhere -- composition, culling, export."""
        p = (self.means if not self.pitch_lr else
             self.means_base + self.pitch[:, None] * self.means_delta)
        if self._depth_free:
            p = p * self.kf_log_depth[self.kf_id].exp()[:, None]
        return p

    def covariances_local(self, idx=None):
        from utils.geometry import build_covariance
        quat = self.quat if idx is None else self.quat[idx]
        ls = self.log_scales if idx is None else self.log_scales[idx]
        q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return build_covariance(self.effective_scales(idx, ls), q)

    def effective_scales(self, idx=None, ls=None):
        """The scales as rendered: the learned scale with the band limit added
        in quadrature. Everything that reads a scale -- the covariance, the
        cull's footprint bound, the exported .ply -- must go through here, or
        the map on disk stops matching the map that was optimized."""
        if ls is None:
            ls = self.log_scales if idx is None else self.log_scales[idx]
        s = ls.exp()
        if not self._has_floor:
            return s
        fl = self.scale_floor if idx is None else self.scale_floor[idx]
        return torch.sqrt(s * s + (fl * fl)[:, None])

    def opacity(self, idx=None):
        lo = self.logit_opacity if idx is None else self.logit_opacity[idx]
        return torch.sigmoid(lo)

    def rgb(self, idx=None):
        f = self.f_dc if idx is None else self.f_dc[idx]
        return (f * C0 + 0.5).clamp(0, 1)

    def world(self, kf_mats, idx=None):
        """Place every Gaussian using its keyframe's CURRENT pose.

        kf_mats: (K,4,4) from sim3_to_mat, the live keyframe poses. Read them
        fresh on every call -- that is what makes a loop closure free.

        idx: optional index of the Gaussians to compose. Composing all of them
        when one view can only see ~13% is where this optimizer actually spends
        its time -- profiled on 360 (7.27M Gaussians): this function is 17.6% of
        a step forward and its backward pushes the total to ~54%, against 10.4%
        for the rasterizer's forward. The rasterizer already frustum-culls
        internally; what it cannot avoid is us handing it 7.27M composed
        primitives in the first place. Gathering first measured 5.5x
        (1410.9 -> 255.1 ms/step). See visible_subset() for how idx is derived.

        This changes no numerical semantics: every Gaussian dropped has exactly
        zero gradient, so the same iteration count must give a bit-identical
        result. Anything else means the mask is wrong.
        """
        kf_id = self.kf_id if idx is None else self.kf_id[idx]
        pos = self.positions
        means = pos if idx is None else pos[idx]
        # Gathering A per Gaussian materializes (M,3,3) -- 262 MB at 7.27M --
        # even though A takes only K distinct values. Grouping by keyframe run
        # to broadcast one (3,3) per run was tried and is SLOWER (1427.7 vs
        # 1383.7 ms/step on 360, numerically identical to 5e-07): the cost is
        # the matmul itself (M x 27 FLOPs either way), and 46 small kernels
        # plus a final cat lose to two large ones. The gather is not the
        # bottleneck; see 16.10.
        A = kf_mats[kf_id, :3, :3]               # (M,3,3), scale folded in
        b = kf_mats[kf_id, :3, 3]                # (M,3)
        means_w = torch.einsum("mij,mj->mi", A, means) + b
        cov_w = A @ self.covariances_local(idx) @ A.transpose(1, 2)
        return means_w, cov_w

    @torch.no_grad()
    def visible_keyframes(self, kf_mats, c2w, K, hw, margin=1.3, tiles=4):
        """Boolean mask of the Gaussians a view at c2w can plausibly see.

        Block-granularity, not per-Gaussian: 46 x tiles^2 box tests instead of
        7.27M projections, using the `kf_id` every Gaussian already carries.

        Each keyframe's cluster is split into `tiles x tiles` blocks along the
        source image raster (the Gaussians are stored in pixel order, so a
        contiguous slice is a contiguous image patch) and each block is bounded
        by an axis-aligned box in world space. Whole-keyframe boxes are sound
        but far too coarse: measured on 360 they keep 57.6% of the map where
        only 12.5% carries gradient, because one keyframe's cluster spans a
        large volume and its corners land in almost any frustum.

        Deliberately loose in the safe direction: a false positive costs a
        little throughput, a false negative silently drops gradient. Verified
        zero false negatives against the actual gradient support.
        """
        h, w = hw
        # Callers hand poses in whichever form they hold them -- refine_local
        # keeps the estimated trajectory as numpy, run_refiner composes torch.
        c2w = torch.as_tensor(c2w, dtype=torch.float64, device=kf_mats.device)
        w2c = torch.linalg.inv(c2w).float()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        sel = torch.zeros_like(self.kf_id, dtype=torch.bool)
        for k in range(kf_mats.shape[0]):
            m = (self.kf_id == k).nonzero(as_tuple=True)[0]
            if m.numel() == 0:
                continue
            A, b = kf_mats[k, :3, :3], kf_mats[k, :3, 3]
            pw = self.positions[m].detach() @ A.transpose(0, 1) + b
            blocks = torch.tensor_split(torch.arange(m.numel(), device=m.device),
                                        tiles * tiles)
            for bl in blocks:
                if bl.numel() == 0:
                    continue
                q = pw[bl]
                lo, hi = q.min(0).values, q.max(0).values
                corners = torch.stack([
                    torch.stack([lo[0] if (i & 1) == 0 else hi[0],
                                 lo[1] if (i & 2) == 0 else hi[1],
                                 lo[2] if (i & 4) == 0 else hi[2]])
                    for i in range(8)])
                cc = corners @ w2c[:3, :3].transpose(0, 1) + w2c[:3, 3]
                z = cc[:, 2]
                u = fx * cc[:, 0] / z.clamp_min(1e-6) + cx
                v = fy * cc[:, 1] / z.clamp_min(1e-6) + cy
                ok = ((z > 1e-3) & (u > -w * (margin - 1)) & (u < w * margin)
                      & (v > -h * (margin - 1)) & (v < h * margin))
                if bool(ok.any()):
                    sel[m[bl]] = True
        return sel

    @torch.no_grad()
    def visible_subset(self, kf_mats, c2w, K, hw, tiles=4, margin=1.3):
        """Index of the Gaussians a view can plausibly see, or None for all."""
        sel = self.visible_keyframes(kf_mats, c2w, K, hw, margin=margin, tiles=tiles)
        n = int(sel.sum())
        if n == 0 or n == sel.numel():
            return None
        return sel.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def visible_exact(self, kf_mats, c2w, K, hw, near=1e-3, n_sigma=3.4,
                      pad_px=6.0):
        """Per-Gaussian frustum test, with each Gaussian's own 3-sigma footprint.

        16.8 chose block AABBs to avoid "7.27M projections". That premise is
        wrong, and it is the reason lever 1 stopped at 2.1x: projecting a point
        is one 3x3 matvec, ~65 MFLOP for the whole map, while the composition
        it guards is two 3x3 matmuls per Gaussian PLUS a backward. The test was
        never the expensive part -- and the block scheme's own cost is a Python
        loop over 46 x tiles^2 boxes.

        No gather of a per-Gaussian (M,3,3): `kf_id` is non-decreasing (clusters
        are concatenated in keyframe order and every later operation is a
        boolean mask, which preserves order), so each keyframe is a contiguous
        slice and its transform applies to that slice as one small matmul. The
        assert below is what keeps that from silently becoming false.

        Exact where the block test is loose, and still conservative: a Gaussian
        is kept if its centre projects within `n_sigma` of its own largest
        screen-space extent of the image rectangle.

        Two things make the bound sound rather than merely measured:

        - `n_sigma = 3.4`, not 3. `forward.cu:345` drops a contribution when
          alpha < 1/255, and for an opacity near 1 that happens at
          sqrt(2 ln 255) = 3.33 sigma, not 3. A 3-sigma test truncates a shell
          that the rasterizer still renders. It costs ~+2 pp of kept fraction.
        - `smax` is the LARGEST scale axis, i.e. a bounding sphere, so an
          elongated Gaussian at any orientation is over-covered rather than
          under-covered. The 2D conic's off-diagonal term cannot produce a
          false negative against a sphere bound.

        It cannot see occlusion --
        nothing geometric can -- but occlusion turns out to be worth only ~12%
        here (`radii > 0` keeps 9.7% against a measured gradient support of
        8.6% on 360), so the box test's 45.4% was almost all frustum slack, not
        occlusion.
        """
        assert bool((self.kf_id[1:] >= self.kf_id[:-1]).all()), \
            "kf_id must be non-decreasing for the contiguous-slice projection"
        h, w = hw
        c2w = torch.as_tensor(c2w, dtype=torch.float64, device=kf_mats.device)
        w2c = torch.linalg.inv(c2w).float()
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        P = w2c @ kf_mats                                  # (K,4,4) local -> view
        bounds = torch.searchsorted(
            self.kf_id, torch.arange(kf_mats.shape[0] + 1,
                                     device=self.kf_id.device))
        smax = self.effective_scales().detach().max(-1).values
        sel = torch.zeros_like(self.kf_id, dtype=torch.bool)
        for k in range(kf_mats.shape[0]):
            lo, hi = int(bounds[k]), int(bounds[k + 1])
            if hi <= lo:
                continue
            cc = (self.positions[lo:hi].detach() @ P[k, :3, :3].transpose(0, 1)
                  + P[k, :3, 3])
            z = cc[:, 2]
            zc = z.clamp_min(1e-6)
            u = fx * cc[:, 0] / zc + cx
            v = fy * cc[:, 1] / zc + cy
            # The scale is in the keyframe's frame; kf_mats carries a Sim3, so
            # the world/view-space size is scaled by |A| -- use the transform's
            # own scale rather than assuming it is 1.
            s_view = smax[lo:hi] * float(P[k, :3, :3].det().abs() ** (1.0 / 3.0))
            # f*s/z understates the footprint off-axis: the projection Jacobian
            # row for u is (f/z, 0, -f x/z^2), whose norm is (f/z)*sqrt(1+(x/z)^2).
            # Dropping that factor leaves a few hundred border Gaussians per
            # view outside the test that still carry gradient -- small, but
            # lever 1's acceptance bar was ZERO false negatives and this is held
            # to the same bar.
            # pad_px covers what the analytic footprint cannot know about: the
            # rasterizer adds 0.3 to the 2D covariance diagonal before taking
            # its own radius, worth ~1.6 px, and everything here is fp32. Six
            # pixels is cheap insurance -- it moves the kept fraction by ~0.1 pp
            # and it is what takes the false negatives to zero.
            ru = (n_sigma * fx * s_view / zc * (1 + (cc[:, 0] / zc) ** 2).sqrt()
                  + pad_px)
            rv = (n_sigma * fy * s_view / zc * (1 + (cc[:, 1] / zc) ** 2).sqrt()
                  + pad_px)
            sel[lo:hi] = ((z > near) & (u > -ru) & (u < w + ru)
                          & (v > -rv) & (v < h + rv))
        n = int(sel.sum())
        if n == 0 or n == sel.numel():
            return None
        return sel.nonzero(as_tuple=True)[0]

    def param_groups(self, extent, lr_means=1.6e-4, lr_f_dc=2.5e-3,
                     lr_opacity=5e-2, lr_scale=5e-3, lr_rot=1e-3,
                     lr_kf_depth=0.0):
        # INRIA's reference rates; the positional one scales with scene extent
        # there and does so here too, even though the means are camera-space --
        # what the rate has to match is the metric size of the scene, which is
        # the same quantity either way.
        return [
            ({"params": [self.means_delta], "lr": lr_means, "name": "means"}
             if self.pitch_lr else
             {"params": [self.means], "lr": lr_means * extent, "name": "means"}),
            {"params": [self.f_dc], "lr": lr_f_dc, "name": "f_dc"},
            {"params": [self.logit_opacity], "lr": lr_opacity, "name": "opacity"},
            {"params": [self.log_scales], "lr": lr_scale, "name": "scale"},
            {"params": [self.quat], "lr": lr_rot, "name": "rotation"},
            {"params": [self.kf_log_depth], "lr": lr_kf_depth,
             "name": "kf_depth"},
        ]


class ViewMaskCache:
    """Per-view cache of the Gaussians that actually carry gradient.

    Lever 1 (`visible_keyframes`) is pure geometry and stops at 2.1x because an
    axis-aligned box cannot represent occlusion: it keeps 45.4% of the map where
    the measured gradient support is 12.5% (skill 16.8). That support is free to
    observe -- after a backward, exactly the Gaussians that contributed colour to
    some pixel have a nonzero `f_dc` gradient. Record it per supervision view,
    submit only that set on the view's next visits, and re-observe every
    `refresh` visits so a Gaussian whose opacity has since grown can re-enter.

    NOT `radii > 0`, which 16.11 proposed and which is wrong: `radii` is written
    in the rasterizer's preprocess from the frustum test and the projected
    extent alone, so it is blind to occlusion -- the exact thing geometry
    already fails at. Both signals are measured side by side in
    scripts/bench_mask_cache.py.

    Unlike lever 1, this is an approximation with a real failure mode: between
    two observations of a view, a Gaussian that should have begun contributing
    receives no gradient. Staleness is bounded by `refresh` and by the union
    below, never eliminated, which is why every arm is scored on held-out
    quality and not only on it/s.
    """

    def __init__(self, refresh=8, union=True):
        self.refresh = refresh
        self.union = union
        self.mask = {}
        self.prev = {}
        self.visits = {}
        self.n_observe = 0
        self.n_cached = 0

    def take(self, vid):
        """(cached index or None, observe?) for this view's next step.

        None means the caller should submit its own fallback set -- the
        geometric cull, which has zero false negatives, not the whole map.

        The refresh counter is PER VIEW, not global: with ~50 views a global
        counter would refresh whichever view happened to land on the multiple
        and starve the rest.
        """
        v = self.visits.get(vid, 0)
        self.visits[vid] = v + 1
        if vid not in self.mask or v % max(self.refresh, 1) == 0:
            self.n_observe += 1
            return None, True
        self.n_cached += 1
        return self.mask[vid], False

    @torch.no_grad()
    def observe(self, vid, model, submitted):
        """Record the support, after backward() and before the next zero_grad.

        `submitted` is the index that was actually rendered (None = whole map);
        the support is intersected with it so a stale entry can never survive by
        having been left out of this step.
        """
        g = model.f_dc.grad
        if g is None:
            return
        sup = g.abs().sum(-1) > 0
        if submitted is not None:
            keep = torch.zeros_like(sup)
            keep[submitted] = True
            sup &= keep
        if self.union and vid in self.prev:
            sup |= self.prev[vid]
        self.prev[vid] = sup.clone()
        self.mask[vid] = sup.nonzero(as_tuple=True)[0]

    def drop(self):
        """Invalidate everything. Call after any event that moves Gaussians
        relative to cameras -- a pose-graph correction, a dedup, an injection
        that renumbers the map."""
        self.mask.clear()
        self.prev.clear()

    def stats(self):
        n = self.n_observe + self.n_cached
        return (f"{self.n_cached}/{n} cached steps, "
                f"{sum(v.numel() for v in self.mask.values()) / max(len(self.mask), 1):,.0f} "
                f"gaussians/view")


def gaussians_from_keyframe(local, img_tensor, h, w, kf_idx, device,
                            spatial_stride=1, depth_max_percentile=0.98,
                            max_scale=0.5, min_confidence=1.5, min_opacity=0.3,
                            aa_sigma_scale=0.0, aa_compensate_opacity=False,
                            max_anisotropy=0.0, streak_opacity=0.0,
                            hard_floor=False, want_conf=False):
    """Camera-space Gaussians for one keyframe, ready to become parameters.

    Delegates to splatt3r_utils.prepare_gaussians_local so the refiner starts
    from byte-identical inputs to what the live renderer draws. Note
    inflate_scales_for_stride=False: that inflation compensates for *display*
    subsampling and has no business in an optimized map.

    hard_floor: return the band limit as a SEVENTH element (the per-Gaussian
    scale floor) instead of folding it into the scales. The two differ only
    once the optimizer runs -- folded in, the floor is an initial condition the
    optimizer may walk off; returned separately it becomes a constraint. Which
    of those is true matters and is measured, not assumed.
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
        aa_sigma_scale=0.0 if hard_floor else aa_sigma_scale,
        aa_compensate_opacity=aa_compensate_opacity,
        max_anisotropy=max_anisotropy,
        streak_opacity=streak_opacity,
        return_pitch=hard_floor,
        return_conf=want_conf,
    )
    if prepared is None:
        return None
    means, scales, rotations, rgb, opas = prepared[:5]
    rest = list(prepared[5:])
    pitch = rest.pop(0) if hard_floor else None
    conf = rest.pop(0) if want_conf else None
    kf_id = torch.full((means.shape[0],), kf_idx, dtype=torch.long, device=device)
    out = (means, scales, rotations, rgb, opas, kf_id)
    if hard_floor:
        out = out + (aa_sigma_scale * pitch,)
    if want_conf:
        # appended LAST so the existing index contract (got[6] is the band
        # limit) is untouched for every caller that does not ask for conf
        out = out + (conf,)
    return out


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


def _optimizer_for(model, extent, old_opt=None, lr_means=1.6e-4):
    """Adam over the map, carrying existing moments across a size change.

    Appending a keyframe changes the parameter count, and rebuilding the
    optimizer from scratch would zero every moment -- a mistake this project
    has already made once (the offline densification path re-created Adam on
    every split/clone and wiped its state). Old moments are copied into the
    prefix of the new buffers; new Gaussians simply start at zero, which is what
    a fresh parameter should do.
    """
    opt = torch.optim.Adam(model.param_groups(extent, lr_means=lr_means),
                           eps=1e-15)
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


def _optimizer_subset(model, extent, old_opt, keep, lr_means=1.6e-4):
    """Adam after REMOVING Gaussians (dedup): moments are subset by the same
    boolean mask, so surviving Gaussians keep their optimization state."""
    opt = torch.optim.Adam(model.param_groups(extent, lr_means=lr_means),
                           eps=1e-15)
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


def cross_cluster_loss(model, kf_mats, voxel, idx=None, max_voxels=200_000,
                       w_pos=1.0, w_rgb=1.0, generator=None):
    """Penalize two clusters that cover one surface and disagree about it.

    This is the only repair route that needs no external truth (skill 17.29,
    after four attempts to import sensor depth all degraded the map). It does
    not need to know the correct depth -- only that two keyframes' Gaussians
    occupy the same voxel and disagree about where the surface is and what
    colour it is. A seam IS that disagreement (17.4: a veil, not a colour step),
    and disagreement is removable without new information (17.27).

    Why this succeeds where the photometric loss cannot: at a held-out baseline
    of 0.057 m a residual depth error projects to 0.66 px (17.17), so the
    supervision term is blind to it by construction. This term is computed in
    3D between clusters and never looks through a camera at all.

    Note this is NOT voxel dedup, which deleted 8% of the map and changed
    nothing (16.6). Deleting picks one of two disagreeing measurements; this
    moves them toward each other, which is a different operation with a
    different gradient.

    Gradient reaches `kf_log_depth` (sliding a whole cluster along its own view
    rays) and, if positions are free, `means`. Pairing it with `kf_log_depth` is
    the point: 17.16 measured that parameter finding a real signal and buying
    +0.003 dB, because photometry could not tell it which way to slide.
    """
    sel = slice(None) if idx is None else idx
    pos = model.positions[sel]
    kf_id = model.kf_id[sel]
    A = kf_mats[kf_id, :3, :3]
    b = kf_mats[kf_id, :3, 3]
    world = torch.einsum("mij,mj->mi", A, pos) + b
    rgb = model.rgb(idx)

    with torch.no_grad():
        v = torch.floor(world.detach() / voxel).long()
        v = v - v.min(0).values
        B = int(v.max().item()) + 2
        key = (v[:, 0] * B + v[:, 1]) * B + v[:, 2]
        # Keep only voxels that actually contain more than one keyframe --
        # a voxel occupied by a single cluster has nothing to disagree about
        # and would otherwise contribute a spurious within-cluster smoothness
        # penalty, which is a different (and unwanted) regularizer.
        order = torch.argsort(key)
        ks, ids = key[order], kf_id[order]
        start = torch.ones_like(ks, dtype=torch.bool)
        start[1:] = ks[1:] != ks[:-1]
        gid = torch.cumsum(start, 0) - 1
        n_g = int(gid[-1]) + 1
        kmin = torch.full((n_g,), 1 << 30, device=ks.device, dtype=ids.dtype)
        kmax = torch.full((n_g,), -1, device=ks.device, dtype=ids.dtype)
        kmin.scatter_reduce_(0, gid, ids, reduce="amin")
        kmax.scatter_reduce_(0, gid, ids, reduce="amax")
        mixed = kmax > kmin                      # >= 2 distinct keyframes
        keep_g = mixed.nonzero(as_tuple=True)[0]
        if keep_g.numel() == 0:
            return world.sum() * 0.0
        if keep_g.numel() > max_voxels:
            pick = torch.randperm(keep_g.numel(), device=keep_g.device,
                                  generator=generator)[:max_voxels]
            keep_g = keep_g[pick]
        remap = torch.full((n_g,), -1, device=ks.device, dtype=torch.long)
        remap[keep_g] = torch.arange(keep_g.numel(), device=ks.device)
        g_of = remap[gid]
        take = g_of >= 0
        rows = order[take]
        grp = g_of[take]
        cnt = torch.zeros(keep_g.numel(), device=ks.device).index_add_(
            0, grp, torch.ones_like(grp, dtype=torch.float32)).clamp_min(1)

    # BETWEEN-cluster variance, not total. A mixed voxel's total variance is
    # within + between, and the within part is legitimate surface structure
    # (texture, curvature) inside one cluster -- it dominates and dilutes the
    # signal to nothing. Measured: penalizing total variance moved the warp
    # deficit -4.3% and saturated across a 10x voxel sweep (17.30). What a seam
    # actually is, is two clusters' MEANS disagreeing, so the per-(voxel,
    # keyframe) means are the quantity, and their spread within a voxel is the
    # loss.
    with torch.no_grad():
        # Dense (voxel, keyframe) pair index. kf ids are small (tens), so a
        # pair key fits comfortably and needs no hashing.
        n_kf = int(model.kf_id.max().item()) + 1
        pair = grp * n_kf + kf_id[rows]
        upair, pinv = torch.unique(pair, return_inverse=True)
        p_grp = torch.div(upair, n_kf, rounding_mode="floor")
        p_cnt = torch.zeros(upair.numel(), device=pair.device).index_add_(
            0, pinv, torch.ones_like(pinv, dtype=torch.float32)).clamp_min(1)
        g_np = torch.zeros(cnt.numel(), device=pair.device).index_add_(
            0, p_grp, torch.ones_like(p_grp, dtype=torch.float32)).clamp_min(1)

    def between_var(x):
        xr = x[rows]
        # mean per (voxel, keyframe)
        pm = torch.zeros((upair.numel(), xr.shape[1]), device=xr.device,
                         dtype=xr.dtype).index_add_(0, pinv, xr) / p_cnt[:, None]
        # mean over keyframes within a voxel, then the spread about it
        gm = torch.zeros((cnt.numel(), xr.shape[1]), device=xr.device,
                         dtype=xr.dtype).index_add_(0, p_grp, pm) / g_np[:, None]
        d = pm - gm[p_grp]
        return (d * d).sum(1).mean()

    return w_pos * between_var(world) + w_rgb * between_var(rgb)


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
            model.positions.detach()[keep].clone(),
            model.log_scales.detach()[keep].exp().clone(),
            model.quat.detach()[keep].clone(),
            model.rgb().detach()[keep].clone(),
            model.opacity().detach()[keep].clone(),
            model.kf_id[keep].clone(),
            # The floor is per-Gaussian and must survive dedup, or the
            # survivors quietly lose their band limit.
            scale_floor=model.scale_floor[keep].clone(),
            pitch=model.pitch[keep].clone() if model.pitch_lr else None
        ).to(model.scale_floor.device)
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
        cap = self.buf[0].shape[0]
        if n > cap:
            # Display-only channel: a map larger than the buffer is
            # stride-subsampled to fit, never an error -- the refined .ply
            # (the actual artifact) is written separately and unaffected.
            stride = (n + cap - 1) // cap
            data = data[::stride]
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
    keyframes' FINAL poses. Same encode path as evaluate.save_gaussian_map,
    with uchar red/green/blue appended for generic viewers."""
    from plyfile import PlyData
    from splatt3r_slam.gaussian_ply_codec import gaussians_to_ply_element

    with torch.no_grad():
        means_w, cov_w = model.world(kf_mats)
        row, col = torch.triu_indices(3, 3)
        cov_tri = cov_w[:, row, col].float()
        element = gaussians_to_ply_element(
            means_w.float(), cov_tri, model.rgb().float(),
            model.opacity().float().reshape(-1))
    PlyData([element]).write(str(path))
    print(f"[refiner] wrote {element.count} refined Gaussians -> {path}",
          flush=True)


def run_refiner(cfg, states, keyframes, sup_frames, K, save_path=None,
                iters_per_cycle=32, log_every=200, duty_cycle=0.25,
                device=None, dedup_voxel=0.0, max_gaussians=4_000_000,
                snapshot=None, aa_sigma=0.0, freeze_means=False, cull=True,
                min_confidence=1.5, polish_flag=None, unfreeze_in_polish=False,
                polish_done=None, polish_tol=0.0, polish_patience=3,
                streak_opacity=0.0, uniform_fade=0.0, scale_cap=0.0,
                conf_fade=0.0):
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

    aa_sigma / freeze_means / cull.

    **aa_sigma and freeze_means default OFF, against what 17.12 concluded.**
    Those conclusions come from scripts/refine_local.py standing in for this
    loop, and a matched pair of full online runs did not reproduce them (17.18).
    Measured here, on the refined map, desk:

        old (both off)          psnr 12.0971  lpips 0.5446
        + aa_sigma only              12.0307        0.5629
        + freeze_means only          11.3756        0.5471
        + min_confidence 0 only      11.5596        0.6054
        all three                     9.3865        0.7327   <- not additive

    Individually mild, jointly -2.71 dB against a -1.33 dB sum. Something in
    this loop that the offline proxy does not have -- most likely that the map
    is still GROWING, so late Gaussians get almost none of the ~104 in-sequence
    steps while early ones are fitted to early frames and never revisited --
    interacts with all three. Until that is understood, this ships what measures
    best in the system it ships in, not what measured best in the proxy.

    `cull` stays on: it is a pure throughput change with zero false negatives
    (17.1, 17.10) and is unaffected by any of the above.

    streak_opacity: hide ray-elongated Gaussians by lowering their opacity in
    proportion to how long they are relative to their own lattice pitch. Worth
    −5.0% lpips OFFLINE at K=0.5 for 0.062 dB and +0.31 pp of black (17.32).
    Defaults OFF here because 17.21 established that offline quality levers do
    not transfer to this loop, and this one has not been tested in it — the
    whole point of that section is that the proxy cannot license a default.

    polish_flag / unfreeze_in_polish: the post-sequence polish phase is
    invisible to this process -- main.py simply sleeps while this keeps running
    -- so a shared flag is the only way for it to know. With unfreeze_in_polish,
    the positional rate is restored when the flag is set. That is Kimi's
    decisive test for the online lpips deficit (17.22): if the deficit comes
    from frozen geometry forcing multi-view colour conflict that L1 resolves by
    averaging, then unfreezing once the map is complete should recover most of
    it; if it does not move, that explanation is falsified.

    polish_done / polish_tol: stop the polish early once the training loss
    plateaus, instead of burning a fixed wall clock. The step counts different
    sequences actually need vary by 2.4x (desk 1581, room 1856, 360 3821 --
    17.25), so a fixed duration either wastes time on the easy ones or starves
    the hard ones. polish_tol is the relative improvement over one window below
    which the map is called converged; 0 disables and keeps the fixed-seconds
    behaviour, which is what ships until this is measured.

    The flags remain, so the offline settings are one argument away. What
    17.12's numbers describe, for reference:

      aa_sigma=0.5    band-limits each Gaussian to its own lattice pitch, held
                      as a constraint (scale_floor) rather than an initial
                      condition. Without it the map is perforated between its
                      own sample points and every render above the source
                      sampling rate shows a halftone lattice (17.2).
      freeze_means    at 300 steps the positional updates buy 0.35 dB of psnr
                      and COST 9.7% of lpips; frozen, lpips improves 9.3% over
                      the initial map instead of 0.4% worse. This is the online
                      lpips regression in docs/online-eval-all-families.md, and
                      this flag is its fix. The sign flips by 3000 steps, so an
                      offline run should set freeze_means=False.
      cull            per-Gaussian frustum test, 8-9x throughput, zero false
                      negatives (17.1, 17.10).

    Per-frame exposure is deliberately NOT wired here: measured inert at this
    budget (+0.003 dB, -0.2% lpips), because 6 parameters per frame cannot
    converge in ~6 visits (17.12).
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
                got = gaussians_from_keyframe(
                    local, kf.img, h, w, k, data_device,
                    min_confidence=min_confidence,
                    aa_sigma_scale=aa_sigma, hard_floor=aa_sigma > 0,
                    streak_opacity=streak_opacity,
                    want_conf=conf_fade > 0)
                if got is not None and (uniform_fade > 0 or scale_cap > 0
                                        or conf_fade > 0):
                    # 17.58: the base checkpoint predicts opacity 1.0 almost
                    # everywhere and head training inflates the scale tail.
                    # These are the injection-time corrections, applied HERE
                    # rather than baked into the checkpoint so they can be armed
                    # per map.
                    got = list(got)
                    if scale_cap > 0:
                        got[1] = got[1].clamp(max=scale_cap)
                    if uniform_fade > 0:
                        got[4] = got[4] * (1.0 - uniform_fade)
                    if conf_fade > 0:
                        # 17.66.7: same mean dose as uniform_fade, allocated by
                        # the backbone's own confidence. Rank-normalized WITHIN
                        # the keyframe because the confidence head is not
                        # calibrated across frames. Floored at 0.1 so that
                        # D >= 0.5 stays a thinning prior rather than becoming a
                        # deletion prior (17.66.6).
                        conf = got.pop()
                        r = torch.argsort(torch.argsort(conf)).float()
                        cn = r / max(len(r) - 1, 1)
                        got[4] = got[4] * (
                            1.0 - conf_fade * 2.0 * (1.0 - cn)).clamp(0.1, 1.0)
                    got = tuple(got)
                if got is not None:
                    parts.append(tuple(t.to(device) for t in got))
            if parts:
                n_field = 7 if aa_sigma > 0 else 6
                new = [torch.cat([p[i] for p in parts]) for i in range(n_field)]
                if model is None:
                    model = LocalGaussianMap(
                        *new[:6],
                        scale_floor=new[6] if aa_sigma > 0 else None).to(device)
                    extent = float(
                        (new[0].max(0).values - new[0].min(0).values).norm() / 2)
                else:
                    with torch.no_grad():
                        merged = [
                            torch.cat([model.positions.detach(), new[0]]),
                            torch.cat([model.log_scales.detach().exp(), new[1]]),
                            # new[1] is the raw scale and new[6] its floor, so
                            # both halves stay separate across the merge --
                            # collapsing them here would bake the filter in.
                            torch.cat([model.quat.detach(), new[2]]),
                            torch.cat([model.rgb().detach(), new[3]]),
                            torch.cat([model.opacity().detach(), new[4]]),
                            torch.cat([model.kf_id, new[5]]),
                        ]
                        floor = (torch.cat([model.scale_floor, new[6]])
                                 if aa_sigma > 0 else None)
                    model = LocalGaussianMap(*merged, scale_floor=floor).to(device)
                opt = _optimizer_for(model, extent, opt,
                                     lr_means=0.0 if freeze_means else 1.6e-4)
                print(f"[refiner] +{new[0].shape[0]:,} gaussians "
                      f"(kf {known_kf}..{n_kf - 1}) -> {model.n:,} total"
                      + (f"  floor {float(model.scale_floor.mean()) * 1e3:.2f} mm "
                         f"median, means "
                         f"{'FROZEN' if freeze_means else 'free'}"
                         if model._has_floor else "  NO FLOOR"),
                      flush=True)
                if dedup_voxel > 0 and model.n > max_gaussians:
                    with keyframes.lock:
                        kf_now = keyframes.T_WC[:n_kf, 0].clone()
                    model, keep_mask = dedup_by_voxel(
                        model, sim3_to_mat(kf_now).to(device), dedup_voxel)
                    opt = _optimizer_subset(model, extent, opt, keep_mask,
                                            lr_means=0.0 if freeze_means else 1.6e-4)
                    print(f"[refiner] dedup @{dedup_voxel} m -> {model.n:,} "
                          f"gaussians", flush=True)
            known_kf = n_kf

        if model is None or opt is None:
            time.sleep(0.01)
            continue

        if (unfreeze_in_polish and freeze_means and polish_flag is not None
                and polish_flag.value and opt is not None):
            for gp in opt.param_groups:
                if gp["name"] == "means":
                    gp["lr"] = 1.6e-4 * extent
            freeze_means = False
            print(f"[refiner] polish: means UNFROZEN (lr {1.6e-4 * extent:.2e})",
                  flush=True)

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
            idx = (model.visible_exact(kf_mats, c2w, K, tgt.shape[-2:])
                   if cull else None)
            means_w, cov_w = model.world(kf_mats, idx)
            pred = render_map(means_w, cov_w, model.rgb(idx), model.opacity(idx),
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

            if (polish_tol > 0 and polish_done is not None
                    and polish_flag is not None and polish_flag.value):
                # Compare WINDOW MEANS, not single samples. `loss` here is one
                # randomly drawn supervision frame, so two samples 200 steps
                # apart differ mostly by which frames were drawn: the first
                # version of this test compared them directly and fired on
                # 360 with rel = -0.61, i.e. it stopped the polish because a
                # noisier frame came up, and called that convergence.
                run_refiner._polish_acc = (
                    getattr(run_refiner, "_polish_acc", 0.0) + float(loss.item()))
                run_refiner._polish_n = getattr(run_refiner, "_polish_n", 0) + 1
                if run_refiner._polish_n >= log_every:
                    cur = run_refiner._polish_acc / run_refiner._polish_n
                    run_refiner._polish_acc, run_refiner._polish_n = 0.0, 0
                    prev = getattr(run_refiner, "_polish_ref", None)
                    if prev is not None and prev > 0:
                        rel = (prev - cur) / prev
                        # A RISING mean is not convergence, but it is equally a
                        # reason to stop -- reported as what it is either way.
                        # PATIENCE. One window mean below tol is not a
                        # plateau: measured on desk, the criterion fired on a
                        # window whose mean had RISEN 1.8% while the map was
                        # still improving (quality at 3526 steps beat quality
                        # at the 1949 where it stopped). N consecutive windows
                        # is the cheapest way to tell a dip from a plateau.
                        run_refiner._polish_hits = (
                            getattr(run_refiner, "_polish_hits", 0) + 1
                            if rel < polish_tol else 0)
                        if run_refiner._polish_hits >= polish_patience:
                            polish_done.value = 1
                            print(f"[refiner] polish stopped: mean loss "
                                  f"{prev:.4f} -> {cur:.4f} ({rel:+.4f} over "
                                  f"{log_every} steps, tol {polish_tol}, "
                                  f"{polish_patience} consecutive)", flush=True)
                    run_refiner._polish_ref = cur

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
        # Save BEFORE publishing: the .ply is the artifact; the snapshot is
        # a viewer convenience and must never be able to block the save.
        if save_path:
            save_refined_map(save_path, model, kf_mats)
        if snapshot is not None:
            snapshot.publish(model, kf_mats)
    print(f"[refiner] terminated after {step} steps", flush=True)
