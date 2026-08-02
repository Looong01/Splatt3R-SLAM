"""LoRA fine-tuning support for MAST3RGaussians.

See the splatt3r-lora-finetuning skill for the full design rationale
(why LoRA is applicable here, what the torch.no_grad() blocker is, and
what was verified vs. assumed).
"""
import einops
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from main import MAST3RGaussians
from utils import geometry, sh_utils


GAUSSIAN_HEAD_MODULES = ["downstream_head1.gaussian_dpt.dpt", "downstream_head2.gaussian_dpt.dpt"]


def attach_lora(model: MAST3RGaussians, r=8, alpha=16, target_modules=("qkv", "proj"), dropout=0.0, resume_from=None):
    """Wrap model.encoder's attention Linear/Conv layers with LoRA adapters,
    in place, and re-enable gradient on the Gaussian head.

    IMPORTANT: peft.get_peft_model() marks ALL base-model parameters as
    non-trainable except the newly-injected LoRA weights -- this clobbers
    the requires_grad_(True) that MAST3RGaussians.__init__ already set on
    downstream_head{1,2}.gaussian_dpt.dpt. Verified empirically (not just
    assumed) during the smoke test in this session: the trainable-param
    count after get_peft_model() matched only the LoRA A/B matrices, not
    the (much larger) Gaussian head. Re-enabling it explicitly below is
    required, not optional.

    GAUSSIAN_HEAD_MODULES is passed as LoraConfig's modules_to_save --
    NOT optional either, and found the hard way (2026-07-26, after this
    whole session's worth of training runs): without it,
    encoder.save_pretrained() only ever wrote the LoRA A/B matrices.
    Verified directly by loading a saved best/adapter_model.safetensors
    and checking its keys: 2,373,632 params, all `lora_`-prefixed, zero
    `gaussian_dpt`/`downstream_head` keys. The Gaussian head IS trainable
    (requires_grad_(True) below still matters) and IS being optimized
    during any single continuous run, but every save/resume across this
    whole debugging saga silently dropped all of its accumulated training
    and restarted the head from the base checkpoint's original weights --
    plausibly a real contributor to why val/psnr kept plateauing despite
    many resumed epochs of "training." modules_to_save tells peft to
    track these specific submodules as fully-trainable-and-saved
    alongside the LoRA weights (wraps them in a
    peft.utils.other.ModulesToSaveWrapper -- transparent for forward(),
    but changes what `base.downstream_head1.gaussian_dpt.dpt` actually
    *is* after this call).

    resume_from: path to a directory saved by a previous
    encoder.save_pretrained() (see SaveLoRAAdapterCallback/
    SaveBestAdapterCallback in scripts/train_lora_per_scene.py). If given,
    loads those adapter weights instead of a fresh LoraConfig init --
    added after the rasterizer's recurring illegal-memory-access crash
    (see the splatt3r-lora-finetuning skill) kept eventually killing long
    runs; r/alpha/target_modules are ignored in this case since the saved
    adapter's own adapter_config.json already encodes them. NOTE: any
    checkpoint saved before this modules_to_save fix has no head weights
    to restore -- resuming from one still only recovers the LoRA half,
    the head silently falls back to its base-checkpoint state (same
    behavior as before this fix, not worse, just not-yet-improved).
    """
    if resume_from is not None:
        model.encoder = PeftModel.from_pretrained(model.encoder, resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            target_modules=list(target_modules),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            modules_to_save=GAUSSIAN_HEAD_MODULES,
        )
        model.encoder = get_peft_model(model.encoder, lora_cfg)

    # NOTE: no manual `base.downstream_headN.gaussian_dpt.dpt
    # .requires_grad_(True)` here any more. Once GAUSSIAN_HEAD_MODULES is
    # passed as modules_to_save, peft already marks the trainable copy of
    # each head as requiring grad, so those calls were redundant -- and
    # actively harmful: ModulesToSaveWrapper holds BOTH a frozen
    # `original_module` and a trainable `modules_to_save[adapter]` copy,
    # and reaching in by attribute path re-enabled grad on both. Measured:
    # that made 40,405,916 extra params trainable which peft never
    # serializes and whose gradients cannot affect the output (the wrapper
    # forwards through the modules_to_save copy while an adapter is
    # active) -- pure optimizer-state and VRAM waste. Trainable total goes
    # 86,986,552 -> 46,580,636 with these removed.
    return model


# Penalize pre-clamp scale overshoot: the hard clamp below is a one-way
# ratchet (torch.clamp's gradient is exactly 0 beyond the bound), so
# Gaussians pinned at max=2.0 could never receive a "shrink" signal --
# measured 2026-07-27: after only 5 epochs at LR 2e-6, p99 scale was 85x
# the base model's (base p99.9 = 0.0214) and 0.349% of Gaussians were
# pinned at 2.0, inflating num_rendered ~1821x and OOMing the rasterizer.
# The penalty is computed on the PRE-clamp scales, so even already-pinned
# Gaussians get a gradient (through exp()) that pulls them back under the
# threshold. relu(s - T)**2 is exactly zero for well-behaved Gaussians:
# no penalty, no gradient, no distortion of the healthy range -- unlike a
# soft saturating clamp (e.g. tanh), which would distort the forward
# mapping everywhere. The hard clamp stays, but purely as a numerical
# guard for the rasterizer, not as the (broken) training signal.
SCALE_PENALTY_THRESHOLD = 0.5  # ~25x the base model's p99.9 (0.0214), 4x below the clamp
SCALE_PENALTY_WEIGHT = 0.1  # logged separately as train/scale_pen; tune from there
# DEFAULT OFF. The penalty above was a patch for a LoRA-specific training
# pathology (runaway scale growth with a LoRA-unfrozen encoder), i.e. a
# confound for any experiment it wasn't designed for. The head-only
# control experiment (scripts/train_lora_per_scene.py HEAD_ONLY=True) must
# NOT have it -- it uses the base MAST3RGaussians class and never runs
# this forward() at all, but this flag additionally guarantees no
# _scale_penalty is ever produced unless a future LoRA-mode run
# explicitly opts back in. Set True only to reproduce the historical LoRA
# runs that used it.
SCALE_PENALTY_ENABLED = False


class MAST3RGaussiansLoRA(MAST3RGaussians):
    """MAST3RGaussians, but forward() runs the encoder/decoder under
    grad-tracking instead of torch.no_grad() -- required for any LoRA
    adapters injected into the encoder to receive gradients at all.
    Everything else (loss calc, logging, optimizer target params) is
    inherited unchanged; the frozen base weights simply never accumulate
    a meaningful gradient since their requires_grad stays False.
    """

    def forward(self, view1, view2):
        base = self.encoder.get_base_model() if hasattr(self.encoder, "get_base_model") else self.encoder

        (shape1, shape2), (feat1, feat2), (pos1, pos2) = base._encode_symmetrized(view1, view2)
        dec1, dec2 = base._decoder(feat1, pos1, feat2, pos2)

        pred1 = base._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = base._downstream_head(2, [tok.float() for tok in dec2], shape2)

        # Sanitize scales/rotations before build_covariance() -- mirrors a
        # fix already applied on the SLAM-inference side
        # (splatt3r_slam/splatt3r_utils.py: bake_gaussians_world(), which
        # normalizes rotations and clamps scales before the same
        # build_covariance() call) but was missing here on the training
        # side entirely. Root cause of a real crash: a run's training_step
        # died with `torch.AcceleratorError: CUDA error: an illegal memory
        # access was encountered` inside the CUDA rasterizer
        # (cuda_splatting.py's tanfovx=tan_fov_x[i].item(), a synchronizing
        # call -- but per PyTorch's own caveat in that error, "CUDA kernel
        # errors might be asynchronously reported at some other API call",
        # meaning the actual corrupting kernel launch happened earlier and
        # unrelated to that specific line). geometry.build_covariance()'s
        # quaternion_to_matrix() only has an eps=1e-8 in its denominator,
        # which prevents an exact division by zero but not a near-zero
        # quaternion norm producing a huge, non-orthonormal "rotation"
        # matrix -- combine that with an unclamped, occasionally-wild
        # early-LoRA-training scale prediction (see the training-collapse
        # analysis in the splatt3r-lora-finetuning skill for other evidence
        # this model's raw predictions aren't always well-behaved yet) and
        # the resulting covariance can be degenerate enough to make the
        # CUDA rasterizer read/write out of bounds internally -- it has no
        # input validation of its own. nan_to_num first (clamp() leaves NaN
        # unchanged since NaN comparisons are always false, so it has to
        # run before the clamp/normalize below, not instead of it).
        # max=0.5, not the earlier max=10.0: found by reading decoder_
        # splatting_cuda.py's render_cuda(..., scale_invariant=True) call --
        # it hardcodes near=0.1 and multiplies gaussian_covariances by
        # scale**2 where scale = 1/near = 10, i.e. covariance (and hence
        # effective screen-space Gaussian size) gets a further 100x
        # multiply on top of whatever's stored in pred["scales"] here. A
        # "generously bounded" max=10.0 pre-multiply is therefore up to
        # ~100x post-multiply in effective standard deviation -- large
        # enough to cover the entire frame and touch every tile, which
        # explodes num_rendered (duplicated Gaussian-tile pairs) in
        # rasterizer_impl.cu regardless of how many input Gaussians there
        # are. This is the likely real reason spatial_stride=2 (4x fewer
        # input Gaussians) still crashed at cuda_rasterizer/
        # rasterizer_impl.cu, just at a different CHECK_CUDA line (326
        # this time, forward-pass tile-range identification, vs. 398
        # backward-pass render before) -- consistent with genuine memory
        # corruption from an oversized-footprint Gaussian, not a simple
        # too-many-Gaussians-in-total problem. 0.5 matched
        # splatt3r_slam/splatt3r_utils.py's own max_scale at the time, but
        # that value made sense specifically to counter the x100 blowup
        # above -- once decoder_splatting_cuda.py's render_cuda() call was
        # changed to scale_invariant=False (removing that x100 entirely),
        # 0.5 became needlessly tight: a run with it survived 16+ epochs
        # without crashing, but psnr never rose above ~9 (vs. ~9-16 in the
        # last healthy pre-crash-mitigation run), i.e. it was trading real
        # reconstruction quality (Gaussians too small to cover/blend
        # properly) for stability scale_invariant=False already provides
        # on its own. Loosened to 2.0 -- still 5x tighter than the
        # original max=10.0 that caused the crash in the first place, not
        # a return to unconstrained. Not yet reverified against a real
        # run for stability -- if the crash comes back, this is the next
        # thing to tighten again before reaching for spatial_stride.
        scale_pen_terms = []
        raw_terms = []
        for pred in (pred1, pred2):
            scales_raw = torch.nan_to_num(
                pred["scales"], nan=1e-6, posinf=10.0, neginf=1e-6
            )
            raw_terms.append(scales_raw)
            if SCALE_PENALTY_ENABLED:
                scale_pen_terms.append(
                    torch.relu(scales_raw - SCALE_PENALTY_THRESHOLD) ** 2
                )
            pred["scales"] = scales_raw.clamp(min=1e-6, max=2.0)
            rotations = torch.nan_to_num(pred["rotations"], nan=0.0, posinf=0.0, neginf=0.0)
            pred["rotations"] = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # Stash for MAST3RGaussiansLoRA.training_step to pick up (see
        # SCALE_PENALTY_THRESHOLD above). Detached stats for logging only.
        if SCALE_PENALTY_ENABLED:
            pen = torch.cat([t.flatten() for t in scale_pen_terms]).mean()
            self._scale_penalty = pen
        all_raw = torch.cat([t.flatten() for t in raw_terms]).detach().float()
        self._scale_stats = (all_raw.mean().item(), torch.quantile(all_raw, 0.99).item())

        pred1["covariances"] = geometry.build_covariance(pred1["scales"], pred1["rotations"])
        pred2["covariances"] = geometry.build_covariance(pred2["scales"], pred2["rotations"])

        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1["sh"])
            new_sh2 = torch.zeros_like(pred2["sh"])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1["original_img"], "b c h w -> b h w c"))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2["original_img"], "b c h w -> b h w c"))
            pred1["sh"] = pred1["sh"] + new_sh1
            pred2["sh"] = pred2["sh"] + new_sh2

        # Sanitizing scales/rotations/covariances above (and this) still
        # wasn't the whole picture: a later real run got much further
        # (crashed at epoch 2/step ~5640 instead of epoch 0/step ~2190 --
        # same exact `tanfovx=tan_fov_x[i].item()` illegal-memory-access
        # signature) but still crashed. means/opacities/sh flow into the
        # same CUDA rasterizer completely unguarded -- only
        # scales/rotations had been covered. A NaN/huge predicted 3D
        # position is just as capable of producing a garbage tile index
        # inside the rasterizer as a degenerate covariance was. Covering
        # everything that reaches render_cuda() now, not just covariance's
        # inputs. Bounds are generous, meant to catch genuine blow-ups
        # (NaN/Inf/astronomical values) without constraining normal
        # training range -- opacities are the one case with a real,
        # known-correct bound ([0, 1], standard alpha).
        for pred in (pred1, pred2):
            pred["means"] = torch.nan_to_num(pred["means"], nan=0.0, posinf=1e3, neginf=-1e3).clamp(min=-1e3, max=1e3)
            pred["opacities"] = torch.nan_to_num(pred["opacities"], nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
            pred["sh"] = torch.nan_to_num(pred["sh"], nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)
            pred["covariances"] = torch.nan_to_num(pred["covariances"], nan=0.0, posinf=100.0, neginf=-100.0)

        pred2["pts3d_in_other_view"] = pred2.pop("pts3d")
        pred2["means_in_other_view"] = pred2.pop("means")

        return pred1, pred2

    def training_step(self, batch, batch_idx):
        # super() runs forward() (which stashes _scale_penalty/_scale_stats)
        # and logs train/loss|mse|psnr|lpips WITHOUT the penalty, keeping
        # those metrics comparable with pre-penalty runs and with val/*.
        # The penalty is added only to the tensor Lightning backprops.
        loss = super().training_step(batch, batch_idx)
        pen = getattr(self, "_scale_penalty", None)
        stats = getattr(self, "_scale_stats", None)
        self._scale_penalty = None
        self._scale_stats = None
        if pen is None:
            return loss
        bs = batch["context"][0]["img"].shape[0]
        self.log("train/scale_pen", pen.detach(), batch_size=bs)
        if stats is not None:
            self.log("train/scale_mean", stats[0], batch_size=bs)
            self.log("train/scale_p99", stats[1], batch_size=bs)
        return loss + SCALE_PENALTY_WEIGHT * pen

    def configure_optimizers(self):
        # AdamW (decoupled weight decay), not plain Adam -- matches upstream
        # configs/main.yaml's weight_decay: 0.05, which the earlier plain
        # Adam call was silently missing entirely (no weight_decay passed
        # -> 0). Note this does NOT by itself explain or fix a training
        # collapse: Adam-family optimizers are already per-parameter
        # adaptive regardless of the Adam/AdamW choice, and that adaptivity
        # doesn't cap the damage from a single outlier gradient spike (its
        # moment estimates are a running average, not a hard bound). What
        # actually guards against that is LR magnitude and
        # GRADIENT_CLIP_VAL (see scripts/train_lora_per_scene.py) -- both
        # were tightened alongside this change after a real run collapsed:
        # loss was stable at ~0.1-0.2 for all of epoch 0, then spiked
        # 0.59->1.28 within one 10-step logging interval partway through
        # epoch 1 and never recovered for the remaining 86 epochs (see the
        # splatt3r-lora-finetuning skill for the full metrics.csv analysis).
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.config.opt.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [max(1, self.config.opt.epochs // 2)], gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
