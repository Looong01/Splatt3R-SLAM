"""
Runtime dump of per-keyframe retrieval features (frame.feat).

frame.feat is the raw Splatt3R encoder feature that the retrieval head
consumes (retrieval_database.py -> prep_features -> prewhiten/projector/
attention). It is attached to every frame before that frame can enter
SharedKeyframes:

- INIT frames: main.py calls splatt3r_utils.splatt3r_inference_mono(),
  which encodes the image via model.encoder._encode_image() and stores the
  result on frame.feat (splatt3r_utils.py:591-594) before
  keyframes.append(frame).
- TRACKING keyframes: tracker.track() -> splatt3r_match_asymmetric() ->
  splatt3r_asymmetric_inference() performs the same encode
  (splatt3r_utils.py:667-669) before main.py appends the frame.

So frame.feat is always non-None at both keyframes.append() sites in
main.py (SharedKeyframes.__setitem__ at frame.py:358 would raise
otherwise). The None check in dump() is only a defensive guard.
"""

import json
import pathlib

import numpy as np
import torch


class RetrievalFeatureDumper:
    """Writes feat_<kf_idx>.npy + metadata.jsonl under <root>/<seq_name>/."""

    def __init__(self, root, seq_name):
        self.dir = pathlib.Path(root) / seq_name
        if self.dir.exists() and any(self.dir.iterdir()):
            # Match the project's rerun semantics (main.py unlinks stale
            # trajectory files before a run): warn and overwrite instead of
            # mixing new features with a previous run's files.
            print(
                f"[retrieval-dump] WARNING: {self.dir} already exists and is "
                f"not empty; overwriting its contents"
            )
            for stale in self.dir.glob("feat_*.npy"):
                stale.unlink()
            meta = self.dir / "metadata.jsonl"
            if meta.exists():
                meta.unlink()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "metadata.jsonl"
        print(f"[retrieval-dump] dumping keyframe retrieval features to {self.dir}")

    def dump(self, kf_idx, frame, timestamp):
        feat = frame.feat
        if feat is None:
            print(
                f"[retrieval-dump] WARNING: kf {kf_idx} (frame {frame.frame_id}) "
                f"has feat=None; skipped"
            )
            return
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat)
        # model.encoder._encode_image returns (1, N, 1024); drop the leading
        # batch dim and record the true stored shape in the metadata.
        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat[0]
        np.save(self.dir / f"feat_{kf_idx:06d}.npy", feat)
        record = {
            "kf_idx": int(kf_idx),
            "frame_id": int(frame.frame_id),
            "timestamp": float(timestamp),
            "img_shape": [int(v) for v in frame.img_shape.flatten().tolist()],
            "feat_shape": list(feat.shape),
        }
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(record) + "\n")
