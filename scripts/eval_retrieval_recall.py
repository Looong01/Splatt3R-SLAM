"""Offline retrieval-recall evaluation: (a) existing MASt3R retrieval assets vs
(c) assets re-fitted on dumped Splatt3R backbone features.

Answers the decision question: does re-fitting the retrieval head (PCA whitening
+ ASMK codebook) on Splatt3R features improve retrieval precision over the
existing MASt3R assets? This is the cheap "(a) vs (c)" check before committing
to full stage 1.

Configurations evaluated per sequence (query = each keyframe, database = the
other keyframes of the same sequence):

  (a)  old whitening + old 64k codebook          (existing MASt3R assets)
  (c)  new whitening + new codebook, size 2048   (re-fitted on Splatt3R feats)
  (c)  new whitening + new codebook, size 8192
  (+)  no-codebook ablation: weighted-spoc global descriptor + brute-force
       cosine, once with old whitening and once with new whitening. This
       decouples "whitening quality" from "codebook quality".

Descriptor pipeline replicates splatt3r_slam/retrieval_database.py:25-41
(prep_features) and mast3r/retrieval/model.py:
  feat (768,1024) -> prewhiten (subtract m, multiply p, float64 -> float32)
  -> attention = per-row l2 norm -> top-300 rows by attention -> local
  descriptors (300,1024). Global descriptor = weighted spoc
  (sum(feat*attn), l2-normalized), cf. model.py:79-85 weighted_spoc.

NOTE on an approximation: splatt3r_slam's RetrievalDatabase quantizes with a
custom torch L2 top-k; here we use asmk's own quantize (faiss CPU
nearest-centroid search). Both are exact nearest-centroid assignment, so this
is equivalent up to tie-breaking.

Positives for a query: GT translation distance < 0.5 m AND |kf_idx gap| > 5
(excludes trivial temporally-adjacent hits). Queries with no positive in the
database are counted and excluded from the recall denominator.

All computation is CPU (DDP training occupies both GPUs); asmk is initialized
with index.gpu_id=None -> FaissL2Index (CPU kmeans/index).

Usage:
    cd /home/share-v5/Codes/Splatt3R-SLAM
    python3 scripts/eval_retrieval_recall.py
"""
import json
import os
import sys
import time

import numpy as np

# sys.path setup mirrors scripts/eval_lora_scenes.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))

import torch  # noqa: E402  (needed to load the retrieval .pth)
from mast3r.retrieval.model import pcawhitenlearn_shrinkage  # noqa: E402
from asmk import asmk_method  # noqa: E402

FEAT_ROOT = os.path.join(REPO_ROOT, "logs", "retrieval_features")
TUM_ROOT = os.path.join(REPO_ROOT, "datasets", "tum")
RETRIEVAL_PTH = os.path.join(
    REPO_ROOT, "checkpoints",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
OLD_CODEBOOK_PKL = os.path.join(
    REPO_ROOT, "checkpoints",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl")
OUT_DIR = os.path.join(REPO_ROOT, "logs", "retrieval_recall")

SEQUENCES = ["rgbd_dataset_freiburg1_room", "rgbd_dataset_freiburg1_360",
             "rgbd_dataset_freiburg1_desk"]
NFEAT = 300          # ckpt['args'].nfeat
POS_DIST_M = 0.5     # GT translation threshold for a positive
MIN_KF_GAP = 5       # exclude temporally adjacent hits
GT_TOL_S = 0.02      # timestamp matching tolerance
NEW_CB_SIZES = [2048, 8192]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_sequence(seq):
    """Return (feats [N,768,1024] float32, kf_idxs [N], timestamps [N])."""
    sdir = os.path.join(FEAT_ROOT, seq)
    metas = [json.loads(l) for l in open(os.path.join(sdir, "metadata.jsonl"))]
    metas.sort(key=lambda m: m["kf_idx"])
    feats = np.stack([np.load(os.path.join(sdir, f"feat_{m['kf_idx']:06d}.npy"))
                      for m in metas])
    return (feats, np.array([m["kf_idx"] for m in metas]),
            np.array([m["timestamp"] for m in metas]))


def load_gt_poses(seq, timestamps):
    """Match each keyframe timestamp to the nearest TUM groundtruth position."""
    gt = np.loadtxt(os.path.join(TUM_ROOT, seq, "groundtruth.txt"),
                    comments="#")
    gt_t, gt_xyz = gt[:, 0], gt[:, 1:4]
    pos = np.empty((len(timestamps), 3))
    for i, ts in enumerate(timestamps):
        j = np.argmin(np.abs(gt_t - ts))
        assert abs(gt_t[j] - ts) <= GT_TOL_S, (seq, i, ts, gt_t[j])
        pos[i] = gt_xyz[j]
    return pos


def compute_positives(pos, kf_idxs):
    """positives[q] = set of db indices that are true positives for query q."""
    n = len(pos)
    positives = {}
    for q in range(n):
        dist = np.linalg.norm(pos - pos[q], axis=1)
        gap = np.abs(kf_idxs - kf_idxs[q])
        positives[q] = set(np.where((dist < POS_DIST_M) & (gap > MIN_KF_GAP))[0])
        positives[q].discard(q)
    return positives


# ---------------------------------------------------------------------------
# descriptor extraction (mirrors RetrievalDatabase.prep_features / model.py)
# ---------------------------------------------------------------------------

def prep_descriptors(feats, m, p):
    """feats (N,768,1024) + whitening (m, p, float64) ->
    (local descriptors [N,300,1024] float32, global descriptors [N,1024])."""
    x = (feats.astype(np.float64) - m) @ p          # prewhiten
    x = x.astype(np.float32)
    attn = np.linalg.norm(x, axis=-1)               # (N,768), featweights='l2norm'
    top = np.argsort(-attn, axis=1)[:, :NFEAT]      # how_select_local, nfeat=300
    locals_ = np.take_along_axis(x, top[:, :, None], axis=1)
    # weighted spoc: sum(feat*attn) then l2-normalize (model.py:79-85)
    g = (x * attn[:, :, None]).sum(axis=1)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return locals_, g


# ---------------------------------------------------------------------------
# ASMK helpers (params copied from mast3r/retrieval/processor.py:91-96,
# with index.gpu_id=None -> CPU FaissL2Index)
# ---------------------------------------------------------------------------

def make_asmk_params(codebook_size):
    return {
        'index': {'gpu_id': None},
        'train_codebook': {'codebook': {'size': codebook_size}},
        'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                      'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
        'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                      'search': {'topk': None},
                      'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}


def load_old_asmk():
    asmk = asmk_method.ASMKMethod.initialize_untrained(make_asmk_params('64k'))
    return asmk.train_codebook(None, cache_path=OLD_CODEBOOK_PKL)


def train_new_asmk(train_descs, size):
    """Train a fresh codebook on the given descriptors (faiss CPU kmeans,
    asmk default niter=10), cached under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = os.path.join(OUT_DIR, f"codebook_new_{size}.pkl")
    asmk = asmk_method.ASMKMethod.initialize_untrained(make_asmk_params(size))
    t0 = time.time()
    asmk = asmk.train_codebook(train_descs, cache_path=cache)
    meta = asmk.metadata['train_codebook']
    if 'load_time' in meta:
        print(f"  codebook {size}: loaded from cache")
    else:
        print(f"  codebook {size}: trained in {meta['train_time']:.1f}s "
              f"(cluster {meta['cluster_time']:.1f}s)")
    print(f"  (total wall {time.time()-t0:.1f}s)")
    return asmk


def asmk_ranked_lists(asmk, locals_):
    """Build an IVF from all frames' descriptors, query with the same set,
    and return per-query ranked db-image lists with the self-hit removed."""
    n = locals_.shape[0]
    vecs = locals_.reshape(n * locals_.shape[1], -1)
    ids = np.repeat(np.arange(n, dtype=np.int64), locals_.shape[1])
    dataset = asmk.build_ivf(vecs, ids)
    _meta, _qids, ranks, _scores = dataset.query_ivf(vecs, ids)
    ranked = []
    for q in range(n):
        ranked.append([int(r) for r in ranks[q] if r != q])
    return ranked


def global_ranked_lists(globals_):
    """Brute-force cosine similarity ranking, self removed."""
    sim = globals_ @ globals_.T
    n = sim.shape[0]
    ranked = []
    for q in range(n):
        order = np.argsort(-sim[q])
        ranked.append([int(r) for r in order if r != q])
    return ranked


def recall_at(ranked, positives, ks=(1, 5)):
    valid = [q for q in range(len(ranked)) if len(positives[q]) > 0]
    out = {}
    for k in ks:
        out[k] = (np.mean([any(r in positives[q] for r in ranked[q][:k])
                           for q in valid]) if valid else float('nan'))
    return out, len(valid), len(ranked) - len(valid)


# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # --- load data ---
    data = {}
    for seq in SEQUENCES:
        feats, kf_idxs, ts = load_sequence(seq)
        pos = load_gt_poses(seq, ts)
        data[seq] = dict(feats=feats, kf_idxs=kf_idxs, pos=pos,
                         positives=compute_positives(pos, kf_idxs))
        print(f"{seq}: {len(feats)} keyframes loaded")

    # --- whitening ---
    ckpt = torch.load(RETRIEVAL_PTH, 'cpu', weights_only=False)
    m_old = ckpt['model']['prewhiten.m'].numpy()
    p_old = ckpt['model']['prewhiten.p'].numpy()

    all_feats = np.concatenate([data[s]['feats'] for s in SEQUENCES],
                               axis=0).reshape(-1, 1024)
    print(f"Fitting new whitening on {all_feats.shape[0]} descriptors...")
    t0 = time.time()
    m_new, p_new = pcawhitenlearn_shrinkage(all_feats)
    print(f"  whitening fit in {time.time()-t0:.1f}s")
    del all_feats

    # --- descriptors per whitening ---
    desc = {}
    for name, (m, p) in {'old': (m_old, p_old), 'new': (m_new, p_new)}.items():
        desc[name] = {}
        for seq in SEQUENCES:
            loc, glob = prep_descriptors(data[seq]['feats'], m, p)
            desc[name][seq] = dict(locals=loc, global_=glob)

    # --- codebooks ---
    print("Loading old 64k codebook...")
    asmk_old = load_old_asmk()
    # new codebooks trained on new-whitened top-300 descriptors of ALL frames
    # (same input distribution as used at query/build time)
    train_descs = np.concatenate([desc['new'][s]['locals'] for s in SEQUENCES],
                                 axis=0).reshape(-1, 1024)
    print(f"Training new codebooks on {train_descs.shape[0]} descriptors...")
    asmk_new = {sz: train_new_asmk(train_descs, sz) for sz in NEW_CB_SIZES}

    configs = (['a_old_white_old_cb64k'] +
               [f'c_new_white_new_cb{sz}' for sz in NEW_CB_SIZES] +
               ['global_old_white', 'global_new_white'])

    # --- evaluate ---
    results = {}
    for seq in SEQUENCES:
        positives = data[seq]['positives']
        n = len(positives)
        n_no_pos = sum(1 for q in range(n) if len(positives[q]) == 0)
        results[seq] = {'n_keyframes': n, 'n_no_positive': n_no_pos}
        for cfg in configs:
            t0 = time.time()
            if cfg.startswith('a_'):
                ranked = asmk_ranked_lists(asmk_old, desc['old'][seq]['locals'])
            elif cfg.startswith('c_'):
                sz = int(cfg.rsplit('cb', 1)[1])
                ranked = asmk_ranked_lists(asmk_new[sz],
                                           desc['new'][seq]['locals'])
            elif cfg == 'global_old_white':
                ranked = global_ranked_lists(desc['old'][seq]['global_'])
            else:
                ranked = global_ranked_lists(desc['new'][seq]['global_'])
            rec, n_valid, _ = recall_at(ranked, positives)
            results[seq][cfg] = {'recall@1': rec[1], 'recall@5': rec[5],
                                 'n_valid_queries': n_valid,
                                 'eval_time_s': round(time.time() - t0, 2)}
            print(f"  {seq} / {cfg}: R@1={rec[1]:.3f} R@5={rec[5]:.3f} "
                  f"({time.time()-t0:.1f}s)")

    # --- report ---
    print("\n=== Recall table (queries with no positive excluded; "
          f"positive = <{POS_DIST_M}m & kf gap >{MIN_KF_GAP}) ===")
    header = f"{'sequence':<32}" + "".join(f"{c:>26}" for c in configs)
    print(header)
    for seq in SEQUENCES:
        r = results[seq]
        print(f"{seq} (kf={r['n_keyframes']}, no-pos={r['n_no_positive']})")
        row = f"{'  R@1 / R@5':<32}"
        for cfg in configs:
            v = r[cfg]
            row += f"{'%0.3f / %0.3f' % (v['recall@1'], v['recall@5']):>26}"
        print(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, 'w') as f:
        json.dump({'config': {'pos_dist_m': POS_DIST_M, 'min_kf_gap': MIN_KF_GAP,
                              'nfeat': NFEAT, 'new_codebook_sizes': NEW_CB_SIZES},
                   'results': results}, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Total wall time: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
