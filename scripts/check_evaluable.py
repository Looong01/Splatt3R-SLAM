"""Is this sequence SCORABLE before we spend two SLAM runs on it?

Written after burning four SLAM runs -- 7-scenes/fire and euroc/V2_01_easy --
on sequences that cannot be evaluated at all, the second pair *after* recording
the lesson from the first pair in the skill file. A rule in a document did not
stop it; a script that fails fast will.

`eval_map_quality.py` needs `<sequence>/groundtruth.txt`. Several releases ship
sequences without it:

    7-scenes   only chess, office, pumpkin have it (4 of 7 do not)
    euroc      only MH_01_easy, V1_01_easy have it (9 of 11 do not)
    eth3d      widely present
    TUM        present
    Replica    generated from traj.txt on first load by the dataloader

Run this before queueing a deployment A/B. Exits non-zero if ANY named
sequence is unscorable, so it can gate a batch script:

    python3 scripts/check_evaluable.py datasets/euroc/V2_01_easy ... || exit 1
    python3 scripts/check_evaluable.py --family euroc      # list what IS usable
"""
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAMILY_GLOB = {
    "tum": "datasets/tum/rgbd_dataset_freiburg1_*",
    "7-scenes": "datasets/7-scenes/*",
    "euroc": "datasets/euroc/*",
    "eth3d": "datasets/eth3d/train/*",
    "replica": "datasets/Replica/office*",
}


def evaluable(path):
    """Replica writes groundtruth.txt from traj.txt on first load, so a
    sequence with traj.txt counts as scorable even before that has happened."""
    if os.path.exists(os.path.join(path, "groundtruth.txt")):
        return True, "groundtruth.txt"
    if os.path.exists(os.path.join(path, "traj.txt")):
        return True, "traj.txt (Replica; dataloader converts on first load)"
    return False, "NO groundtruth.txt and NO traj.txt"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    if args[0] == "--family":
        fams = args[1:] or list(FAMILY_GLOB)
        for fam in fams:
            seqs = sorted(d for d in glob.glob(os.path.join(REPO, FAMILY_GLOB[fam]))
                          if os.path.isdir(d))
            ok = [os.path.basename(d) for d in seqs if evaluable(d)[0]]
            bad = [os.path.basename(d) for d in seqs if not evaluable(d)[0]]
            print(f"{fam:10s} {len(ok)}/{len(seqs)} scorable")
            print(f"{'':10s}   usable : {', '.join(ok) if ok else '(none)'}")
            if bad:
                print(f"{'':10s}   SKIP   : {', '.join(bad)}")
        return 0

    bad = 0
    for p in args:
        full = p if os.path.isabs(p) else os.path.join(REPO, p)
        if not os.path.isdir(full):
            print(f"MISSING    {p}")
            bad += 1
            continue
        ok, why = evaluable(full)
        print(f"{'ok        ' if ok else 'UNSCORABLE'} {p}   ({why})")
        bad += (not ok)
    if bad:
        print(f"\n{bad} sequence(s) cannot be scored by eval_map_quality.py. "
              f"Two SLAM runs each would be wasted.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
