#!/bin/bash
# Train all four LoRA families, one fresh `python3` process per family.
#
# Why this exists instead of just `python3 scripts/train_lora_per_scene.py`
# (which also loops over all 4 families internally, in ONE process): under
# multi-GPU DDP (DEVICES=2 in that script), each `L.Trainer(...).fit()` call
# initializes and tears down a torch.distributed process group. Lightning
# is *supposed* to support creating a fresh Trainer and calling .fit() again
# afterward in the same process (a fairly common pattern, e.g. k-fold CV
# loops), but this was never actually exercised end-to-end in this repo
# across 4 sequential DDP fit() calls -- not run, per the user's explicit
# instruction this round (see the splatt3r-lora-finetuning skill). Giving
# each family a genuinely fresh process sidesteps that question entirely:
# every family gets a clean process-group init from scratch, at the cost of
# reloading the base checkpoint once per family instead of reusing an
# already-loaded one (a few seconds, irrelevant next to hours of training).
#
# Usage:
#   cd /home/share-v5/Codes/Splatt3R-SLAM
#   bash scripts/train_lora_all_families.sh                  # all 4
#   bash scripts/train_lora_all_families.sh tum 7-scenes      # just these
#
# A family that crashes (not just errors immediately, but exits nonzero
# after real training progress) is retried up to MAX_RETRIES times before
# this script gives up on it -- added after a rasterizer illegal-memory-
# access crash (see the splatt3r-lora-finetuning skill) kept eventually
# killing long runs, even after several fixes pushed the crash further
# out each time (training now reliably survives multiple full epochs
# first). scripts/train_lora_per_scene.py's find_resume_checkpoint()
# means a retry warm-starts from the last saved adapter instead of
# starting over from scratch -- NOT a true Lightning resume (optimizer/
# scheduler/epoch-count state is not preserved, only the LoRA weights
# themselves), but far better than losing everything on every crash.
# Still fails loudly (propagates a nonzero exit) if a family exhausts its
# retries, rather than silently moving on -- a multi-hour/day unattended
# run should surface a real, unresolved problem, not hide it.
set -e
cd "$(dirname "$0")/.."

MAX_RETRIES=5
# Seconds to wait before each retry -- both a crash-loop brake (a family
# that fails near-instantly, e.g. from a bad config rather than a rare
# runtime crash, would otherwise burn through MAX_RETRIES in seconds) and
# time for orphaned processes (see cleanup_orphans below) to actually
# finish exiting and release GPU memory before the next attempt starts.
RETRY_DELAY=30

# When rank0 of a DDP pair dies (e.g. the illegal-memory-access crash --
# see the splatt3r-lora-finetuning skill), rank1 does NOT reliably die
# with it: observed in practice getting stuck spamming NCCL "Broken pipe"
# messages indefinitely, as an orphan, while this script had already moved
# on to a fresh retry attempt. Left unchecked, each retry adds another
# stuck rank1 (and its DataLoader workers) without ever cleaning up the
# previous one -- caught in practice after several retries piled up ~100
# stray processes and both GPUs pinned at 100% from orphans, not real
# training.
#
# Each attempt is launched under `setsid` so it becomes the leader of its
# OWN process group (pgid == its pid), which makes `kill -- -$pid` reap
# exactly that attempt's whole tree (rank0, rank1, every DataLoader
# worker) and nothing else.
#
# The previous implementation instead did a global
# `pgrep -f train_lora_per_scene.py | xargs kill -KILL`, which was a real
# bug, not just imprecision: it kills matching processes belonging to ANY
# concurrently-running instance of this script. Hit in practice -- an
# earlier wrapper survived as an orphan (ppid=1) after its parent shell
# was killed and kept retrying for hours; once a second wrapper was
# started, the two fought over both GPUs (causing OOM, since each job
# alone needs ~23GiB of 48GiB) while each one's cleanup killed the
# other's freshly-spawned ranks, an endless mutual-destruction loop.
cleanup_attempt() {
  local pid="$1"
  [ -n "$pid" ] && kill -KILL -- -"$pid" 2>/dev/null || true
}

# `setsid` above is what makes per-attempt cleanup precise, but it also
# detaches the training tree from this terminal's foreground process
# group -- so a Ctrl-C reaches only this wrapper, and without this trap
# the wrapper would die while the whole DDP tree kept running as an
# orphan, holding both GPUs. That is the exact failure this script's
# cleanup was written to prevent, just arrived at from the other
# direction. Exit status follows the 128+signal convention: 130 for
# SIGINT, 143 for SIGTERM.
attempt_pid=""
on_signal() {
  local sig_name="$1" exit_code="$2"
  echo ""
  echo "Interrupted (SIG${sig_name}) -- terminating current training attempt..."
  cleanup_attempt "$attempt_pid"
  exit "$exit_code"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

FAMILIES=("$@")
if [ ${#FAMILIES[@]} -eq 0 ]; then
  FAMILIES=(tum 7-scenes euroc eth3d)
fi

for family in "${FAMILIES[@]}"; do
  echo ""
  echo "======================================================================"
  echo "Training family: $family  ($(date))"
  echo "======================================================================"
  attempt=1
  while true; do
    setsid python3 scripts/train_lora_per_scene.py "$family" &
    attempt_pid=$!
    # `set -e` must not abort the script on a failed attempt -- we want to
    # observe the status and decide whether to retry.
    status=0
    wait "$attempt_pid" || status=$?
    if [ "$status" -eq 0 ]; then
      break
    fi
    cleanup_attempt "$attempt_pid"
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      echo "Family $family failed $MAX_RETRIES times, giving up."
      exit 1
    fi
    echo "Family $family crashed (attempt $attempt/$MAX_RETRIES), cleaning up and retrying from last checkpoint in ${RETRY_DELAY}s ($(date))..."
    sleep "$RETRY_DELAY"
    attempt=$((attempt + 1))
  done
done

echo ""
echo "All families done: ${FAMILIES[*]}"
