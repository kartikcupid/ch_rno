#!/bin/bash
# One-time setup of this project on the IIT Jodhpur HPC, run on the login node
# from the project folder (no Slurm, no GPU, a few seconds):
#
#   cd /scratch/data/m25ph1006/ch_rno && bash hpc_setup.sh
#   UNO_DIR=/path/to/uno/copy bash hpc_setup.sh       another UNO copy
#
# It links, never copies, what the UNO project already has, so nothing large
# is duplicated (home was at 184 of 200 GB on 2026-10-15):
#
#   env/                    -> $UNO_DIR/env  (the venv the UNO jobs ran in:
#                              torch 2.13 + numpy + matplotlib is all this
#                              project needs; skipped when env/ exists)
#   Work/gt_cache/          -> $UNO_DIR/Work/gt_cache  (~112 GB of ground truth
#                              to t = 10000, same file names and format; a
#                              file it lacks would be solved into that folder)
#   Work/data/test_t<END>.npz and .json -> the UNO datasets that still exist
#   Work/data/memmap_t<END>/{train,val}_X_f32.npy -> the UNO float32 memmaps
#                              of the first Work/models/train_t<END>*/cache/
#                              that has them (the dataset .npz can be gone)
#
# Existing files or links are left alone.  Everything this project writes
# (models, rollout caches, Results/) stays in its own folder.

set -euo pipefail

UNO_DIR=${UNO_DIR:-/iitjhome/m25ph1006/chc_uno_v1}
cd "$(dirname "$0")"

if [ ! -d "$UNO_DIR/Work" ]; then
    echo "UNO project not found at $UNO_DIR (no Work/ there); set UNO_DIR" >&2
    exit 1
fi

link() {
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "  keep   $2"
    else
        ln -s "$1" "$2"
        echo "  link   $2 -> $1"
    fi
}

echo "UNO project: $UNO_DIR"
mkdir -p Work/data

if [ -d "$UNO_DIR/env" ]; then
    link "$UNO_DIR/env" env
else
    echo "  no $UNO_DIR/env: create a venv here (README, 'HPC')"
fi

link "$UNO_DIR/Work/gt_cache" Work/gt_cache

for f in "$UNO_DIR"/Work/data/test_t*.npz "$UNO_DIR"/Work/data/test_t*.json; do
    [ -e "$f" ] || continue
    case "$f" in *.partial.*) continue ;; esac
    link "$f" "Work/data/$(basename "$f")"
done

for END in 0300 0500 1000 2000; do
    for d in "$UNO_DIR"/Work/models/train_t${END}_lap/cache \
             "$UNO_DIR"/Work/models/train_t${END}/cache \
             "$UNO_DIR"/Work/models/train_t${END}_*/cache; do
        if [ -e "$d/train_X_f32.npy" ] && [ -e "$d/val_X_f32.npy" ]; then
            mkdir -p "Work/data/memmap_t$END"
            link "$d/train_X_f32.npy" "Work/data/memmap_t$END/train_X_f32.npy"
            link "$d/val_X_f32.npy" "Work/data/memmap_t$END/val_X_f32.npy"
            break
        fi
    done
done

echo
echo "Python in env/: $(env/bin/python --version 2>&1 || echo 'missing')"
echo "Check with:   source env/bin/activate && python check_code.py"
echo "Then e.g.:    sbatch --export=ALL,END=1000 rtp.sh"
