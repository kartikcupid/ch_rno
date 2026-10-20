#!/bin/bash
#SBATCH --job-name=ch_gpu
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=ch_gpu_%j.out
#SBATCH --error=ch_gpu_%j.err

# Training dataset Work/data/test_t<END>.npz (ch.py's data, frame for frame).
# Not needed when hpc_setup.sh has linked the UNO project's datasets or
# memmaps.  Exits 1 when the file already exists.
#
#   sbatch --export=ALL,END=1000 ch_gpu.sh

set -euo pipefail

END=${END:-1000}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version

echo "======================= DATASET (END=$END) ======================="
python ch_gpu.py --end "$END"

echo
echo "Job finished at $(date)"
