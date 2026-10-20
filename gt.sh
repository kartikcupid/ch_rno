#!/bin/bash
#SBATCH --job-name=rno_gt
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=gt_%j.out
#SBATCH --error=gt_%j.err

# Ground truth only: every file a prediction to T_END needs, into
# Work/gt_cache/ (test + tune seeds for all five quenches at l = 128, all
# box-size seeds at l = 160/192/256), solved in float64 on the GPU.
#
# With Work/gt_cache linked to the UNO project's cache (hpc_setup.sh) every
# file already exists and this job only confirms it ("0 solved"); it is
# needed only for a fresh cache.
#
#   sbatch gt.sh                                T_END = 10000
#   g=$(sbatch --parsable gt.sh)
#   sbatch --dependency=afterok:$g --export=ALL,END=1000 rtp.sh

set -euo pipefail

T_END=${T_END:-10000}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= GROUND TRUTH (t_end=$T_END) ======================="
python rno_predict.py --gt-only --t-end "$T_END"

echo
echo "Job finished at $(date)"
