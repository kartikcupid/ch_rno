#!/bin/bash
#SBATCH --job-name=rno_predict
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=rp_%j.out
#SBATCH --error=rp_%j.err

# Predict and draw with an already trained RNO: Work/models/rno_t<END>_lap/
# (VARIANT=lap, the default) or Work/models/rno_t<END>/ (VARIANT=base);
# WIDTH / MODES / LAYERS / SEED select a tagged model as in rt.sh.
#
#   sbatch --export=ALL,END=1000 rp.sh                  predict to 10000
#   sbatch --export=ALL,END=1000,WIDTH=24,MODES=12 rp.sh
#   t=$(sbatch --parsable --export=ALL,END=1000 rt.sh)
#   sbatch --dependency=afterok:$t --export=ALL,END=1000 rp.sh
#
# Resumable: seeds already in Work/rollout_cache/<label>/ for the same
# checkpoint and schedule are skipped.

set -euo pipefail

END=${END:-1000}
VARIANT=${VARIANT:-lap}
T_END=${T_END:-10000}
MODES=${MODES:-16}
WIDTH=${WIDTH:-32}
LAYERS=${LAYERS:-4}
SEED=${SEED:-0}

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "======================= PREDICTION (END=$END, VARIANT=$VARIANT, WIDTH=$WIDTH, MODES=$MODES, LAYERS=$LAYERS, SEED=$SEED, T_END=$T_END) ======================="
python rno_predict.py --end "$END" --variant "$VARIANT" --modes "$MODES" \
    --width "$WIDTH" --layers "$LAYERS" --seed "$SEED" --t-end "$T_END"

echo
echo "======================= RESULTS ======================="
python make_submission.py --overview

echo
echo "Job finished at $(date)"
