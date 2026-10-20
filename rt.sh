#!/bin/bash
#SBATCH --job-name=rno_train
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=rt_%j.out
#SBATCH --error=rt_%j.err

# Train one RNO with the survival-based checkpoint selection.
#
#   sbatch --export=ALL,END=1000 rt.sh                      Laplacian output (lap)
#   sbatch --export=ALL,END=1000,VARIANT=base rt.sh         direct-increment head
#   sbatch --export=ALL,END=1000,WIDTH=24,MODES=12 rt.sh    a smaller model
#   sbatch --export=ALL,END=1000,SEED=1 rt.sh               a seed replicate
#   BENCH=1: sbatch --export=ALL,BENCH=1 rt.sh              time fp32/tf32/fp64 only
#
# VARIANT=lap (default) writes Work/models/rno_t<END>_lap/; VARIANT=base
# writes Work/models/rno_t<END>/.  WIDTH (32), MODES (16), LAYERS (4) and
# SEED (0) at non-default values add a tag, e.g. rno_t1000_lap_w24m12d3s1/.
set -euo pipefail

END=${END:-1000}
VARIANT=${VARIANT:-lap}
MODES=${MODES:-16}
WIDTH=${WIDTH:-32}
LAYERS=${LAYERS:-4}
SEED=${SEED:-0}
ARGS=(--end "$END" --variant "$VARIANT" --modes "$MODES" --width "$WIDTH"
      --layers "$LAYERS" --seed "$SEED")

echo "Job started on $(hostname) at $(date)"

module load python
source env/bin/activate

echo "Python Path: $(which python)"
python --version
echo "CUDA Available Devices: ${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch GPU Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

if [ "${BENCH:-0}" = "1" ]; then
    echo "======================= BENCHMARK ======================="
    python rno_train.py "${ARGS[@]}" --bench
else
    echo "======================= TRAINING ======================="
    python rno_train.py "${ARGS[@]}"
fi

echo
echo "Job finished at $(date)"
