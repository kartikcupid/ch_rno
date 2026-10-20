#!/bin/bash
#SBATCH --job-name=rno_ch
#SBATCH --partition=dgx_fat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=rtp_%j.out
#SBATCH --error=rtp_%j.err

# Train, predict to 10000 and draw one RNO.
#
#   sbatch --export=ALL,END=1000 rtp.sh                     Laplacian output (lap)
#   sbatch --export=ALL,END=1000,VARIANT=base rtp.sh        direct-increment head
#   sbatch --export=ALL,END=1000,WIDTH=24,MODES=12 rtp.sh   a smaller model
#   sbatch --export=ALL,END=1000,LAYERS=3 rtp.sh            3 RNO layers
#   sbatch --export=ALL,END=1000,SEED=1 rtp.sh              a seed replicate
#
# The label is rno_t<END>_lap_pred_t10000 (VARIANT=base: rno_t<END>_pred_t10000;
# a non-default size or seed adds a tag, e.g. rno_t1000_lap_w24m12_pred_t10000).
# Training resumes from Work/models/<run>/rno_ch_state.pt and prediction from
# the rollout cache, so a job that hits the wall clock can be resubmitted as is.
# The run's Results/ folder is drawn by rno_predict.py; the last step only
# refreshes Results/README.md and the comparison images.

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

echo "======================= TRAINING ======================="
python rno_train.py "${ARGS[@]}"
echo
echo "======================= PREDICTION ======================="
python rno_predict.py "${ARGS[@]}"

echo
echo "======================= SUBMISSION ======================="
python make_submission.py --overview

echo
echo "Job finished at $(date)"
