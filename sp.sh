#!/bin/bash
#SBATCH --job-name=rno_draw
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=sp_%j.out
#SBATCH --error=sp_%j.err

# Full redraw of every complete run folder and the index (no model, no GPU).

set -euo pipefail

module load python
source env/bin/activate

python make_submission.py
