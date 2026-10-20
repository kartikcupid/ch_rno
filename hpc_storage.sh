#!/bin/bash
#SBATCH --job-name=hpc_storage
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=hs_%j.out
#SBATCH --error=hs_%j.err

# Used / available storage as seen from a compute node (hpc_storage.py is
# stdlib-only, so the venv is not needed).  Arguments after the script name are
# passed straight to hpc_storage.py:
#
#   sbatch hpc_storage.sh                      standard locations + quota
#   sbatch hpc_storage.sh /path/a /path/b      plus extra paths
#   sbatch hpc_storage.sh --du Work Results    plus `du -sh` of those paths
#
# `du` walks every file, which is slow on a shared filesystem and unwelcome on
# a login node, so run --du here, not interactively.  /tmp and $TMPDIR are
# node-local: the job reports the compute node's, not the login node's.
# Output: hs_<jobid>.out

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

module load python

echo "Python Path: $(which python3)"

python3 hpc_storage.py "$@"

echo
echo "Job finished at $(date)"
