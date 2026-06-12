#!/bin/bash
#SBATCH --job-name=profile-light-1gcd
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

export LUMI_CONTAINER_IMAGE="${LUMI_CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
export LUMI_CONTAINER_USE_ROCM=0

PROFILER_DIR="${PROFILER_DIR:-/scratch/project_462000131/anisrahm/lumi-job-profiler}"
source "${PROFILER_DIR}/scripts/profile_hook.sh"

DEMO="${SLURM_SUBMIT_DIR}/examples/demo_pytorch_rocm.py"

profile_run -- python3 "${DEMO}" \
  --seconds 60 \
  --size 4096 \
  --dtype fp16
