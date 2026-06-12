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
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

CONTAINER_IMAGE_DEFAULT="/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif"
export LUMI_CONTAINER_IMAGE="${LUMI_CONTAINER_IMAGE:-${CONTAINER_IMAGE_DEFAULT}}"

PROFILER_DIR="${PROFILER_DIR:-/scratch/project_462000131/anisrahm/lumi-job-profiler}"
source "${PROFILER_DIR}/scripts/profile_hook.sh"

DEMO="${PROFILER_DIR}/examples/demo_pytorch_rocm.py"

profile_run -- python3 "${DEMO}" \
  --seconds 60 \
  --size 4096 \
  --dtype fp16
