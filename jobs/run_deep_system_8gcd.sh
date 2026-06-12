#!/bin/bash
#SBATCH --job-name=profile-deep-system-8gcd
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
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

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PROFILER_DIR="${PROFILER_DIR:-/scratch/project_462000131/anisrahm/lumi-job-profiler}"

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export LUMI_PROFILE_MODE=deep-system
export ROCPROFSYS_INSTALL_PREFIX="${ROCPROFSYS_INSTALL_PREFIX:-/scratch/project_462000131/${USER}/tools/rocprofiler-systems-container}"

source "${PROFILER_DIR}/scripts/profile_hook.sh"

DEMO="${REPO_DIR}/examples/demo_pytorch_distributed_rocm.py"

profile_run_distributed -- \
  srun --nodes=1 --ntasks-per-node=8 --cpu-bind=none -- \
  python3 "${DEMO}" --seconds 30 --size 2048 --dtype fp16
