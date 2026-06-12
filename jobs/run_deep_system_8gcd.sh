#!/bin/bash
#SBATCH --job-name=profile-deep-system-8gcd
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
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
export SINGULARITY_BIND="${SINGULARITY_BIND:+${SINGULARITY_BIND},}/usr/lib64:/opt/hostlibs,/opt/rocm-6.3.4/lib:/opt/rocm6libs"
export SINGULARITYENV_LD_LIBRARY_PATH="/opt/hostlibs:/opt/rocm6libs"

PROFILER_DIR="${PROFILER_DIR:-/scratch/project_462000131/anisrahm/lumi-job-profiler}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID: -4}"
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE
export LUMI_PROFILE_MODE=deep-system
export ROCPROFSYS_INSTALL_PREFIX="${ROCPROFSYS_INSTALL_PREFIX:-/scratch/project_462000131/${USER}/tools/rocprofiler-systems-container}"

source "${PROFILER_DIR}/scripts/profile_hook.sh"

DEMO="${SLURM_SUBMIT_DIR}/examples/demo_pytorch_distributed_rocm.py"

profile_run_distributed -- \
  srun --nodes=1 --ntasks-per-node=8 --cpu-bind=none -- \
  python3 "${DEMO}" --seconds 30 --size 2048 --dtype fp16
