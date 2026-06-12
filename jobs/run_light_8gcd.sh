#!/bin/bash
#SBATCH --job-name=profile-light-8gcd
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

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

LUMI_CONTAINER_IMAGE=${LUMI_CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
PROFILER_DIR=${PROFILER_DIR:-/scratch/project_462000131/anisrahm/lumi-job-profiler}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

source "$PROFILER_DIR/scripts/profile_hook.sh"

DEMO=${DEMO:-$PROFILER_DIR/examples/demo_pytorch_distributed_rocm.py}

profile_run srun singularity exec \
  --bind "$PWD,$SCRATCH" \
  --rocm \
  "$LUMI_CONTAINER_IMAGE" \
  python "$DEMO" \
    --seconds 60 \
    --size 2048 \
    --dtype fp16
