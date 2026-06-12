#!/bin/bash
#SBATCH --job-name=profile-light-8gcd
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

PROFILER_DIR="${PROFILER_DIR:-/scratch/project_462000131/anisrahm/lumi-job-profiler}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID: -4}"
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

source "${PROFILER_DIR}/scripts/profile_hook.sh"

DEMO="${SLURM_SUBMIT_DIR}/examples/demo_pytorch_distributed_rocm.py"

profile_start
srun --cpu-bind=v,mask_cpu=${CPU_BIND_MASKS} singularity run "${LUMI_CONTAINER_IMAGE}" \
  bash -c "export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python3 ${DEMO} --seconds 60 --size 2048 --dtype fp16"
profile_stop
profile_summarize
