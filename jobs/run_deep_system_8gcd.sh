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

# rocprofiler-systems needs librocprofiler-sdk.so.0 and libamd_smi.so.25 from
# ROCm 6.3.4 (it was built against ROCm 6.x sonames). Binding all of
# /opt/rocm-6.3.4/lib poisons LD_LIBRARY_PATH with ROCm 6.x libamdhip64.so.7,
# breaking the container's ROCm 7.0.2 stack. Stage only the two needed files.
ROCM6_SDK_STAGING="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/tools/rocm6sdklibs"
mkdir -p "$ROCM6_SDK_STAGING"
cp -n /opt/rocm-6.3.4/lib/librocprofiler-sdk.so.0 "$ROCM6_SDK_STAGING/" 2>/dev/null || true
cp -n /opt/rocm-6.3.4/lib/libamd_smi.so.25 "$ROCM6_SDK_STAGING/" 2>/dev/null || true

export SINGULARITY_BIND="${SINGULARITY_BIND:+${SINGULARITY_BIND},}/usr/lib64:/opt/hostlibs,${ROCM6_SDK_STAGING}:/opt/rocm6sdklibs"
export SINGULARITYENV_LD_LIBRARY_PATH="/opt/hostlibs:/opt/rocm6sdklibs"

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
