#!/bin/bash
#SBATCH --job-name=profile-light-1gcd
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

LUMI_CONTAINER_IMAGE=${LUMI_CONTAINER_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
PROFILER_DIR=${PROFILER_DIR:-$SCRATCH/lumi-job-profiler}

source "$PROFILER_DIR/scripts/profile_hook.sh"

DEMO=${DEMO:-$PROFILER_DIR/examples/demo_pytorch_rocm.py}

profile_run singularity exec \
  --bind "$PWD,$SCRATCH" \
  --rocm \
  "$LUMI_CONTAINER_IMAGE" \
  python "$DEMO" \
    --seconds 60 \
    --size 4096 \
    --dtype fp16
