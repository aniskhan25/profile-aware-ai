#!/usr/bin/env python3
"""Minimal distributed PyTorch ROCm demo workload for LUMI.

Expects one rank per GPU launched via srun. Initialises torch.distributed,
runs sustained GEMMs on the local GPU, and periodically executes an
all-reduce to generate distributed compute and communication activity.
"""

import argparse
import os
import socket
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed PyTorch ROCm demo workload")
    parser.add_argument("--seconds", type=int, default=30, help="Runtime duration")
    parser.add_argument("--size", type=int, default=2048, help="Matrix size (NxN)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Compute dtype")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--sync-interval", type=int, default=5, help="Run all-reduce every N iterations")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations on rank 0")
    return parser.parse_args()


def dtype_from_str(name):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def setup_distributed_env():
    """Map Slurm env vars to torch.distributed env vars when not already set."""
    for src, dst in [
        ("SLURM_PROCID",  "RANK"),
        ("SLURM_NTASKS",  "WORLD_SIZE"),
        ("SLURM_LOCALID", "LOCAL_RANK"),
    ]:
        if dst not in os.environ and src in os.environ:
            os.environ[dst] = os.environ[src]
    os.environ.setdefault("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ:
        fallback = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "")
        if fallback:
            os.environ["MASTER_ADDR"] = fallback


def main():
    args = parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        print(f"PyTorch distributed not available: {exc}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False — ROCm/CUDA not available.")
        sys.exit(1)

    setup_distributed_env()

    if "MASTER_ADDR" not in os.environ:
        print("MASTER_ADDR is not set. Export MASTER_ADDR before launching.", flush=True)
        sys.exit(1)

    rank       = int(os.environ.get("SLURM_PROCID",  0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS",  1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype  = dtype_from_str(args.dtype)

    dist.init_process_group(backend="nccl", init_method="env://")

    props = torch.cuda.get_device_properties(device)
    if rank == 0:
        print(f"world_size={world_size} backend=nccl")
    print(
        f"rank={rank} local_rank={local_rank} host={socket.gethostname()} "
        f"device={props.name} total_mem_gb={props.total_memory / (1024**3):.2f}",
        flush=True,
    )

    torch.manual_seed(rank)
    a = torch.randn((args.size, args.size), device=device, dtype=dtype)
    b = torch.randn((args.size, args.size), device=device, dtype=dtype)
    sync_tensor = torch.tensor([float(rank + 1)], device=device)

    for _ in range(args.warmup):
        c = a @ b
        sync_tensor += c.float().mean()
        dist.all_reduce(sync_tensor)
        torch.cuda.synchronize()

    start    = time.time()
    iters    = 0
    last_log = start

    while time.time() - start < args.seconds:
        c = a @ b
        if args.sync_interval > 0 and iters % args.sync_interval == 0:
            sync_tensor.copy_(c.float().mean().reshape(1))
            dist.all_reduce(sync_tensor)
        torch.cuda.synchronize()
        iters += 1

        if rank == 0 and iters % args.log_interval == 0:
            now = time.time()
            print(
                f"iter={iters} elapsed={int(now - start)}s "
                f"interval={now - last_log:.2f}s reduced={float(sync_tensor.item()):.4f}",
                flush=True,
            )
            last_log = now

    dist.barrier()
    elapsed = time.time() - start
    if rank == 0:
        print(f"done iters={iters} elapsed={elapsed:.2f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
