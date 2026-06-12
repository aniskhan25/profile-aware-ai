#!/usr/bin/env python3
"""Distributed GEMM + all-reduce loop — one rank per GPU via srun."""

import os
import time
import torch
import torch.distributed as dist

SECONDS    = 60
SIZE       = 2048
DTYPE      = torch.float16
RESERVE_GB = 20.0

rank       = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
world_size = int(os.environ["SLURM_NTASKS"])

os.environ.setdefault("RANK",       str(rank))
os.environ.setdefault("LOCAL_RANK", str(local_rank))
os.environ.setdefault("WORLD_SIZE", str(world_size))
os.environ.setdefault("MASTER_PORT", "29500")

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl", init_method="env://")

_bytes = int(RESERVE_GB * 1024 ** 3)
_reserved = torch.empty(_bytes // 2, device=device, dtype=torch.float16)

if rank == 0:
    print(f"world_size={world_size}", flush=True)

a = torch.randn((SIZE, SIZE), device=device, dtype=DTYPE)
b = torch.randn((SIZE, SIZE), device=device, dtype=DTYPE)

# warmup
for _ in range(3):
    _ = a @ b
torch.cuda.synchronize()

start = time.time()
iters = 0
while time.time() - start < SECONDS:
    c = a @ b
    torch.cuda.synchronize()
    iters += 1

dist.barrier()
if rank == 0:
    print(f"done iters={iters} elapsed={time.time() - start:.1f}s", flush=True)
dist.destroy_process_group()
