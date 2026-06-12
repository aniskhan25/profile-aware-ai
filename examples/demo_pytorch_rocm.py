#!/usr/bin/env python3
"""Single-GPU GEMM loop — generates GPU utilisation and memory activity."""

import time
import torch

SECONDS = 60
SIZE    = 4096
DTYPE   = torch.float16

device = torch.device("cuda:0")
torch.cuda.set_device(device)

a = torch.randn((SIZE, SIZE), device=device, dtype=DTYPE)
b = torch.randn((SIZE, SIZE), device=device, dtype=DTYPE)

# warmup
for _ in range(5):
    _ = a @ b
torch.cuda.synchronize()

start = time.time()
iters = 0
while time.time() - start < SECONDS:
    _ = a @ b
    torch.cuda.synchronize()
    iters += 1

print(f"done iters={iters} elapsed={time.time() - start:.1f}s")
