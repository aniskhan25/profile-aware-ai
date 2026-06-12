#!/usr/bin/env python3
"""Single-GPU GEMM loop — generates GPU utilisation and memory activity."""

import time
import torch

SECONDS      = 60
SIZE         = 4096
DTYPE        = torch.float16
RESERVE_GB   = 20.0

device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Reserve a fixed block of VRAM so memory utilisation appears in the profile.
_bytes = int(RESERVE_GB * 1024 ** 3)
_reserved = torch.empty(_bytes // 2, device=device, dtype=torch.float16)

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
