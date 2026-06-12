#!/usr/bin/env python3
"""Minimal PyTorch ROCm demo workload.

Runs a sustained GEMM loop on GPU to generate utilization and memory activity.
"""

import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ROCm demo workload")
    parser.add_argument("--seconds", type=int, default=30, help="Runtime duration")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (NxN)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Compute dtype")
    parser.add_argument("--device", default="cuda:0", help="Device string (default: cuda:0)")
    parser.add_argument("--reserve-mem-gb", type=float, default=0.0, help="Optional GPU memory to reserve")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N iterations")
    return parser.parse_args()


def dtype_from_str(name):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def reserve_memory(device, dtype, gb):
    import torch
    if gb <= 0:
        return None
    bytes_per_elem = torch.tensor([], device=device, dtype=dtype).element_size()
    num_elems = int((gb * (1024 ** 3)) / bytes_per_elem)
    return torch.empty(num_elems, device=device, dtype=dtype) if num_elems > 0 else None


def main():
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        print(f"PyTorch not available: {exc}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is False — ROCm/CUDA not available.")
        sys.exit(1)

    device = torch.device(args.device)
    dtype = dtype_from_str(args.dtype)

    torch.manual_seed(0)
    torch.cuda.set_device(device.index if device.index is not None else 0)

    props = torch.cuda.get_device_properties(device)
    print(f"device={props.name} total_mem_gb={props.total_memory / (1024**3):.2f}")
    if torch.version.hip:
        print(f"ROCm version: {torch.version.hip}")

    reserve_memory(device, dtype, args.reserve_mem_gb)

    n = args.size
    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)

    for _ in range(args.warmup):
        c = a @ b
        torch.cuda.synchronize()

    print("Starting compute loop...")
    start = time.time()
    iters = 0
    last_log = start

    while time.time() - start < args.seconds:
        c = a @ b
        if dtype != torch.float32:
            c = c.float().sum()
        torch.cuda.synchronize()
        iters += 1

        if iters % args.log_interval == 0:
            now = time.time()
            print(f"iter={iters} elapsed={int(now - start)}s interval={now - last_log:.2f}s")
            last_log = now

    print(f"done iters={iters} elapsed={time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
