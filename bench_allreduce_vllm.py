#!/usr/bin/env python3
"""Benchmark iris all-reduce variants vs RCCL at small message sizes.

Usage:
    torchrun --nproc_per_node=8 bench_allreduce_vllm.py
"""

import os
import time

import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


WARMUP = 200
ITERS = 500
DTYPE = torch.bfloat16

SIZES_BYTES = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
]

VARIANTS = [
    ("rccl", None),
    ("one_shot_vllm", Config(all_reduce_variant="one_shot_vllm")),
    ("two_shot", Config(block_size_m=32, block_size_n=64, all_reduce_distribution=1)),
    ("one_shot", Config(block_size_m=32, block_size_n=64, all_reduce_variant="one_shot")),
]


def bench_rccl(inp, out, warmup, iters):
    for _ in range(warmup):
        out.copy_(inp)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out.copy_(inp)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def bench_iris(inp, out, ctx, config, warmup, iters):
    workspace = ctx.ccl.all_reduce_preamble(out, inp, config=config)
    ctx.barrier()

    for _ in range(warmup):
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        ctx.ccl.all_reduce(out, inp, config=config, workspace=workspace)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def format_size(nbytes):
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.0f}MB"
    return f"{nbytes / 1024:.0f}KB"


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()

    ctx = iris.iris(2**33)
    elem_size = torch.tensor([], dtype=DTYPE).element_size()

    results = {}

    for size_bytes in SIZES_BYTES:
        numel = size_bytes // elem_size
        M, N = 1, numel

        results[size_bytes] = {}

        for name, config in VARIANTS:
            if name == "rccl":
                inp = torch.randn(M, N, dtype=DTYPE, device=f"cuda:{rank}")
                inp.fill_(float(rank + 1))
                out = inp.clone()
                lat = bench_rccl(inp, out, WARMUP, ITERS)
            else:
                if name in ("one_shot", "two_shot"):
                    inp = ctx.zeros((M, N), dtype=DTYPE)
                else:
                    inp = ctx.zeros((M, N), dtype=DTYPE)
                inp.fill_(float(rank + 1))
                out = ctx.zeros((M, N), dtype=DTYPE)
                lat = bench_iris(inp, out, ctx, config, WARMUP, ITERS)

            results[size_bytes][name] = lat

    if rank == 0:
        header = f"{'Size':>8s}"
        for name, _ in VARIANTS:
            header += f" | {name:>16s}"
        header += f" | {'speedup':>8s}"
        print()
        print(f"All-Reduce Benchmark — {world_size} GPUs, {DTYPE}, {ITERS} iters")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for size_bytes in SIZES_BYTES:
            row = f"{format_size(size_bytes):>8s}"
            rccl_lat = results[size_bytes]["rccl"]
            vllm_lat = results[size_bytes].get("one_shot_vllm", rccl_lat)
            for name, _ in VARIANTS:
                lat = results[size_bytes][name]
                row += f" | {lat * 1e6:>13.1f} us"
            speedup = rccl_lat / vllm_lat if vllm_lat > 0 else 0
            row += f" | {speedup:>7.2f}x"
            print(row)

        print()
        bus_header = f"{'Size':>8s}"
        for name, _ in VARIANTS:
            bus_header += f" | {name:>16s}"
        print("Bus Bandwidth (GB/s)")
        print("=" * len(bus_header))
        print(bus_header)
        print("-" * len(bus_header))

        for size_bytes in SIZES_BYTES:
            row = f"{format_size(size_bytes):>8s}"
            for name, _ in VARIANTS:
                lat = results[size_bytes][name]
                bw = 2 * (world_size - 1) / world_size * size_bytes / lat / 1e9
                row += f" | {bw:>13.2f} GB"
                pass
            print(row)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
