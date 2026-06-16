# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-gather collective operation using Gluon.
"""

import os

import pytest
import torch
import torch.distributed as dist

# Try to import Gluon, skip tests if not available
try:
    import iris
    from iris.ccl import Config
    from triton.experimental import gluon  # noqa: F401

    GLUON_AVAILABLE = True
except ImportError:
    GLUON_AVAILABLE = False


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        # block_size_n must be a multiple of (threads_per_warp * num_warps).
        # With defaults (threads_per_warp=64, num_warps=4), minimum is 256.
        # elems_per_thread = block_size_n / 256: higher = wider vector loads.
        (256, 256, 32, 256),  # Small: elems_per_thread=1 (scalar loads)
        (1024, 512, 32, 512),  # Medium: elems_per_thread=2 (dword loads)
        (8192, 8192, 32, 1024),  # Large: elems_per_thread=4 (dwordx4, optimal)
    ],
)
def test_all_gather_gluon(dtype, M, N, block_size_m, block_size_n):
    """Test all-gather functionality using Gluon by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    # Size heap to fit input (M*N) + output (max_ranks*M*N) with headroom
    max_ranks = int(os.environ.get("WORLD_SIZE", 8))
    elem_size = torch.tensor([], dtype=dtype).element_size()
    needed = (1 + max_ranks) * M * N * elem_size
    heap_size = max(2**30, int(needed * 2))  # 2x headroom, minimum 1GB
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # Each rank has an M x N input tensor
    # Output is (world_size * M, N) - concatenated along dimension 0
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

    # Create output tensor for PyTorch: (world_size * M, N)
    pytorch_output_tensor = torch.zeros(world_size * M, N, dtype=dtype, device=f"cuda:{rank}")

    # Run PyTorch's all_gather_into_tensor to get reference output
    shmem.barrier()
    dist.all_gather_into_tensor(pytorch_output_tensor, pytorch_input_tensor)
    torch.cuda.synchronize()

    # Now set up Iris Gluon all_gather
    iris_input_tensor = shmem.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = shmem.zeros((world_size * M, N), dtype=dtype)

    # Run Iris Gluon all_gather
    shmem.barrier()
    config = Config(use_gluon=True, block_size_m=block_size_m, block_size_n=block_size_n)
    shmem.ccl.all_gather(iris_output_tensor, iris_input_tensor, config=config)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris Gluon output doesn't match PyTorch's all_gather_into_tensor"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: shmem.barrier() already does cuda.synchronize()
        shmem.barrier()
        # Explicitly delete the shmem instance to trigger cleanup
        del shmem
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


def _all_gather_step(impl, src, stage_buf, result, shmem=None, config=None):
    """One replay unit: stage src into the input buffer, then all-gather.

    Module-level (not a closure) so the test can `del shmem` for IPC cleanup
    without an enclosing closure keeping the instance alive.
    """
    stage_buf.copy_(src)
    if impl == "gluon":
        # async_op=True matches the communicator; async_op=False runs a trailing
        # barrier that host-syncs, which is illegal inside graph capture.
        shmem.ccl.all_gather(result, stage_buf, config=config, async_op=True)
    else:
        dist.all_gather_into_tensor(result, stage_buf)


@pytest.mark.skipif(not GLUON_AVAILABLE, reason="Gluon not available")
@pytest.mark.parametrize("impl", ["torch", "gluon"])
@pytest.mark.parametrize("vary", [False, True])
@pytest.mark.parametrize(
    "M, N, block_size_m, block_size_n",
    [
        (64, 8192, 32, 1024),
        (256, 8192, 32, 1024),
    ],
)
def test_all_gather_gluon_graph_capture(impl, vary, M, N, block_size_m, block_size_n, dtype=torch.bfloat16):
    """HIP-graph capture/replay of the gluon all-gather, across input regimes.

    all-gather had NO graph-capture coverage (the existing test is eager); this
    adds it with the same axes as the all-reduce twin in test_all_reduce.py
    (test_all_reduce_graph_capture):

      impl  "torch" is the known-good control (must pass every cell); "gluon" is
            the path the vLLM/aiter communicator dispatches to.
      vary  False replays the SAME input (a stale block read returns identical-
            correct data and passes); True copies a fresh input each replay and
            checks each gathered block against its own value, so a dropped/stale
            peer slot surfaces -- how the communicator drives it per token.

    Inputs are small integers (rank r's block = 1 + r + replay%16), exact in
    bf16/fp16, so any >=1 mismatch is a real drop.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    num_replays = 200
    max_ranks = int(os.environ.get("WORLD_SIZE", 8))
    elem_size = torch.tensor([], dtype=dtype).element_size()
    needed = (1 + max_ranks) * M * N * elem_size
    heap_size = max(2**30, int(needed * 2))
    shmem = iris.iris(heap_size)
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()

    # src holds each replay's activation, off the symmetric heap.
    src = torch.empty((M, N), dtype=dtype, device=f"cuda:{rank}")

    if impl == "gluon":
        stage_buf = shmem.zeros((M, N), dtype=dtype)
        result = shmem.zeros((world_size * M, N), dtype=dtype)
        config = Config(use_gluon=True, block_size_m=block_size_m, block_size_n=block_size_n)
    else:
        stage_buf = torch.empty((M, N), dtype=dtype, device=f"cuda:{rank}")
        result = torch.empty((world_size * M, N), dtype=dtype, device=f"cuda:{rank}")
        config = None
    shmem.barrier()

    def fill_src(replay):
        src.fill_(float(1 + rank + (replay % 16)))

    # Warmup (runs any lazy setup) then capture copy+all_gather as one unit.
    fill_src(0)
    _all_gather_step(impl, src, stage_buf, result, shmem=shmem, config=config)
    torch.cuda.synchronize()
    shmem.barrier()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        _all_gather_step(impl, src, stage_buf, result, shmem=shmem, config=config)
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)

    atol = 0.5  # inputs are exact integers; >=1 mismatch is a real drop
    failures = []
    try:
        for i in range(num_replays):
            replay = i if vary else 0
            fill_src(replay)
            graph.replay()
            torch.cuda.synchronize()
            diffs = [
                torch.abs(result[r * M : (r + 1) * M] - float(1 + r + (replay % 16))).max().item()
                for r in range(world_size)
            ]
            bad = [r for r in range(world_size) if diffs[r] > atol]
            if bad:
                failures.append((i, round(max(diffs[r] for r in bad), 4), bad))
        print(
            f"[rank {rank}] all_gather graph impl={impl} vary={vary} {M}x{N}: "
            f"{num_replays - len(failures)}/{num_replays} ok" + (f"  FAIL first={failures[0]}" if failures else ""),
            flush=True,
        )
        assert not failures, (
            f"impl={impl} vary={vary} {M}x{N}: {len(failures)}/{num_replays} replays wrong "
            f"(first replay {failures[0][0]}, max|diff|={failures[0][1]}, bad blocks={failures[0][2]}). "
            f"torch and vary=False must pass; gluon+vary=True failing localizes the bug to the gluon kernel."
        )
    finally:
        del graph
        shmem.barrier()
        del shmem
        import gc

        gc.collect()
