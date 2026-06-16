# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for all-reduce collective operation.
"""

import pytest
import torch
import torch.distributed as dist
import iris
from iris.ccl import Config


@pytest.mark.parametrize(
    "variant",
    [
        "atomic",
        # "ring",
        "two_shot",
        "one_shot_legacy",
        "one_shot",
        # TODO enable these tests when support for cache-modifiers is in place.
        # "spinlock",
    ],
)
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
        (128, 64, 32, 64),  # Small
        (128, 128, 32, 32),  # BLOCK_N < N/world_size (partial-width, multi-block per rank)
        (256, 128, 32, 16),  # Minimum BLOCK_N=16 (16-bit vectorization path)
        (1024, 256, 32, 64),  # Medium
        (8192, 8192, 32, 64),  # Large
    ],
)
def test_all_reduce(variant, dtype, M, N, block_size_m, block_size_n):
    """Test all-reduce functionality by comparing against PyTorch's implementation."""
    # Ensure torch.distributed is initialized (should be done by test runner)
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33  # 8GB
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()

    # PyTorch's all_reduce format: each rank has M x N data
    # All ranks compute the sum of all tensors
    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    # Fill with deterministic values for easier debugging
    pytorch_input_tensor.fill_(float(rank + 1))

    # Run PyTorch's all_reduce to get reference output
    pytorch_output_tensor = pytorch_input_tensor.clone()
    ctx.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Now set up Iris all_reduce format
    # Iris format: same as PyTorch - input and output are both (M, N)
    iris_input_tensor = ctx.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = ctx.zeros((M, N), dtype=dtype)

    # Run Iris all_reduce with specified variant
    ctx.barrier()
    config = Config(all_reduce_variant=variant, block_size_m=block_size_m, block_size_n=block_size_n)
    if variant == "two_shot":
        # Test both distribution modes for two_shot
        config.all_reduce_distribution = 0  # striding
    if variant == "ring":
        config.all_reduce_num_rings = min(2, config.comm_sms)

    # Explicitly call preamble to ensure proper initialization and synchronization
    # This helps with test isolation when tests run sequentially
    workspace = ctx.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    ctx.barrier()  # Ensure all ranks have completed preamble before starting kernel

    # Now call all_reduce with the prepared workspace
    ctx.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    # Compare results
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris output doesn't match PyTorch's all_reduce (variant={variant})"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: ctx.barrier() already does cuda.synchronize()
        ctx.barrier()
        # Explicitly delete the ctx instance to trigger cleanup
        del ctx
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


@pytest.mark.parametrize(
    "distribution",
    [
        0,  # striding
        1,  # block
    ],
)
def test_all_reduce_two_shot_distribution(distribution, dtype=torch.float32, M=1024, N=256):
    """Test two-shot all-reduce with different distribution modes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()

    pytorch_input_tensor = torch.randn(M, N, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input_tensor.fill_(float(rank + 1))

    pytorch_output_tensor = pytorch_input_tensor.clone()
    ctx.barrier()
    dist.all_reduce(pytorch_output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input_tensor = ctx.zeros((M, N), dtype=dtype)
    iris_input_tensor.copy_(pytorch_input_tensor)

    iris_output_tensor = ctx.zeros((M, N), dtype=dtype)

    ctx.barrier()
    config = Config(all_reduce_variant="two_shot", all_reduce_distribution=distribution)

    # Explicitly call preamble to ensure proper initialization and synchronization
    workspace = ctx.ccl.all_reduce_preamble(iris_output_tensor, iris_input_tensor, config=config)
    ctx.barrier()  # Ensure all ranks have completed preamble before starting kernel

    # Now call all_reduce with the prepared workspace
    ctx.ccl.all_reduce(iris_output_tensor, iris_input_tensor, config=config, workspace=workspace)
    torch.cuda.synchronize()

    atol = 1e-5
    max_diff = torch.abs(iris_output_tensor - pytorch_output_tensor).max().item()

    try:
        assert torch.allclose(iris_output_tensor, pytorch_output_tensor, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: Iris two-shot output doesn't match PyTorch (distribution={distribution})"
        )
    finally:
        # Final barrier to ensure all ranks complete before test cleanup
        # This helps with test isolation when running multiple tests
        # Note: ctx.barrier() already does cuda.synchronize()
        ctx.barrier()
        # Explicitly delete the ctx instance to trigger cleanup
        del ctx
        # Force garbage collection to ensure IPC handles are cleaned up
        import gc

        gc.collect()


def test_all_reduce_spinlock_lock_too_small():
    """Test that ValueError is raised when the spinlock lock array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then all_reduce
    is called with smaller block sizes (more tiles). workspace.matches() skips the preamble,
    and the undersized lock array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)

    M, N = 512, 512

    iris_input = ctx.zeros((M, N), dtype=torch.float32)
    iris_output = ctx.zeros((M, N), dtype=torch.float32)

    ctx.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller lock array
    config_large = Config(all_reduce_variant="spinlock", block_size_m=128, block_size_n=128)
    workspace = ctx.ccl.all_reduce_preamble(iris_output, iris_input, config=config_large)

    # Step 2: call all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized lock array is detected.
    config_small = Config(all_reduce_variant="spinlock", block_size_m=64, block_size_n=64)
    with pytest.raises(ValueError, match="Lock array too small"):
        ctx.ccl.all_reduce(iris_output, iris_input, config=config_small, workspace=workspace)

    ctx.barrier()
    del ctx
    import gc

    gc.collect()


@pytest.mark.parametrize("numel", [1024, 4096, 16384, 32768, 65536, 131072])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_all_reduce_one_shot_small(numel, dtype):
    """Test one_shot at vLLM-relevant small message sizes."""
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()

    pytorch_input = torch.randn(numel, dtype=dtype, device=f"cuda:{rank}")
    pytorch_input.fill_(float(rank + 1))

    pytorch_output = pytorch_input.clone()
    ctx.barrier()
    dist.all_reduce(pytorch_output, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iris_input = ctx.zeros((1, numel), dtype=dtype)
    iris_input.view(-1).copy_(pytorch_input)
    iris_output = ctx.zeros((1, numel), dtype=dtype)

    ctx.barrier()
    config = Config(all_reduce_variant="one_shot")
    workspace = ctx.ccl.all_reduce_preamble(iris_output, iris_input, config=config)
    ctx.barrier()
    ctx.ccl.all_reduce(iris_output, iris_input, config=config, workspace=workspace)
    torch.cuda.synchronize()

    atol = 1e-3 if dtype == torch.float16 else 1e-3
    max_diff = torch.abs(iris_output.view(-1) - pytorch_output).max().item()

    try:
        assert torch.allclose(iris_output.view(-1), pytorch_output, atol=atol), (
            f"Max difference: {max_diff}, expected < {atol}\n"
            f"Rank {rank}: one_shot output doesn't match PyTorch (numel={numel}, dtype={dtype})"
        )
    finally:
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


def _all_reduce_step(impl, src, stage_buf, result, ctx=None, config=None, workspace=None):
    """One replay unit: stage src into the input buffer, then all-reduce.

    Module-level (not a closure) so the test can `del ctx` for IPC cleanup
    without an enclosing closure keeping the instance alive. async_op=True
    matches how the vLLM/aiter communicator dispatches (no trailing barrier).
    """
    if impl == "torch":
        result.copy_(src)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    else:
        stage_buf.copy_(src)
        ctx.ccl.all_reduce(result, stage_buf, config=config, workspace=workspace, async_op=True)


@pytest.mark.parametrize("impl", ["torch", "one_shot", "one_shot_gluon"])
@pytest.mark.parametrize("vary", [False, True])
@pytest.mark.parametrize("numel", [4096, 32768, 131072])
def test_all_reduce_graph_capture(impl, vary, numel, dtype=torch.bfloat16):
    """HIP-graph capture/replay of all-reduce, across impls and input regimes.

    Generalizes the original one_shot, identical-input capture test with two
    axes that surface a cross-rank race the original could not see:

      impl  "torch" is the known-good control (torch.distributed through the
            identical harness; it must pass every cell). "one_shot" (Triton) and
            "one_shot_gluon" (the variant the vLLM/aiter communicator dispatches
            to) are under test.
      vary  False replays the SAME input, so a stale peer read returns
            identical-correct data and passes (the original coverage). True
            copies a fresh input into the static buffer before each replay and
            checks each against its own reference -- how the communicator drives
            the collective per token -- so a dropped/stale peer slot surfaces.

    Inputs are small integers (rank r contributes 1 + r + replay%16) so the
    reduced sum is exact in bf16/fp16 and any >=1 mismatch is a real drop, not
    fp rounding.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    num_replays = 200
    heap_size = 2**33
    ctx = iris.iris(heap_size)
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()

    # src holds each replay's activation, off the symmetric heap (like the input
    # vLLM hands the communicator before each captured step).
    src = torch.empty((1, numel), dtype=dtype, device=f"cuda:{rank}")

    if impl == "torch":
        stage_buf = None
        result = torch.empty((1, numel), dtype=dtype, device=f"cuda:{rank}")
        config = workspace = None
    else:
        stage_buf = ctx.zeros((1, numel), dtype=dtype)
        result = ctx.zeros((1, numel), dtype=dtype)
        config = Config(all_reduce_variant=impl, use_gluon=(impl == "one_shot_gluon"))
        workspace = ctx.ccl.all_reduce_preamble(result, stage_buf, config=config)
    ctx.barrier()

    def fill_src(replay):
        src.fill_(float(1 + rank + (replay % 16)))

    def expected(replay):
        # sum_r (1 + r + replay%16) = world*(1 + replay%16) + world*(world-1)/2
        return float(world_size * (1 + (replay % 16)) + world_size * (world_size - 1) // 2)

    # Warmup (JIT / lazy setup) then capture copy+all_reduce as one unit.
    fill_src(0)
    _all_reduce_step(impl, src, stage_buf, result, ctx=ctx, config=config, workspace=workspace)
    torch.cuda.synchronize()
    ctx.barrier()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        _all_reduce_step(impl, src, stage_buf, result, ctx=ctx, config=config, workspace=workspace)
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
            max_diff = torch.abs(result.view(-1) - expected(replay)).max().item()
            if not max_diff <= atol:
                failures.append((i, round(max_diff, 4)))
        print(
            f"[rank {rank}] all_reduce graph impl={impl} vary={vary} numel={numel}: "
            f"{num_replays - len(failures)}/{num_replays} ok" + (f"  FAIL first={failures[:8]}" if failures else ""),
            flush=True,
        )
        assert not failures, (
            f"impl={impl} vary={vary} numel={numel}: {len(failures)}/{num_replays} replays wrong "
            f"(first replay {failures[0][0]}, max|diff|={failures[0][1]}). "
            f"torch and vary=False must pass; one_shot_gluon+vary=True failing localizes the bug to the gluon kernel."
        )
    finally:
        del graph
        ctx.barrier()
        del ctx
        import gc

        gc.collect()


def test_all_reduce_ring_flags_too_small():
    """Test that ValueError is raised when the ring flags array is too small for current tile count.

    Scenario: workspace is prepared with larger block sizes (fewer tiles), then all_reduce
    is called with smaller block sizes (more tiles). workspace.matches() skips the preamble,
    and the undersized flags array is detected.
    """
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    heap_size = 2**33
    ctx = iris.iris(heap_size)
    world_size = ctx.get_num_ranks()

    M, N = 512, 512

    # Choose block_size_n values divisible by world_size for both configs
    # Use 128 and 64 which are divisible by typical world sizes (1, 2, 4, 8)
    block_size_n_large = (128 // world_size) * world_size
    block_size_n_small = (64 // world_size) * world_size
    if block_size_n_large == 0 or block_size_n_small == 0 or block_size_n_large == block_size_n_small:
        del ctx
        pytest.skip(f"Cannot create two distinct block sizes divisible by world_size={world_size}")

    iris_input = ctx.zeros((M, N), dtype=torch.float32)
    iris_output = ctx.zeros((M, N), dtype=torch.float32)

    ctx.barrier()

    # Step 1: run preamble with larger block sizes → allocates a smaller flags array
    config_large = Config(
        all_reduce_variant="ring",
        block_size_m=128,
        block_size_n=block_size_n_large,
    )
    workspace = ctx.ccl.all_reduce_preamble(iris_output, iris_input, config=config_large)

    # Step 2: call all_reduce with smaller block sizes that need more tiles —
    # workspace.matches() returns True (same shape/dtype/variant), preamble is skipped,
    # and the undersized flags array is detected.
    config_small = Config(
        all_reduce_variant="ring",
        block_size_m=64,
        block_size_n=block_size_n_small,
    )
    with pytest.raises(ValueError, match="Flags array too small"):
        ctx.ccl.all_reduce(iris_output, iris_input, config=config_small, workspace=workspace)

    ctx.barrier()
    del ctx
    import gc

    gc.collect()
