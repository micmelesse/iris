# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import gc
import pytest
import torch
import triton
import triton.language as tl
import iris


BARRIER_TYPES = ["host", "device"]


def _call_barrier(shmem, barrier_type):
    if barrier_type == "host":
        shmem.barrier()
    else:
        shmem.device_barrier()


@triton.jit
def _read_remote_kernel(
    buf_ptr,
    result_ptr,
    cur_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    data = iris.load(buf_ptr + offsets, cur_rank, remote_rank, heap_bases)
    tl.store(result_ptr + offsets, data)


@triton.jit
def _write_remote_kernel(
    buf_ptr,
    value,
    cur_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    heap_bases: tl.tensor,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    data = tl.full([BLOCK_SIZE], value, dtype=tl.float32)
    iris.store(buf_ptr + offsets, data, cur_rank, remote_rank, heap_bases)


@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_basic(barrier_type):
    shmem = iris.iris(1 << 20)

    try:
        _call_barrier(shmem, barrier_type)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_multiple(barrier_type):
    shmem = iris.iris(1 << 20)

    try:
        for _ in range(10):
            _call_barrier(shmem, barrier_type)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize("num_barriers", [1, 2, 4])
@pytest.mark.parametrize("op", ["load", "store", "both"])
@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_cross_rank(barrier_type, op, num_barriers):
    """Verify cross-rank data visibility after barrier.

    - load: each rank reads neighbor's buffer via iris.load()
    - store: each rank writes to neighbor's buffer via iris.store()
    - both: load and store in the same test

    Parametrized over num_barriers to test idempotency: extra barriers
    with no new work must not corrupt state or deadlock.
    """
    N = 256
    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()
    num_ranks = shmem.get_num_ranks()
    heap_bases = shmem.get_heap_bases()
    neighbor = (rank + 1) % num_ranks
    writer = (rank - 1 + num_ranks) % num_ranks

    buf = shmem.zeros((N,), dtype=torch.float32)
    result = shmem.zeros((N,), dtype=torch.float32)

    try:
        if op in ("load", "both"):
            # Each rank writes its rank ID to its own buffer.
            buf.fill_(float(rank))

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            # Read neighbor's buffer.
            _read_remote_kernel[(1,)](
                buf, result, rank, neighbor, N, heap_bases,
            )

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            expected = torch.full((N,), float(neighbor), dtype=torch.float32, device="cuda")
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

        if op in ("store", "both"):
            # Reset buffer before store test.
            buf.fill_(0.0)

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            # Each rank writes its rank ID into neighbor's buffer.
            _write_remote_kernel[(1,)](
                buf, float(rank), rank, neighbor, N, heap_bases,
            )

            for _ in range(num_barriers):
                _call_barrier(shmem, barrier_type)

            # Each rank checks its own buffer was written by writer.
            expected = torch.full((N,), float(writer), dtype=torch.float32, device="cuda")
            torch.testing.assert_close(buf, expected, rtol=0, atol=0)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


@pytest.mark.parametrize(
    "barrier_type",
    [
        pytest.param("host", marks=pytest.mark.skip(reason="Host barrier has no reusable state")),
        "device",
    ],
)
def test_barrier_state_reuse(barrier_type):
    shmem = iris.iris(1 << 20)

    try:
        _call_barrier(shmem, barrier_type)
        assert None in shmem._device_barrier_state
        flags_ptr = shmem._device_barrier_state[None][0].data_ptr()

        _call_barrier(shmem, barrier_type)
        assert shmem._device_barrier_state[None][0].data_ptr() == flags_ptr
        assert shmem._device_barrier_state[None][1] == 2
    finally:
        shmem.barrier()
        del shmem
        gc.collect()


# Host barrier is not graph-capturable (uses NCCL which crashes with
# hipErrorStreamCaptureUnsupported on ROCm). Only test device barrier here.
# To experiment with host, add "host" back to the parametrize list.
@pytest.mark.parametrize("barrier_type", ["device"])
def test_barrier_graph_capture(barrier_type, destroy_pg, recreate_pg):
    shmem = iris.iris(1 << 20)

    try:
        _call_barrier(shmem, barrier_type)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _call_barrier(shmem, barrier_type)
        stream.synchronize()

        if barrier_type == "host":
            # Host barrier needs PG destroyed to stop the NCCL watchdog
            # which crashes with hipErrorStreamCaptureUnsupported on ROCm.
            # PG destroy/recreate is broken upstream (pytorch#55967,
            # #66547, #119196) so this path is disabled by default.
            destroy_pg()
            with pytest.raises(Exception):
                with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                    _call_barrier(shmem, barrier_type)
            recreate_pg()
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _call_barrier(shmem, barrier_type)
            for _ in range(3):
                graph.replay()
                stream.synchronize()

        _call_barrier(shmem, barrier_type)
    finally:
        shmem.barrier()
        del shmem
        gc.collect()
