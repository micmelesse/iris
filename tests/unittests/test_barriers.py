# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import gc
import pytest
import torch
import iris


BARRIER_TYPES = ["host", "device"]


def _call_barrier(shmem, barrier_type):
    if barrier_type == "host":
        shmem.barrier()
    else:
        shmem.device_barrier()


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


@pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
def test_barrier_synchronizes_data(barrier_type):
    shmem = iris.iris(1 << 20)
    rank = shmem.get_rank()

    buf = shmem.zeros((256,), dtype=torch.float32)

    try:
        _call_barrier(shmem, barrier_type)

        if rank == 0:
            buf.fill_(42.0)

        _call_barrier(shmem, barrier_type)

        if rank == 0:
            expected = torch.full((256,), 42.0, dtype=torch.float32, device="cuda")
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

        # Destroy the NCCL process group before graph capture. On ROCm
        # the watchdog calls hipEventQuery which crashes with
        # hipErrorStreamCaptureUnsupported when any stream is capturing.
        destroy_pg()

        if barrier_type == "host":
            with pytest.raises(Exception):
                with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                    _call_barrier(shmem, barrier_type)
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _call_barrier(shmem, barrier_type)
            for _ in range(3):
                graph.replay()
                stream.synchronize()
    finally:
        recreate_pg()
        shmem.barrier()
        del shmem
        gc.collect()
