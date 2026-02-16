# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Callable, Generator, Optional

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


@dataclass
class _ProcessGroupState:
    rank: int
    world_size: int
    backend: str
    local_rank: int


_process_group_state: Optional[_ProcessGroupState] = None


def _save_and_destroy_pg() -> _ProcessGroupState:
    """Save process group config, synchronize, and destroy it."""
    state = _ProcessGroupState(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        backend=dist.get_backend(),
        local_rank=torch.cuda.current_device(),
    )
    torch.cuda.synchronize()
    dist.destroy_process_group()
    c10d._world.group_count = 0
    return state


def _restore_pg(state: _ProcessGroupState) -> None:
    """Flush GPU errors and recreate the process group."""
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    if not dist.is_initialized():
        dist.init_process_group(
            backend=state.backend,
            rank=state.rank,
            world_size=state.world_size,
            device_id=torch.device(f"cuda:{state.local_rank}"),
        )


@pytest.fixture
def destroy_pg() -> Generator[Callable[[], None], None, None]:
    """Destroy the NCCL process group to stop the watchdog thread.

    On ROCm the NCCL watchdog calls hipEventQuery which crashes with
    hipErrorStreamCaptureUnsupported when any stream is in capture mode.
    Call this before graph capture and ``recreate_pg`` after.
    """

    def _destroy() -> None:
        global _process_group_state
        _process_group_state = _save_and_destroy_pg()

    yield _destroy

    # Safety net: restore PG if test forgot or failed to call recreate_pg.
    global _process_group_state
    if _process_group_state is not None:
        _restore_pg(_process_group_state)
        _process_group_state = None


@pytest.fixture
def recreate_pg() -> Callable[[], None]:
    """Recreate the NCCL process group after graph capture."""

    def _recreate() -> None:
        global _process_group_state
        if _process_group_state is not None:
            _restore_pg(_process_group_state)
            _process_group_state = None

    return _recreate
