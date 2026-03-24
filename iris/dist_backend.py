# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Distributed backend interface for collective operations.

Iris needs collective operations (allgather, broadcast, barrier) for
initialization and runtime synchronization. This module abstracts
the backend so iris internals never touch torch.distributed directly.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist


@runtime_checkable
class DistBackend(Protocol):
    """Interface for all external collective operations.

    Used for both init-time coordination (heap setup, fd passing)
    and runtime synchronization (CCL barriers after all_reduce, etc).
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def allgather(self, data: np.ndarray) -> np.ndarray:
        """All-gather a 1D numpy array. Returns (world_size, len(data))."""
        ...

    def allgather_multidim(self, data: np.ndarray) -> np.ndarray:
        """All-gather a multi-dim array. Returns (world_size, -1)."""
        ...

    def broadcast_scalar(self, value: Any, root: int) -> Any:
        """Broadcast a scalar from root to all ranks."""
        ...

    def broadcast_tensor(self, value: Any, root: int) -> np.ndarray:
        """Broadcast a tensor/array from root to all ranks."""
        ...

    def barrier(self) -> None:
        """Synchronize all ranks (init-time, host-side)."""
        ...

    def ccl_barrier(self, shmem: Any) -> None:
        """Synchronize all ranks during runtime CCL operations.

        Called after all_reduce, all_gather, reduce_scatter, etc.
        Implementations choose between host-side (shmem.barrier)
        or device-side (shmem.device_barrier) synchronization.
        """
        ...


class NCCLBackend:
    """DistBackend using the default NCCL process group.

    Uses host-side NCCL barriers for both init and runtime.
    """

    def __init__(self) -> None:
        self._group = dist.group.WORLD

    @property
    def rank(self) -> int:
        return dist.get_rank(self._group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self._group)

    def allgather(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        device = torch.device("cuda", torch.cuda.current_device())
        data_tensor = torch.from_numpy(data).to(device)
        gathered = [torch.empty_like(data_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, data_tensor, group=self._group)
        return torch.stack(gathered, dim=0).cpu().numpy()

    def allgather_multidim(self, data: np.ndarray) -> np.ndarray:
        device = torch.device("cuda", torch.cuda.current_device())
        input_tensor = torch.as_tensor(data).to(device)
        tensor_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, input_tensor, group=self._group)
        stacked = torch.stack(tensor_list, dim=0).view(self.world_size, -1)
        return stacked.cpu().numpy()

    def broadcast_scalar(self, value: Any, root: int) -> Any:
        obj = [value if self.rank == root else None]
        dist.broadcast_object_list(obj, src=root, group=self._group)
        return obj[0]

    def broadcast_tensor(self, value: Any, root: int) -> np.ndarray:
        if self.rank == root:
            tensor = torch.as_tensor(value)
            metadata = [tensor.shape, tensor.dtype]
        else:
            metadata = [None, None]
            tensor = None

        dist.broadcast_object_list(metadata, src=root, group=self._group)
        shape, dtype = metadata

        if self.rank != root:
            tensor = torch.empty(shape, dtype=dtype)

        device = torch.device("cuda", torch.cuda.current_device())
        tensor = tensor.to(device)
        dist.broadcast(tensor, src=root, group=self._group)
        return tensor.cpu().numpy()

    def barrier(self) -> None:
        dist.barrier(group=self._group)

    def ccl_barrier(self, shmem: Any) -> None:
        shmem.barrier()


class GlooBackend:
    """DistBackend using a gloo process group.

    Avoids NCCL watchdog thread which crashes during CUDA graph
    capture on ROCm (hipErrorStreamCaptureUnsupported).
    Init ops run on CPU. Runtime CCL uses GPU-side device barriers.
    """

    def __init__(self) -> None:
        self._group = dist.new_group(backend="gloo")

    @property
    def rank(self) -> int:
        return dist.get_rank(self._group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self._group)

    def allgather(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        data_tensor = torch.from_numpy(data)
        if data_tensor.dtype == torch.uint64:
            obj_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(obj_list, data, group=self._group)
            return np.stack(obj_list, axis=0)
        gathered = [torch.empty_like(data_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, data_tensor, group=self._group)
        return torch.stack(gathered, dim=0).numpy()

    def allgather_multidim(self, data: np.ndarray) -> np.ndarray:
        input_tensor = torch.as_tensor(data)
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()
        tensor_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, input_tensor, group=self._group)
        stacked = torch.stack(tensor_list, dim=0).view(self.world_size, -1)
        return stacked.numpy()

    def broadcast_scalar(self, value: Any, root: int) -> Any:
        obj = [value if self.rank == root else None]
        dist.broadcast_object_list(obj, src=root, group=self._group)
        return obj[0]

    def broadcast_tensor(self, value: Any, root: int) -> np.ndarray:
        if self.rank == root:
            tensor = torch.as_tensor(value)
            metadata = [tensor.shape, tensor.dtype]
        else:
            metadata = [None, None]
            tensor = None

        dist.broadcast_object_list(metadata, src=root, group=self._group)
        shape, dtype = metadata

        if self.rank != root:
            tensor = torch.empty(shape, dtype=dtype)

        if tensor.is_cuda:
            tensor = tensor.cpu()

        dist.broadcast(tensor, src=root, group=self._group)
        return tensor.numpy()

    def barrier(self) -> None:
        dist.barrier(group=self._group)

    def ccl_barrier(self, shmem: Any) -> None:
        shmem.device_barrier()
