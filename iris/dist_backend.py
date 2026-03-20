# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Distributed backend for iris.

This is the ONLY module in iris that imports torch.distributed.
Contains the DistBackend protocol and NCCLBackend/GlooBackend implementations.

Backends are thin wrappers around torch.distributed primitives.
They handle device placement (GPU for NCCL, CPU for Gloo) transparently.
Device-side barriers are owned by Iris, not by the backends.

GlooBackend creates a fully isolated ProcessGroupGloo with its own
TCPStore so that iris collective operations never interfere with the
caller's torch.distributed groups (e.g. vLLM).
"""

import logging
import os
import pickle
from datetime import timedelta
from typing import Any, List, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist

_logger = logging.getLogger(__name__)


@runtime_checkable
class DistBackend(Protocol):
    """Interface for all external collective operations.

    Methods are thin wrappers around torch.distributed primitives.
    Backends handle device placement (GPU for NCCL, CPU for Gloo)
    transparently. Callers pass data and get data back.

    ``broadcast`` and ``all_gather`` are type-dispatching:
    pass a tensor for tensor collectives, or any Python object
    for object collectives. The backend picks the right path.
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def is_initialized(self) -> bool: ...

    def extract_group_info(self, group) -> Tuple[int, int, int, int, int]: ...

    def get_process_group_ranks(self, group) -> List[int]: ...

    def all_gather(self, data: Union[torch.Tensor, Any]) -> list: ...

    def broadcast(self, data: Any, src: int) -> Any: ...

    def host_barrier(self, stream=None) -> None: ...

    def send(self, tensor: torch.Tensor, dst: int) -> None: ...

    def recv(self, tensor: torch.Tensor, src: int) -> None: ...


class NCCLBackend:
    """DistBackend using the default NCCL process group."""

    def __init__(self) -> None:
        if not self.is_initialized():
            raise RuntimeError("PyTorch distributed is not initialized. Call dist.init_process_group() first.")
        self._group = dist.group.WORLD

    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def rank(self) -> int:
        return dist.get_rank(self._group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self._group)

    def get_process_group_ranks(self, group) -> List[int]:
        return dist.get_process_group_ranks(group)

    def extract_group_info(self, group) -> Tuple[int, int, int, int, int]:
        if group is None:
            return self.rank, self.rank, self.world_size, 0, 1

        group_ranks = dist.get_process_group_ranks(group)
        world_size = len(group_ranks)

        if self.rank not in group_ranks:
            raise RuntimeError(
                f"Rank {self.rank} is not part of the specified process group. Group contains ranks: {group_ranks}"
            )

        rank_in_group = group_ranks.index(self.rank)

        if len(group_ranks) > 1:
            strides = [group_ranks[i] - group_ranks[i - 1] for i in range(1, len(group_ranks))]
            if not all(s == strides[0] for s in strides):
                raise NotImplementedError(
                    f"Non-strided process groups are not yet supported. Group ranks: {group_ranks}."
                )
            rank_start = group_ranks[0]
            rank_stride = strides[0]
            if rank_stride == 0:
                raise ValueError(f"Invalid process group: rank_stride is 0. Group ranks: {group_ranks}.")
        else:
            rank_start = group_ranks[0]
            rank_stride = 1

        return rank_in_group, self.rank, world_size, rank_start, rank_stride

    def all_gather(self, data: Union[torch.Tensor, Any]) -> list:
        if isinstance(data, torch.Tensor):
            gathered = [torch.empty_like(data) for _ in range(self.world_size)]
            dist.all_gather(gathered, data, group=self._group)
            return gathered
        obj_list = [None] * self.world_size
        dist.all_gather_object(obj_list, data, group=self._group)
        return obj_list

    def broadcast(self, data: Any, src: int) -> Any:
        # Tensor: all ranks have it allocated, broadcast in-place
        if isinstance(data, torch.Tensor):
            dist.broadcast(data, src=src, group=self._group)
            return data
        # ndarray on src, None on others: broadcast metadata then tensor data
        is_array = isinstance(data, np.ndarray)
        if is_array or (data is None and self.rank != src):
            if is_array:
                tensor = torch.from_numpy(data).cuda()
                meta = [tensor.shape, tensor.dtype]
            else:
                meta = [None, None]
            dist.broadcast_object_list(meta, src=src, group=self._group)
            if meta[0] is not None:
                if not is_array:
                    tensor = torch.empty(meta[0], dtype=meta[1], device="cuda")
                dist.broadcast(tensor, src=src, group=self._group)
                return tensor.cpu().numpy()
        # Scalar/object: broadcast via pickle
        obj = [data]
        dist.broadcast_object_list(obj, src=src, group=self._group)
        return obj[0]

    def host_barrier(self, stream=None) -> None:
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()
        dist.barrier(group=self._group)

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor, dst=dst, group=self._group)

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        dist.recv(tensor, src=src, group=self._group)


class GlooBackend:
    """DistBackend using an isolated gloo process group.

    Creates its own TCPStore on a separate port (MASTER_PORT + offset)
    so that iris gloo operations are fully isolated from the caller's
    torch.distributed groups (e.g. vLLM).

    Avoids NCCL watchdog thread which crashes during CUDA graph
    capture on ROCm (hipErrorStreamCaptureUnsupported).
    Init ops run on CPU. Runtime CCL uses GPU-side device barriers
    (dispatched by Iris.barrier()).
    """

    def __init__(self) -> None:
        if not self.is_initialized():
            raise RuntimeError("PyTorch distributed is not initialized. Call dist.init_process_group() first.")

        # Read rank/world_size from existing torch.distributed
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._op_counter = 0

        # Create isolated TCPStore on a separate port
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        base_port = int(os.environ.get("MASTER_PORT", "29500"))
        port_offset = int(os.environ.get("IRIS_GLOO_PORT_OFFSET", "100"))
        iris_port = base_port + port_offset

        try:
            self._store = dist.TCPStore(
                host_name=master_addr,
                port=iris_port,
                world_size=self._world_size,
                is_master=(self._rank == 0),
                timeout=timedelta(seconds=30),
            )
        except Exception as e:
            _logger.error("rank=%d: TCPStore creation failed on port %d: %s", self._rank, iris_port, e)
            raise

        try:
            try:
                from torch.distributed.distributed_c10d import ProcessGroupGloo
            except ImportError:
                from torch._C._distributed_c10d import ProcessGroupGloo

            self._group = ProcessGroupGloo(self._store, self._rank, self._world_size)
        except Exception as e:
            _logger.error("rank=%d: ProcessGroupGloo creation failed: %s", self._rank, e)
            raise

    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def get_process_group_ranks(self, group) -> List[int]:
        return dist.get_process_group_ranks(group)

    def extract_group_info(self, group) -> Tuple[int, int, int, int, int]:
        if group is None:
            return self._rank, self._rank, self._world_size, 0, 1

        group_ranks = dist.get_process_group_ranks(group)
        world_size = len(group_ranks)

        if self._rank not in group_ranks:
            raise RuntimeError(
                f"Rank {self._rank} is not part of the specified process group. Group contains ranks: {group_ranks}"
            )

        rank_in_group = group_ranks.index(self._rank)

        if len(group_ranks) > 1:
            strides = [group_ranks[i] - group_ranks[i - 1] for i in range(1, len(group_ranks))]
            if not all(s == strides[0] for s in strides):
                raise NotImplementedError(
                    f"Non-strided process groups are not yet supported. Group ranks: {group_ranks}."
                )
            rank_start = group_ranks[0]
            rank_stride = strides[0]
            if rank_stride == 0:
                raise ValueError(f"Invalid process group: rank_stride is 0. Group ranks: {group_ranks}.")
        else:
            rank_start = group_ranks[0]
            rank_stride = 1

        return rank_in_group, self._rank, world_size, rank_start, rank_stride

    # -- collective operations (use PG methods directly, not dist.*) ----------

    def all_gather(self, data: Union[torch.Tensor, Any]) -> list:
        if isinstance(data, torch.Tensor):
            cpu_tensor = data.cpu() if data.is_cuda else data
            gathered = [torch.empty_like(cpu_tensor) for _ in range(self._world_size)]
            try:
                self._group.allgather([gathered], [cpu_tensor]).wait()
            except Exception as e:
                _logger.error("rank=%d: allgather failed: %s", self._rank, e)
                raise
            if data.is_cuda:
                return [t.cuda() for t in gathered]
            return gathered
        # Object collective via store (avoids dist.all_gather_object
        # which requires the PG to be registered in torch.distributed)
        return self._all_gather_object(data)

    def _all_gather_object(self, obj: Any) -> list:
        tag = self._op_counter
        self._op_counter += 1
        key = f"iris_ag_{tag}_{self._rank}"
        self._store.set(key, pickle.dumps(obj))
        self._group.barrier().wait()
        result = []
        for i in range(self._world_size):
            result.append(pickle.loads(self._store.get(f"iris_ag_{tag}_{i}")))
        self._group.barrier().wait()
        return result

    def broadcast(self, data: Any, src: int) -> Any:
        # Tensor: all ranks have it allocated, broadcast in-place via CPU
        if isinstance(data, torch.Tensor):
            try:
                from torch._C._distributed_c10d import BroadcastOptions
            except ImportError:
                from torch.distributed.distributed_c10d import BroadcastOptions
            opts = BroadcastOptions()
            opts.rootRank = src

            if data.is_cuda:
                cpu_tensor = data.cpu()
                try:
                    self._group.broadcast([cpu_tensor], opts).wait()
                except Exception as e:
                    _logger.error("rank=%d: broadcast failed: %s", self._rank, e)
                    raise
                data.copy_(cpu_tensor.cuda())
            else:
                try:
                    self._group.broadcast([data], opts).wait()
                except Exception as e:
                    _logger.error("rank=%d: broadcast failed: %s", self._rank, e)
                    raise
            return data
        # Object/ndarray/scalar: serialize via store
        tag = self._op_counter
        self._op_counter += 1
        if self._rank == src:
            self._store.set(f"iris_bc_{tag}", pickle.dumps(data))
        self._group.barrier().wait()
        result = pickle.loads(self._store.get(f"iris_bc_{tag}"))
        self._group.barrier().wait()
        return result

    def host_barrier(self, stream=None) -> None:
        try:
            self._group.barrier().wait()
        except Exception as e:
            _logger.error("rank=%d: barrier failed: %s", self._rank, e)
            raise

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        cpu_tensor = tensor.cpu()
        try:
            self._group.send([cpu_tensor], dst, 0).wait()
        except Exception as e:
            _logger.error("rank=%d: send(dst=%d) failed: %s", self._rank, dst, e)
            raise

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        cpu_tensor = torch.empty_like(tensor, device="cpu")
        try:
            self._group.recv([cpu_tensor], src, 0).wait()
        except Exception as e:
            _logger.error("rank=%d: recv(src=%d) failed: %s", self._rank, src, e)
            raise
        tensor.copy_(cpu_tensor.cuda())
