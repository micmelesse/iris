# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Distributed backend for iris.

This is the ONLY module in iris that imports torch.distributed.
Contains the DistBackend protocol and NCCLBackend/GlooBackend implementations.

Backends are thin wrappers around torch.distributed primitives.
They handle device placement (GPU for NCCL, CPU for Gloo) transparently.
Device-side barriers are owned by Iris, not by the backends.
"""

from typing import List, Protocol, Tuple, runtime_checkable

import torch
import torch.distributed as dist


@runtime_checkable
class DistBackend(Protocol):
    """Interface for all external collective operations.

    Methods are thin wrappers around torch.distributed primitives.
    Backends handle device placement (GPU for NCCL, CPU for Gloo)
    transparently. Callers pass tensors and get tensors back.
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def is_initialized(self) -> bool: ...

    def extract_group_info(self, group) -> Tuple[int, int, int, int, int]: ...

    def get_process_group_ranks(self, group) -> List[int]: ...

    def all_gather(self, tensor_list: list, tensor: torch.Tensor) -> None: ...

    def broadcast(self, tensor: torch.Tensor, src: int) -> None: ...

    def broadcast_object_list(self, obj_list: list, src: int) -> None: ...

    def barrier(self, stream=None) -> None: ...

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

    def all_gather(self, tensor_list: list, tensor: torch.Tensor) -> None:
        dist.all_gather(tensor_list, tensor, group=self._group)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        dist.broadcast(tensor, src=src, group=self._group)

    def broadcast_object_list(self, obj_list: list, src: int) -> None:
        dist.broadcast_object_list(obj_list, src=src, group=self._group)

    def barrier(self, stream=None) -> None:
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
    """DistBackend using a gloo process group.

    Avoids NCCL watchdog thread which crashes during CUDA graph
    capture on ROCm (hipErrorStreamCaptureUnsupported).
    Init ops run on CPU. Runtime CCL uses GPU-side device barriers
    (dispatched by Iris.barrier()).
    """

    def __init__(self) -> None:
        if not self.is_initialized():
            raise RuntimeError("PyTorch distributed is not initialized. Call dist.init_process_group() first.")
        self._group = dist.new_group(backend="gloo")

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

    def all_gather(self, tensor_list: list, tensor: torch.Tensor) -> None:
        cpu_list = [t.cpu() for t in tensor_list]
        cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor
        dist.all_gather(cpu_list, cpu_tensor, group=self._group)
        for i, t in enumerate(cpu_list):
            if tensor_list[i].is_cuda:
                tensor_list[i].copy_(t.cuda())
            else:
                tensor_list[i].copy_(t)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if tensor.is_cuda:
            cpu_tensor = tensor.cpu()
            dist.broadcast(cpu_tensor, src=src, group=self._group)
            tensor.copy_(cpu_tensor.cuda())
        else:
            dist.broadcast(tensor, src=src, group=self._group)

    def broadcast_object_list(self, obj_list: list, src: int) -> None:
        dist.broadcast_object_list(obj_list, src=src, group=self._group)

    def barrier(self, stream=None) -> None:
        dist.barrier(group=self._group)

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        cpu_tensor = tensor.cpu()
        dist.send(cpu_tensor, dst=dst, group=self._group)

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        cpu_tensor = torch.empty_like(tensor, device="cpu")
        dist.recv(cpu_tensor, src=src, group=self._group)
        tensor.copy_(cpu_tensor.cuda())
