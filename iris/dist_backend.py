# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Distributed backend for iris.

This is the ONLY module in iris that imports torch.distributed.
Contains the DistBackend protocol and NCCLBackend/GlooBackend implementations.
"""

from typing import Any, List, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist


@runtime_checkable
class DistBackend(Protocol):
    """Interface for all external collective operations."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def is_initialized(self) -> bool: ...

    def extract_group_info(self, group) -> Tuple[int, int, int, int, int]: ...

    def get_process_group_ranks(self, group) -> List[int]: ...

    def allgather(self, data: np.ndarray) -> np.ndarray: ...

    def allgather_multidim(self, data: np.ndarray) -> np.ndarray: ...

    def allgather_strings(self, local_string: str) -> List[str]: ...

    def broadcast_scalar(self, value: Any, root: int) -> Any: ...

    def broadcast_tensor(self, value: Any, root: int) -> np.ndarray: ...

    def barrier(self) -> None: ...

    def all_gather_cuda(self, tensor_list: list, tensor: torch.Tensor) -> None: ...

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
                f"Rank {self.rank} is not part of the specified process group. "
                f"Group contains ranks: {group_ranks}"
            )

        rank_in_group = group_ranks.index(self.rank)

        if len(group_ranks) > 1:
            strides = [group_ranks[i] - group_ranks[i - 1] for i in range(1, len(group_ranks))]
            if not all(s == strides[0] for s in strides):
                raise NotImplementedError(
                    f"Non-strided process groups are not yet supported. "
                    f"Group ranks: {group_ranks}."
                )
            rank_start = group_ranks[0]
            rank_stride = strides[0]
            if rank_stride == 0:
                raise ValueError(
                    f"Invalid process group: rank_stride is 0. Group ranks: {group_ranks}."
                )
        else:
            rank_start = group_ranks[0]
            rank_stride = 1

        return rank_in_group, self.rank, world_size, rank_start, rank_stride

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

    def allgather_strings(self, local_string: str) -> List[str]:
        local_bytes = local_string.encode("utf-8")
        local_len = len(local_bytes)

        device = torch.device("cuda", torch.cuda.current_device())
        len_tensor = torch.tensor([local_len], dtype=torch.long, device=device)
        len_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(self.world_size)]
        dist.all_gather(len_list, len_tensor, group=self._group)
        max_len = max(t.item() for t in len_list)

        if max_len == 0:
            return [""] * self.world_size

        padded = bytearray(local_bytes) + bytearray(max_len - local_len)
        local_tensor = torch.frombuffer(padded, dtype=torch.uint8).to(device, copy=True)
        gathered = [torch.zeros(max_len, dtype=torch.uint8, device=device) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor, group=self._group)

        results = []
        for t, length_t in zip(gathered, len_list):
            length = int(length_t.item())
            raw = bytes(t[:length].cpu().numpy())
            results.append(raw.decode("utf-8"))
        return results

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

    def all_gather_cuda(self, tensor_list: list, tensor: torch.Tensor) -> None:
        dist.all_gather(tensor_list, tensor, group=self._group)

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
                f"Rank {self.rank} is not part of the specified process group. "
                f"Group contains ranks: {group_ranks}"
            )

        rank_in_group = group_ranks.index(self.rank)

        if len(group_ranks) > 1:
            strides = [group_ranks[i] - group_ranks[i - 1] for i in range(1, len(group_ranks))]
            if not all(s == strides[0] for s in strides):
                raise NotImplementedError(
                    f"Non-strided process groups are not yet supported. "
                    f"Group ranks: {group_ranks}."
                )
            rank_start = group_ranks[0]
            rank_stride = strides[0]
            if rank_stride == 0:
                raise ValueError(
                    f"Invalid process group: rank_stride is 0. Group ranks: {group_ranks}."
                )
        else:
            rank_start = group_ranks[0]
            rank_stride = 1

        return rank_in_group, self.rank, world_size, rank_start, rank_stride

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

    def allgather_strings(self, local_string: str) -> List[str]:
        obj_list = [None for _ in range(self.world_size)]
        dist.all_gather_object(obj_list, local_string, group=self._group)
        return obj_list

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

    def all_gather_cuda(self, tensor_list: list, tensor: torch.Tensor) -> None:
        cpu_list = [t.cpu() for t in tensor_list]
        cpu_tensor = tensor.cpu()
        dist.all_gather(cpu_list, cpu_tensor, group=self._group)
        for i, t in enumerate(cpu_list):
            tensor_list[i].copy_(t.cuda())

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        cpu_tensor = tensor.cpu()
        dist.send(cpu_tensor, dst=dst, group=self._group)

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        cpu_tensor = torch.empty_like(tensor, device="cpu")
        dist.recv(cpu_tensor, src=src, group=self._group)
        tensor.copy_(cpu_tensor.cuda())
