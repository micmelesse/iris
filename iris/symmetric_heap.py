# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Symmetric heap abstraction for Iris.

Provides a high-level interface for distributed symmetric memory management,
hiding the details of allocators and inter-process memory sharing.
"""

import sys
import numpy as np
import torch
import os


def _dbg(msg: str, rank: object = "?") -> None:
    """Debug print to stderr with flush for crash debugging."""
    print(f"[iris heap rank={rank}] {msg}", file=sys.stderr, flush=True)

from iris.allocators import TorchAllocator, VMemAllocator
from iris.fd_passing import setup_fd_infrastructure
from iris.util import is_simulation_env


class SymmetricHeap:
    """
    High-level symmetric heap abstraction.

    Manages distributed memory with symmetric addressing across ranks,
    handling all allocator coordination and memory sharing internally.

    Supports multiple allocator backends: 'torch' (default) and 'vmem'.
    """

    def __init__(
        self,
        heap_size: int,
        device_id: int,
        cur_rank: int,
        num_ranks: int,
        allocator_type: str = "torch",
        dist_backend=None,
    ):
        """
        Initialize symmetric heap.

        Args:
            heap_size: Size of the heap in bytes
            device_id: GPU device ID
            cur_rank: Current process rank
            num_ranks: Total number of ranks
            allocator_type: Type of allocator ("torch" or "vmem"); default "torch"
            dist_backend: DistBackend for collective ops.

        Raises:
            ValueError: If allocator_type is not supported
        """
        assert heap_size > 0, f"heap_size must be positive, got {heap_size}"
        assert device_id >= 0, f"device_id must be non-negative, got {device_id}"
        assert num_ranks > 0, f"num_ranks must be positive, got {num_ranks}"
        assert 0 <= cur_rank < num_ranks, f"cur_rank={cur_rank} out of range [0, {num_ranks})"
        if num_ranks > 1:
            assert dist_backend is not None, "dist_backend required when num_ranks > 1"

        self.heap_size = heap_size
        self.device_id = device_id
        self.cur_rank = cur_rank
        self.num_ranks = num_ranks
        self._dist = dist_backend
        allocator_type = os.environ.get("IRIS_ALLOCATOR", allocator_type).lower()

        if is_simulation_env():
            allocator_type = "torch"

        _dbg(f"creating allocator type={allocator_type}", cur_rank)
        if allocator_type == "torch":
            self.allocator = TorchAllocator(heap_size, device_id, cur_rank, num_ranks)
        elif allocator_type == "vmem":
            self.allocator = VMemAllocator(heap_size, device_id, cur_rank, num_ranks)
        else:
            raise ValueError(f"Unknown allocator type: {allocator_type}. Supported: 'torch', 'vmem'")
        base_addr = self.allocator.get_base_address()
        assert base_addr != 0, f"Allocator base address is null (rank={cur_rank})"
        _dbg(f"allocator created OK, base={base_addr:#x}", cur_rank)

        _dbg("setup_fd_infrastructure start", cur_rank)
        self.fd_conns = setup_fd_infrastructure(cur_rank, num_ranks, dist_backend=dist_backend)
        if num_ranks > 1:
            assert self.fd_conns is not None, f"fd_conns is None with num_ranks={num_ranks}"
            expected_peers = num_ranks - 1
            assert len(self.fd_conns) == expected_peers, (
                f"fd_conns has {len(self.fd_conns)} entries, expected {expected_peers} (num_ranks-1)"
            )
        _dbg(f"setup_fd_infrastructure done (fd_conns={'None' if self.fd_conns is None else len(self.fd_conns)})", cur_rank)

        device = self.allocator.get_device()

        # Use int64 instead of uint64 for gloo backend compatibility
        # Create from numpy array to avoid kernel issue (torch.zeros on small tensors triggers problematic kernel)
        heap_bases_array = np.zeros(self.num_ranks, dtype=np.int64)
        # Create on CPU first, then move to device to avoid FFM ioctl issue
        if is_simulation_env():
            self.heap_bases = torch.tensor(heap_bases_array, device="cpu", dtype=torch.int64)
            self.heap_bases = self.heap_bases.to(device)
        else:
            self.heap_bases = torch.tensor(heap_bases_array, device=device, dtype=torch.int64)
        _dbg("heap_bases tensor created, calling refresh_peer_access", cur_rank)

        self.refresh_peer_access()
        self.validate()
        _dbg("SymmetricHeap init complete", cur_rank)

    def _assert_not_capturing(self, op: str) -> None:
        """Raise if called during CUDA graph stream capture.

        HIP VMem operations (mem_create, mem_map, mem_set_access,
        mem_import_from_shareable_handle) and host barriers are illegal
        during stream capture and will crash with
        hipErrorStreamCaptureUnsupported. Guard all public methods that
        perform these operations.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"SymmetricHeap.{op}() called during CUDA graph stream capture. "
                f"HIP VMem operations are not allowed during capture. "
                f"Move this call before graph capture begins."
            )

    def allocate(self, num_elements: int, dtype: torch.dtype, alignment: int = 1024) -> torch.Tensor:
        """
        Allocate a tensor on the symmetric heap.

        Always allocates at least the allocator's minimum allocation size so that
        even zero-element requests get a buffer on the heap; for num_elements==0
        we return a zero-length slice of that buffer so the tensor is still on heap.

        Args:
            num_elements: Number of elements to allocate
            dtype: PyTorch data type
            alignment: Alignment requirement in bytes (default: 1024)

        Returns:
            Allocated tensor on the symmetric heap (shape (num_elements,) or (0,) for empty)

        Note:
            This should be called collectively across all ranks to maintain
            symmetric heap consistency. After allocation, peer access is refreshed.
        """
        self._assert_not_capturing("allocate")
        min_bytes = self.allocator.get_minimum_allocation_size()
        element_size = torch.tensor([], dtype=dtype).element_size()
        min_elements = max(1, (min_bytes + element_size - 1) // element_size)
        actual_elements = max(num_elements, min_elements)
        tensor = self.allocator.allocate(actual_elements, dtype, alignment)
        tensor = tensor[:num_elements]
        self.refresh_peer_access()
        return tensor

    def get_device(self) -> torch.device:
        """Get the torch device for this heap."""
        return self.allocator.get_device()

    def on_symmetric_heap(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is allocated on the symmetric heap.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is on the symmetric heap, False otherwise
        """
        return self.allocator.owns_tensor(tensor)

    def is_symmetric(self, tensor: torch.Tensor) -> bool:
        """
        Check if a tensor is allocated on the symmetric heap.

        This method provides a public API to check whether a tensor resides in the
        symmetric heap, making it accessible for RMA operations across ranks.

        Args:
            tensor: PyTorch tensor to check

        Returns:
            True if tensor is on the symmetric heap, False otherwise

        Example:
            >>> ctx = iris.iris(heap_size=2**30)
            >>> symmetric_tensor = ctx.zeros(1000, dtype=torch.float32)
            >>> external_tensor = torch.zeros(1000, dtype=torch.float32, device='cuda')
            >>> ctx.heap.is_symmetric(symmetric_tensor)  # True
            >>> ctx.heap.is_symmetric(external_tensor)   # False
        """
        return self.on_symmetric_heap(tensor)

    def get_heap_bases(self) -> torch.Tensor:
        """Get heap base addresses for all ranks as a tensor."""
        return self.heap_bases

    def validate(self):
        """
        Validate core symmetric heap invariants.

        Call after init or after any operation that modifies heap state.
        Raises AssertionError with a diagnostic message on failure.
        """
        rank = self.cur_rank

        # 1. Allocator base must be non-zero
        base = self.allocator.get_base_address()
        assert base != 0, f"[rank={rank}] Allocator base address is null"

        # 2. Every rank has a non-zero heap base
        bases = [int(self.heap_bases[r]) for r in range(self.num_ranks)]
        for r, b in enumerate(bases):
            assert b != 0, (
                f"[rank={rank}] heap_bases[{r}] is null. All bases: {[hex(x) for x in bases]}"
            )

        # 3. All heap bases are unique (no two ranks mapped to the same VA)
        assert len(set(bases)) == self.num_ranks, (
            f"[rank={rank}] Duplicate heap bases: {[hex(x) for x in bases]}"
        )

        # 4. Local heap_bases entry matches allocator base
        assert bases[rank] == base, (
            f"[rank={rank}] heap_bases[{rank}]={hex(bases[rank])} != allocator base={hex(base)}"
        )

        # 5. Peer VA ranges (if set) don't overlap local heap
        if hasattr(self, "_peer_va_ranges"):
            local_end = base + self.heap_size
            for peer, peer_va in self._peer_va_ranges.items():
                assert not (base <= peer_va < local_end), (
                    f"[rank={rank}] Peer {peer} VA {hex(peer_va)} overlaps "
                    f"local heap [{hex(base)}, {hex(local_end)})"
                )

        # 6. FD connections match expected count (peers only, no self-entry)
        if self.num_ranks > 1 and self.fd_conns is not None:
            expected_peers = self.num_ranks - 1
            assert len(self.fd_conns) == expected_peers, (
                f"[rank={rank}] fd_conns has {len(self.fd_conns)} entries, expected {expected_peers}"
            )

        _dbg(f"validate OK, bases={[hex(b) for b in bases]}", rank)

    def refresh_peer_access(self):
        """
        Refresh peer DMA-BUF imports using segmented export/import.
        Collective: all ranks must call together. Do not cache heap_bases.
        """
        self._assert_not_capturing("refresh_peer_access")
        from iris.fd_passing import send_fd, recv_fd
        from iris.hip import (
            export_dmabuf_handle,
            mem_import_from_shareable_handle,
            mem_map,
            mem_set_access,
            mem_address_reserve,
            hipMemAccessDesc,
            hipMemLocationTypeDevice,
            hipMemAccessFlagsProtReadWrite,
        )

        _dbg("refresh_peer_access: entering", self.cur_rank)

        if self._dist is not None:
            _dbg("refresh_peer_access: host_barrier (pre-gather)", self.cur_rank)
            self._dist.host_barrier()

        my_base = self.allocator.get_base_address()
        assert my_base != 0, f"Allocator base address is null (rank={self.cur_rank})"
        _dbg(f"refresh_peer_access: my_base={my_base:#x}", self.cur_rank)
        # Use int64 instead of uint64 to avoid gloo issues
        local_tensor = torch.tensor([my_base], dtype=torch.int64)
        _dbg("refresh_peer_access: calling all_gather", self.cur_rank)
        gathered = self._dist.all_gather(local_tensor)
        _dbg("refresh_peer_access: all_gather done", self.cur_rank)
        all_bases_arr = torch.stack(gathered).numpy().reshape(self.num_ranks).astype(np.int64)
        for r in range(self.num_ranks):
            assert int(all_bases_arr[r]) != 0, (
                f"Rank {r} has null base address after all_gather (seen by rank={self.cur_rank}). "
                f"All bases: {[hex(int(b)) for b in all_bases_arr]}"
            )
        self.heap_bases[self.cur_rank] = int(all_bases_arr[self.cur_rank])

        if self.num_ranks == 1 or self.fd_conns is None:
            _dbg("refresh_peer_access: single rank or no fd_conns, returning early", self.cur_rank)
            return

        if not hasattr(self.allocator, "get_allocation_segments"):
            if hasattr(self.allocator, "establish_peer_access"):
                # In simulation, all ranks share the same device, so skip peer access setup
                from iris.util import is_simulation_env

                if is_simulation_env():
                    # Just set heap_bases directly from all_bases_arr
                    for r in range(self.num_ranks):
                        self.heap_bases[r] = int(all_bases_arr[r])
                else:
                    all_bases = {r: int(all_bases_arr[r]) for r in range(self.num_ranks)}
                    self.allocator.establish_peer_access(all_bases, self.fd_conns)
                    for r in range(self.num_ranks):
                        self.heap_bases[r] = int(self.allocator.heap_bases_array[r])
            return

        _dbg("refresh_peer_access: getting allocation segments", self.cur_rank)
        my_segments = self.allocator.get_allocation_segments()
        _dbg(f"refresh_peer_access: {len(my_segments)} segments", self.cur_rank)
        my_exported_fds = []
        for offset, size, va in my_segments:
            _dbg(f"refresh_peer_access: export_dmabuf_handle(va={va:#x}, size={size})", self.cur_rank)
            dmabuf_fd, export_base, export_size = export_dmabuf_handle(va, size)
            assert dmabuf_fd >= 0, (
                f"export_dmabuf_handle returned invalid fd={dmabuf_fd} for va={va:#x} size={size} (rank={self.cur_rank})"
            )
            assert export_size > 0, (
                f"export_dmabuf_handle returned zero export_size for va={va:#x} (rank={self.cur_rank})"
            )
            _dbg(f"refresh_peer_access: exported fd={dmabuf_fd}, size={export_size}", self.cur_rank)
            my_exported_fds.append((dmabuf_fd, export_size, offset))

        access_desc = hipMemAccessDesc()
        access_desc.location.type = hipMemLocationTypeDevice
        access_desc.location.id = self.device_id
        access_desc.flags = hipMemAccessFlagsProtReadWrite

        for peer, sock in self.fd_conns.items():
            if peer == self.cur_rank:
                continue

            _dbg(f"refresh_peer_access: processing peer={peer}", self.cur_rank)

            if not hasattr(self, "_peer_va_ranges"):
                self._peer_va_ranges = {}

            if peer not in self._peer_va_ranges:
                _dbg(f"refresh_peer_access: mem_address_reserve for peer={peer}", self.cur_rank)
                peer_va_base = mem_address_reserve(self.heap_size, self.allocator.granularity, 0)
                assert peer_va_base != 0, (
                    f"mem_address_reserve returned null VA for peer={peer} (rank={self.cur_rank})"
                )
                my_end = my_base + self.heap_size
                assert not (my_base <= peer_va_base < my_end), (
                    f"Peer {peer} VA {peer_va_base:#x} overlaps local heap "
                    f"[{my_base:#x}, {my_end:#x}) (rank={self.cur_rank})"
                )
                _dbg(f"refresh_peer_access: peer={peer} va_base={peer_va_base:#x}", self.cur_rank)
                self._peer_va_ranges[peer] = peer_va_base
            else:
                peer_va_base = self._peer_va_ranges[peer]

            peer_fds = []
            for seg_idx, (my_fd, my_size, my_offset) in enumerate(my_exported_fds):
                # Exchange FDs (higher rank sends first to avoid deadlock)
                _dbg(f"refresh_peer_access: FD exchange peer={peer} seg={seg_idx} (rank {'sends' if self.cur_rank > peer else 'recvs'} first)", self.cur_rank)
                if self.cur_rank > peer:
                    send_fd(sock, my_fd)
                    peer_fd, _ = recv_fd(sock)
                else:
                    peer_fd, _ = recv_fd(sock)
                    send_fd(sock, my_fd)
                assert peer_fd >= 0, (
                    f"Received invalid peer_fd={peer_fd} from peer={peer} seg={seg_idx} (rank={self.cur_rank})"
                )
                _dbg(f"refresh_peer_access: FD exchange peer={peer} seg={seg_idx} done, peer_fd={peer_fd}", self.cur_rank)

                peer_fds.append((peer_fd, my_size, my_offset))

            if not hasattr(self, "_peer_cumulative_sizes"):
                self._peer_cumulative_sizes = {}
            cumulative_size = self._peer_cumulative_sizes.get(peer, 0)

            if not hasattr(self, "_peer_imported_segments"):
                self._peer_imported_segments = {}
            if peer not in self._peer_imported_segments:
                self._peer_imported_segments[peer] = set()

            for peer_fd, segment_size, offset in peer_fds:
                segment_key = (offset, segment_size)
                if segment_key in self._peer_imported_segments[peer]:
                    import os

                    os.close(peer_fd)
                    continue

                _dbg(f"refresh_peer_access: mem_import_from_shareable_handle(fd={peer_fd}) peer={peer}", self.cur_rank)
                imported_handle = mem_import_from_shareable_handle(peer_fd)
                assert imported_handle is not None and imported_handle != 0, (
                    f"mem_import_from_shareable_handle returned invalid handle={imported_handle} "
                    f"for peer={peer} fd={peer_fd} (rank={self.cur_rank})"
                )
                _dbg(f"refresh_peer_access: import done, handle={imported_handle}", self.cur_rank)
                import os

                os.close(peer_fd)

                peer_va = peer_va_base + offset
                _dbg(f"refresh_peer_access: mem_map(va={peer_va:#x}, size={segment_size}) peer={peer}", self.cur_rank)
                mem_map(peer_va, segment_size, 0, imported_handle)
                _dbg(f"refresh_peer_access: mem_map done peer={peer}", self.cur_rank)
                self._peer_imported_segments[peer].add(segment_key)

                new_cumulative = offset + segment_size
                if new_cumulative > cumulative_size:
                    cumulative_size = new_cumulative
                    _dbg(f"refresh_peer_access: mem_set_access(va={peer_va_base:#x}, size={cumulative_size}) peer={peer}", self.cur_rank)
                    mem_set_access(peer_va_base, cumulative_size, access_desc)
                    _dbg(f"refresh_peer_access: mem_set_access done peer={peer}", self.cur_rank)

            self._peer_cumulative_sizes[peer] = cumulative_size
            self.heap_bases[peer] = peer_va_base
            _dbg(f"refresh_peer_access: peer={peer} complete", self.cur_rank)

        for fd, _, _ in my_exported_fds:
            import os

            os.close(fd)

        _dbg("refresh_peer_access: all peers done, final barrier", self.cur_rank)
        if self._dist is not None:
            self._dist.host_barrier()
        _dbg("refresh_peer_access: complete", self.cur_rank)

    def as_symmetric(self, external_tensor: torch.Tensor) -> torch.Tensor:
        """
        Place an external PyTorch tensor on the symmetric heap.

        With the torch allocator: allocates on the heap and copies the data;
        the returned tensor is independent of the input. With the vmem
        allocator: imports the memory so both tensors share the same storage.

        Args:
            external_tensor: External PyTorch tensor (must be CUDA, contiguous)

        Returns:
            Tensor on the symmetric heap (same shape/dtype; copy or shared per allocator)

        Raises:
            RuntimeError: If allocator doesn't support imports or import fails
        """
        self._assert_not_capturing("as_symmetric")
        if not hasattr(self.allocator, "import_external_tensor"):
            raise RuntimeError(f"{type(self.allocator).__name__} does not support as_symmetric().")

        imported = self.allocator.import_external_tensor(external_tensor)
        self.refresh_peer_access()
        return imported
