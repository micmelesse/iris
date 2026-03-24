# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Device-side barrier using atomic operations on the symmetric heap.
CUDA graph capturable. Used by Iris.device_barrier().
"""

import triton
import triton.language as tl


@triton.jit
def _translate_ptr(ptr, from_rank, to_rank, heap_bases):
    """Translate a pointer from one rank's address space to another's."""
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    offset = tl.cast(ptr, tl.uint64) - from_base
    translated_ptr = tl.cast(tl.cast(to_base, tl.pointer_type(tl.int8)) + offset, ptr.dtype)
    return translated_ptr


@triton.jit
def _device_barrier_kernel(
    flags_ptr,
    iris_rank,
    world_size: tl.constexpr,
    rank_start,
    rank_stride,
    heap_bases,
    MAX_SPINS: tl.constexpr = 1_000_000_000,
):
    """
    Device-side barrier using atomic operations on the symmetric heap.
    CUDA graph capturable. Single CTA increments own flag then polls remotes.
    """
    own_flag_ptr = flags_ptr + iris_rank
    own_translated = _translate_ptr(own_flag_ptr, iris_rank, iris_rank, heap_bases)
    old = tl.atomic_add(own_translated, 1, sem="release", scope="sys")
    target = old + 1

    for i in range(world_size):
        remote_rank = rank_start + i * rank_stride
        if remote_rank != iris_rank:
            remote_flag_ptr = flags_ptr + remote_rank
            remote_translated = _translate_ptr(remote_flag_ptr, iris_rank, remote_rank, heap_bases)
            spin_count = 0
            while (
                tl.atomic_cas(
                    remote_translated,
                    target,
                    target,
                    sem="acquire",
                    scope="sys",
                )
                < target
            ):
                spin_count += 1
                tl.device_assert(spin_count < MAX_SPINS, "device_barrier: timeout")


def device_barrier(flags, rank_global, world_size, rank_start, rank_stride, heap_bases):
    """
    Device-side barrier using atomic operations on the symmetric heap.
    CUDA graph capturable.

    Unlike host-side ``torch.distributed.barrier()``, this launches a
    single-CTA Triton kernel that synchronizes via device-side atomics,
    making it safe to use during CUDA graph capture.

    Stateless w.r.t. host-side epoch tracking: each rank's flag on the
    symmetric heap serves as its own epoch counter, managed entirely by
    the GPU via atomic_add. A persistent per-group flags tensor is cached
    in ``_device_barrier_state``.

    Args:
        flags: int32 tensor on symmetric heap, one element per rank.
        rank_global: Global rank of this process.
        world_size: Number of ranks in the group.
        rank_start: Starting global rank of the group.
        rank_stride: Stride between consecutive ranks in the group.
        heap_bases: Tensor of heap base addresses for all ranks.
    """
    _device_barrier_kernel[(1,)](
        flags,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        heap_bases,
    )
