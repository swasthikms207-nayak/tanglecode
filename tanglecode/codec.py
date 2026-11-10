"""Rank-only codec for the .tngl binary format: topological braid compression."""

import struct
from typing import Sequence, Union

import numpy as np

from tanglecode.braid import extract_braid_word, rank_vector

MAGIC = b"TNGL"
HEADER_STRUCT = "<4sBIIB"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 14 bytes
MODE_RANK_ONLY = 0


def _pack_nibble(g: int) -> int:
    """Pack a signed Artin generator into a 4-bit nibble.

    Bit 3 (0x8): sign bit (0 = positive, 1 = negative).
    Bits 0-2 (0x7): generator index (1 to 7).
    """
    sign_bit = 1 if g < 0 else 0
    idx = abs(g) & 0x07
    return (sign_bit << 3) | idx


def _unpack_nibble(nibble: int) -> int:
    """Unpack a 4-bit nibble into a signed Artin generator."""
    sign_bit = (nibble >> 3) & 0x01
    idx = nibble & 0x07
    return -idx if sign_bit else idx


def _pack_generators(generators: Sequence[int]) -> bytes:
    """Pack a sequence of generators 2-per-byte as 4-bit nibbles."""
    byte_list = bytearray()
    n_gens = len(generators)
    for i in range(0, n_gens, 2):
        high = _pack_nibble(generators[i])
        low = _pack_nibble(generators[i + 1]) if i + 1 < n_gens else 0
        byte_list.append((high << 4) | (low & 0x0F))
    return bytes(byte_list)


def _unpack_generators(payload: bytes, count: int) -> list[int]:
    """Unpack count generators from 4-bit nibble bytes."""
    generators = []
    for b in payload:
        high = (b >> 4) & 0x0F
        generators.append(_unpack_nibble(high))
        if len(generators) == count:
            break
        low = b & 0x0F
        generators.append(_unpack_nibble(low))
        if len(generators) == count:
            break
    return generators


def _rle_encode(counts: Sequence[int]) -> bytes:
    """Run-length encode per-tick generator counts."""
    num_ticks = len(counts)
    if num_ticks == 0:
        return struct.pack("<II", 0, 0)

    runs: list[tuple[int, int]] = []
    current_val = counts[0]
    current_len = 1

    for val in counts[1:]:
        if val == current_val:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val = val
            current_len = 1
    runs.append((current_val, current_len))

    buf = bytearray()
    buf.extend(struct.pack("<II", num_ticks, len(runs)))
    for val, run_len in runs:
        buf.extend(struct.pack("<HI", int(val), int(run_len)))
    return bytes(buf)


def _rle_decode(data: bytes, offset: int) -> tuple[list[int], int]:
    """Decode run-length encoded per-tick counts."""
    num_ticks, num_runs = struct.unpack_from("<II", data, offset)
    offset += 8

    counts: list[int] = []
    for _ in range(num_runs):
        val, run_len = struct.unpack_from("<HI", data, offset)
        offset += 6
        counts.extend([val] * run_len)

    if len(counts) != num_ticks:
        raise ValueError(f"RLE decoded {len(counts)} ticks, expected {num_ticks}")

    return counts, offset


def _reduce_tagged_generators(
    raw_gens: Sequence[int],
    raw_ticks: Sequence[int],
) -> list[tuple[int, int]]:
    """Reduce generators across ticks with commutativity-aware stack cancellation."""
    stack: list[tuple[int, int]] = []

    for g, t in zip(raw_gens, raw_ticks):
        cancelled = False
        for i in range(len(stack) - 1, -1, -1):
            prev_g, prev_t = stack[i]
            if prev_g == -g:
                stack.pop(i)
                cancelled = True
                break
            elif abs(abs(g) - abs(prev_g)) < 2:
                break

        if not cancelled:
            stack.append((g, t))

    return stack


def encode(
    data: Union[np.ndarray, Sequence[Sequence[float]]],
    mode: int = MODE_RANK_ONLY,
) -> bytes:
    """Encode multichannel time series data into rank-only .tngl binary format.

    Format:
    - Header (14 bytes): magic "TNGL", n (uint8), raw_generator_count (uint32),
      reduced_generator_count (uint32), mode (uint8).
    - Anchor snapshot: first frame as float32 per channel.
    - Timing: RLE encoded per-tick generator counts.
    - Payload: reduced generators packed 2-per-byte as 4-bit nibbles.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_channels)
        Multichannel sensor data (n_channels up to 8).
    mode : int
        Format mode (0 = MODE_RANK_ONLY).

    Returns
    -------
    bytes
        Encoded binary .tngl data.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"data must be 2D array, got shape {arr.shape}")

    n_samples, n_channels = arr.shape
    if not (1 <= n_channels <= 8):
        raise ValueError(f"tngl format supports 1 to 8 channels, got {n_channels}")
    if n_samples < 1:
        raise ValueError("data must have at least 1 sample")

    num_ticks = n_samples - 1

    # Extract raw generators and tick tags
    raw_generators: list[int] = []
    raw_tick_tags: list[int] = []

    for t in range(num_ticks):
        r_prev = rank_vector(arr[t])
        r_curr = rank_vector(arr[t + 1])
        step_word = extract_braid_word(r_prev, r_curr)
        for g in step_word:
            raw_generators.append(g)
            raw_tick_tags.append(t)

    raw_generator_count = len(raw_generators)

    # Topological reduction with tick tracking
    surviving = _reduce_tagged_generators(raw_generators, raw_tick_tags)
    reduced_generators = [g for g, _ in surviving]
    reduced_generator_count = len(reduced_generators)

    per_tick_counts = [0] * num_ticks
    for _, t in surviving:
        per_tick_counts[t] += 1

    # Build Header
    header = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        n_channels,
        raw_generator_count,
        reduced_generator_count,
        MODE_RANK_ONLY,
    )

    # Anchor snapshot (float32 per channel)
    anchor = struct.pack(f"<{n_channels}f", *arr[0])

    # Timing: RLE encoded per-tick generator counts
    timing = _rle_encode(per_tick_counts)

    # Payload: 4-bit packed generators
    payload = _pack_generators(reduced_generators)

    return header + anchor + timing + payload


def decode(data_bytes: bytes) -> np.ndarray:
    """Decode a rank-only .tngl binary stream into a reconstructed rank-order array.

    Reconstructs the channel rank order per tick. Values correspond to each channel's
    rank position (0.0 to n-1.0).

    Parameters
    ----------
    data_bytes : bytes
        Binary data formatted according to the .tngl specification.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_channels) with reconstructed rank values.
    """
    if len(data_bytes) < HEADER_SIZE:
        raise ValueError(f"Data too short to contain header ({len(data_bytes)} bytes)")

    magic, n_channels, raw_gen_count, reduced_gen_count, mode = struct.unpack_from(
        HEADER_STRUCT, data_bytes, 0
    )
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes {magic!r}, expected {MAGIC!r}")

    offset = HEADER_SIZE

    # Read Anchor snapshot
    anchor_size = n_channels * 4
    if len(data_bytes) < offset + anchor_size:
        raise ValueError("Corrupt file: missing anchor snapshot")
    anchor = np.array(struct.unpack_from(f"<{n_channels}f", data_bytes, offset), dtype=np.float32)
    offset += anchor_size

    # Read Timing (RLE per-tick generator counts)
    per_tick_counts, offset = _rle_decode(data_bytes, offset)
    num_ticks = len(per_tick_counts)
    n_samples = num_ticks + 1

    # Read Payload (Packed generators)
    packed_size = (reduced_gen_count + 1) // 2
    if len(data_bytes) < offset + packed_size:
        raise ValueError("Corrupt file: truncated generator payload")
    gen_bytes = data_bytes[offset : offset + packed_size]
    offset += packed_size
    reduced_generators = _unpack_generators(gen_bytes, reduced_gen_count)

    # Reconstruct rank order per tick
    reconstructed = np.zeros((n_samples, n_channels), dtype=np.float32)
    current_order = rank_vector(anchor)
    for c in range(n_channels):
        reconstructed[0, c] = float(current_order.index(c))

    gen_ptr = 0
    for t in range(num_ticks):
        cnt = per_tick_counts[t]
        tick_gens = reduced_generators[gen_ptr : gen_ptr + cnt]
        gen_ptr += cnt

        for g in tick_gens:
            swap_idx = abs(g) - 1
            current_order[swap_idx], current_order[swap_idx + 1] = (
                current_order[swap_idx + 1],
                current_order[swap_idx],
            )

        for c in range(n_channels):
            reconstructed[t + 1, c] = float(current_order.index(c))

    return reconstructed
