from __future__ import annotations

import struct
from typing import Any, Sequence
import numpy as np

from tanglecode.braid import (
    Crossing,
    order_from_frame,
    reduce_crossings,
    word_from_frames,
)

MAGIC = b"TNGL"
HEADER_STRUCT = "<4sBIIIB"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)  # 18 bytes
MODE_DEFAULT = 0


class DecodedCrossing(tuple):
    """Represents a decoded crossing as a (tick, gen) tuple with attribute access."""

    def __new__(cls, tick: int, gen: int):
        return super().__new__(cls, (int(tick), int(gen)))

    @property
    def tick(self) -> int:
        return self[0]

    @property
    def gen(self) -> int:
        return self[1]

    def __repr__(self) -> str:
        return f"({self[0]}, {self[1]})"


def encode_leb128(val: int) -> bytes:
    """Encode an unsigned integer into LEB128 varint bytes."""
    if val < 0:
        raise ValueError("val must be non-negative for unsigned LEB128")
    out = bytearray()
    while True:
        byte = val & 0x7F
        val >>= 7
        if val != 0:
            byte |= 0x80
        out.append(byte)
        if val == 0:
            break
    return bytes(out)


def decode_leb128(buffer: bytes, offset: int) -> tuple[int, int]:
    """Decode an unsigned LEB128 varint from buffer starting at offset.

    Returns (value, new_offset).
    """
    result = 0
    shift = 0
    while True:
        if offset >= len(buffer):
            raise ValueError("Unexpected end of buffer while decoding LEB128 varint")
        byte = buffer[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, offset


def pack_nibble(g: int) -> int:
    """Pack a signed Artin generator into a 4-bit nibble.

    Bit 3 = sign (0 = positive, 1 = negative).
    Bits 0-2 = generator index (1 to 7).
    """
    sign_bit = 1 if g < 0 else 0
    idx = abs(g) & 0x07
    return (sign_bit << 3) | idx


def unpack_nibble(nibble: int) -> int:
    """Unpack a 4-bit nibble into a signed Artin generator."""
    sign_bit = (nibble >> 3) & 0x01
    idx = nibble & 0x07
    return -idx if sign_bit else idx


def pack_generators(generators: Sequence[int]) -> bytes:
    """Pack generators two per byte as 4-bit nibbles. Handles odd counts."""
    out = bytearray()
    count = len(generators)
    for i in range(0, count, 2):
        high = pack_nibble(generators[i])
        low = pack_nibble(generators[i + 1]) if (i + 1 < count) else 0
        out.append((high << 4) | (low & 0x0F))
    return bytes(out)


def unpack_generators(payload: bytes, count: int) -> list[int]:
    """Unpack count generators from 4-bit nibble bytes."""
    generators: list[int] = []
    for b in payload:
        high = (b >> 4) & 0x0F
        generators.append(unpack_nibble(high))
        if len(generators) == count:
            break
        low = b & 0x0F
        generators.append(unpack_nibble(low))
        if len(generators) == count:
            break
    return generators


def encode(data: Any) -> tuple[bytes, dict[str, Any]]:
    """Encode multichannel time series data into .tngl binary format.

    Returns (bytes, stats) where stats includes:
      raw_crossings, reduced_crossings, cancelled_pairs, percent_eliminated, tngl_bytes.
    """
    # If a tuple from load_csv (data, warning) was passed
    if isinstance(data, tuple) and len(data) == 2 and hasattr(data, "data"):
        data = data.data

    if hasattr(data, "to_numpy"):
        arr = data.to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"data must be a 2D array, got shape {arr.shape}")

    T, n = arr.shape
    if not (1 <= n <= 8):
        raise ValueError(f".tngl format supports 1 to 8 channels, got {n}")
    if T < 1:
        raise ValueError("data must contain at least 1 sample")

    initial_order, raw_crossings = word_from_frames(arr)
    reduced_crossings, cancelled_pairs = reduce_crossings(raw_crossings)

    raw_count = len(raw_crossings)
    reduced_count = len(reduced_crossings)

    # 1. Header (18 bytes, little endian)
    # magic b"TNGL"; n (uint8, n <= 8); raw crossing count (uint32);
    # reduced count (uint32); T (uint32); mode byte 0.
    header = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        n,
        raw_count,
        reduced_count,
        T,
        MODE_DEFAULT,
    )

    # 2. Anchor: n float32 (first frame continuous values)
    anchor = struct.pack(f"<{n}f", *arr[0])

    # 3. Timing: for each surviving crossing, the tick delta from the previous
    # survivor as an unsigned LEB128 varint.
    timing_bytes = bytearray()
    prev_tick = 0
    for c in reduced_crossings:
        delta = c.tick - prev_tick
        timing_bytes.extend(encode_leb128(delta))
        prev_tick = c.tick

    # 4. Payload: reduced generators packed two per byte as 4-bit nibbles
    payload = pack_generators([c.gen for c in reduced_crossings])

    tngl_bytes = header + anchor + bytes(timing_bytes) + payload

    percent_eliminated = (
        ((raw_count - reduced_count) / raw_count * 100.0) if raw_count > 0 else 0.0
    )

    stats = {
        "raw_crossings": raw_count,
        "reduced_crossings": reduced_count,
        "cancelled_pairs": cancelled_pairs,
        "percent_eliminated": percent_eliminated,
        "tngl_bytes": len(tngl_bytes),
    }

    return tngl_bytes, stats


def decode(data_bytes: bytes) -> tuple[int, int, list[int], list[DecodedCrossing]]:
    """Decode .tngl binary bytes.

    Returns:
      n: int (channel count)
      T: int (total frames/ticks)
      anchor_order: list[int] (channel order at tick 0)
      reduced: list[DecodedCrossing] (list of (tick, gen) surviving crossings)
    """
    if len(data_bytes) < HEADER_SIZE:
        raise ValueError(f"Data too short for .tngl header: {len(data_bytes)} bytes")

    magic, n, raw_count, reduced_count, T, mode = struct.unpack_from(
        HEADER_STRUCT, data_bytes, 0
    )
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes {magic!r}, expected {MAGIC!r}")
    if mode != MODE_DEFAULT:
        raise ValueError(f"Unsupported mode byte: {mode}")

    offset = HEADER_SIZE

    # Read Anchor: n float32
    anchor_size = n * 4
    if len(data_bytes) < offset + anchor_size:
        raise ValueError("Corrupt file: missing anchor frame")
    anchor_floats = struct.unpack_from(f"<{n}f", data_bytes, offset)
    offset += anchor_size
    anchor_order = order_from_frame(anchor_floats)

    # Read Timing: reduced_count unsigned LEB128 varints
    ticks: list[int] = []
    curr_tick = 0
    for _ in range(reduced_count):
        delta, offset = decode_leb128(data_bytes, offset)
        curr_tick += delta
        ticks.append(curr_tick)

    # Read Payload: packed generators
    packed_size = (reduced_count + 1) // 2
    if len(data_bytes) < offset + packed_size:
        raise ValueError(
            f"Corrupt file: expected {packed_size} bytes of generator payload, only {len(data_bytes) - offset} remaining"
        )
    payload_bytes = data_bytes[offset : offset + packed_size]
    generators = unpack_generators(payload_bytes, reduced_count)

    reduced = [
        DecodedCrossing(tick=t, gen=g)
        for t, g in zip(ticks, generators)
    ]

    return n, T, anchor_order, reduced
