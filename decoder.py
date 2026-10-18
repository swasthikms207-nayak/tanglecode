import struct

class TangleDecoder:
    MAGIC_BYTES = b"TNGL"

    @classmethod
    def unpack_braid_file(cls, filepath: str) -> tuple[int, list[float], list[int]]:
        """Reads a .tngl file and unpacks channels, baseline floats, and generator array."""
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != cls.MAGIC_BYTES:
                raise ValueError(f"Invalid format: expected {cls.MAGIC_BYTES}, got {magic}")

            header = f.read(9)
            n_channels, _, reduced_count = struct.unpack("!BII", header)

            baseline = []
            for _ in range(n_channels):
                val = struct.unpack("!f", f.read(4))[0]
                baseline.append(val)

            braid = []
            while len(braid) < reduced_count:
                byte_val = f.read(1)
                if not byte_val:
                    break
                b = byte_val[0]
                braid.append((b >> 4) & 0x0F)
                if len(braid) < reduced_count:
                    braid.append(b & 0x0F)

        return n_channels, baseline, braid

    @classmethod
    def braid_to_relative_ranks(cls, n_channels: int, braid: list[int]) -> list[list[int]]:
        """Reconstructs channel rank trajectory by applying elementary transpositions."""
        current_ranks = list(range(n_channels))
        rank_history = [list(current_ranks)]

        for gen in braid:
            idx = gen - 1
            if 0 <= idx < n_channels - 1:
                current_ranks[idx], current_ranks[idx + 1] = current_ranks[idx + 1], current_ranks[idx]
                rank_history.append(list(current_ranks))

        return rank_history
