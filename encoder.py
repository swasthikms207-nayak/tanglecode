import struct

class TangleEncoder:
    MAGIC_BYTES = b"TNGL"

    @classmethod
    def pack_braid_file(cls, filepath: str, n_channels: int, braid: list[int], initial_baseline: list[float]):
        """Packs metadata, baseline floats, and 4-bit generator nibbles into .tngl format."""
        with open(filepath, "wb") as f:
            f.write(cls.MAGIC_BYTES)
            f.write(struct.pack("!BII", n_channels, len(braid), len(braid)))
            
            for val in initial_baseline:
                f.write(struct.pack("!f", val))

            # Pack two 4-bit generators per byte
            byte_buffer = 0
            nibble_count = 0
            for gen in braid:
                byte_buffer = (byte_buffer << 4) | (gen & 0x0F)
                nibble_count += 1
                if nibble_count == 2:
                    f.write(bytes([byte_buffer]))
                    byte_buffer = 0
                    nibble_count = 0

            if nibble_count == 1:
                f.write(bytes([byte_buffer << 4]))
