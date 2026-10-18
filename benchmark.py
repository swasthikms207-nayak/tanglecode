import os
import gzip
import zstandard as zstd
from tangle_core import TangleCore
from encoder import TangleEncoder

def run_benchmark():
    core = TangleCore(n_channels=8)
    frames = core.generate_mock_telemetry(steps=1000, jitter=0.18)

    # 1. Raw CSV
    raw_csv = "raw_telemetry.csv"
    with open(raw_csv, "w") as f:
        f.write("ch0,ch1,ch2,ch3,ch4,ch5,ch6,ch7\n")
        for row in frames:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")

    with open(raw_csv, "rb") as f:
        csv_bytes = f.read()

    # 2. Gzip
    gz_file = "telemetry.csv.gz"
    with open(gz_file, "wb") as f:
        f.write(gzip.compress(csv_bytes, compresslevel=6))

    # 3. Zstd
    zst_file = "telemetry.csv.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(zst_file, "wb") as f:
        f.write(cctx.compress(csv_bytes))

    # 4. TangleCode
    raw_braid = core.frames_to_braid(frames)
    reduced_braid, cancelled = core.untangle(raw_braid)
    tngl_file = "telemetry.tngl"
    TangleEncoder.pack_braid_file(tngl_file, 8, reduced_braid, frames[0])

    # Sizes
    raw_sz = os.path.getsize(raw_csv)
    gz_sz = os.path.getsize(gz_file)
    zst_sz = os.path.getsize(zst_file)
    tngl_sz = os.path.getsize(tngl_file)

    print("\n" + "=" * 55)
    print("       TANGLECODE TELEMETRY COMPRESSION BENCHMARK")
    print("=" * 55)
    print(f"Time Frames:         {len(frames)}")
    print(f"Raw Crossings:       {len(raw_braid)}")
    print(f"Untangled Crossings: {len(reduced_braid)} (Cancelled {cancelled} loops)")
    print("-" * 55)
    print(f"Raw CSV File:        {raw_sz:>8} bytes  (100.0%)")
    print(f"Gzip (Level 6):      {gz_sz:>8} bytes  ({gz_sz / raw_sz * 100:.1f}%)")
    print(f"Zstandard (Level 3): {zst_sz:>8} bytes  ({zst_sz / raw_sz * 100:.1f}%)")
    print(f"TangleCode (.tngl):  {tngl_sz:>8} bytes  ({tngl_sz / raw_sz * 100:.1f}%)")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    run_benchmark()
