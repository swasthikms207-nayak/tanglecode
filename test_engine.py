import os
from tangle_core import TangleCore
from encoder import TangleEncoder
from decoder import TangleDecoder

def test_full_pipeline():
    core = TangleCore(n_channels=8)
    frames = core.generate_mock_telemetry(steps=100, jitter=0.15)

    raw_braid = core.frames_to_braid(frames)
    reduced_braid, _ = core.untangle(raw_braid)

    test_file = "test_run.tngl"
    TangleEncoder.pack_braid_file(test_file, 8, reduced_braid, frames[0])

    channels, baseline, unpacked_braid = TangleDecoder.unpack_braid_file(test_file)

    # Clean up test artifact
    if os.path.exists(test_file):
        os.remove(test_file)

    assert channels == 8, "Channel count mismatch"
    assert len(unpacked_braid) == len(reduced_braid), "Generator length mismatch"
    assert unpacked_braid == reduced_braid, "Unpacked braid does not match reduced braid"
    print("✓ All roundtrip integrity tests passed successfully!")

if __name__ == "__main__":
    test_full_pipeline()
