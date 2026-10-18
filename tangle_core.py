import math
import random

class TangleCore:
    def __init__(self, n_channels: int = 8):
        self.n = n_channels

    def frames_to_braid(self, frames: list[list[float]]) -> list[int]:
        """Converts multi-channel telemetry snapshots into elementary braid crossings (sigma_i)."""
        braid = []
        if len(frames) < 2:
            return braid

        # Initial ranking by amplitude
        curr_ranks = sorted(range(self.n), key=lambda i: frames[0][i])

        for frame in frames[1:]:
            target_ranks = sorted(range(self.n), key=lambda i: frame[i])
            temp_ranks = list(curr_ranks)

            # Bubble-sort transpositions map directly to Artin braid generators
            for i in range(len(temp_ranks)):
                for j in range(len(temp_ranks) - 1 - i):
                    if target_ranks.index(temp_ranks[j]) > target_ranks.index(temp_ranks[j + 1]):
                        gen = j + 1  # Generator 1 to N-1
                        braid.append(gen)
                        temp_ranks[j], temp_ranks[j + 1] = temp_ranks[j + 1], temp_ranks[j]
            curr_ranks = target_ranks
        return braid

    def untangle(self, braid: list[int]) -> tuple[list[int], int]:
        """Reidemeister Type-II reduction: adjacent identical transpositions cancel out."""
        stack = []
        cancellations = 0
        for gen in braid:
            if stack and stack[-1] == gen:
                stack.pop()
                cancellations += 1
            else:
                stack.append(gen)
        return stack, cancellations

    @staticmethod
    def generate_mock_telemetry(steps: int = 500, jitter: float = 0.15) -> list[list[float]]:
        """Generates 8-channel telemetry simulating drone IMU jitter."""
        frames = []
        for t in range(steps):
            row = [
                12.0 + 0.25 * math.sin(t * 0.4) + random.uniform(-jitter, jitter),
                12.0 + 0.25 * math.cos(t * 0.4) + random.uniform(-jitter, jitter),
                24.0 + 0.02 * t,
                36.0 - 0.01 * t,
                48.0 + 1.5 * math.sin(t * 0.05),
                60.0 + random.uniform(-0.02, 0.02),
                75.0,
                90.0 + 0.05 * t
            ]
            frames.append(row)
        return frames