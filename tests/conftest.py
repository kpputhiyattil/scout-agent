import numpy as np
import pandas as pd
import pytest

FPS = 25.0


def make_tracks(specs: dict[int, tuple[str, list[tuple[int, float, float]]]]) -> pd.DataFrame:
    """specs: track_id -> (team, [(frame, x_m, y_m), ...])"""
    rows = []
    for tid, (team, pts) in specs.items():
        for f, x, y in pts:
            rows.append((f, tid, team, x, y))
    return pd.DataFrame(rows, columns=["frame", "track_id", "team", "x_m", "y_m"])


def straight_run(frames: range, x0: float, y0: float, vx: float, vy: float = 0.0, fps: float = FPS):
    return [(f, x0 + vx * (f - frames.start) / fps, y0 + vy * (f - frames.start) / fps) for f in frames]


@pytest.fixture
def fps():
    return FPS
