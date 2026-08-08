"""Pixel -> pitch-meter projection.

Preferred: a pitch-keypoint model (e.g. YOLOv8-pose fine-tuned on the Roboflow
football-field-detection dataset) re-estimated every N frames for panning cameras.
MVP fallback: 4 coach-clicked reference points (corners / box corners) captured
once in the dashboard -> single static homography. Both paths produce the same
interface: project(frame, x_px, y_px) -> (x_m, y_m).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scout.config import PITCH_LENGTH_M, PITCH_WIDTH_M


class RelativeProjector:
    """Camera-relative fallback for footage with no usable homography.

    Scale comes from player bounding-box height: a youth player is roughly
    `player_height_m` tall, so metres_per_pixel = player_height_m / median bbox
    height in that frame. Distances between nearby objects are approximately
    right, which is all the event rules (possession radius, duel radius, ball
    speed) actually need. Absolute positions are NOT meaningful — they shift
    whenever the camera pans or zooms — so pitch-position metrics are suppressed
    downstream.
    """

    mode = "relative"

    def __init__(self, scale: pd.Series):
        # scale: index = frame, value = metres per pixel
        self.scale = scale.replace([np.inf, -np.inf], np.nan).dropna()
        self.median = float(self.scale.median()) if len(self.scale) else 0.02

    def scale_at(self, frame: int) -> float:
        v = self.scale.get(frame)
        return float(v) if v and np.isfinite(v) else self.median

    def project_df(self, df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
        out = df.copy()
        mpp = df["frame"].map(self.scale).fillna(self.median).to_numpy()
        out["x_m"] = df[xcol].to_numpy() * mpp
        out["y_m"] = df[ycol].to_numpy() * mpp
        return out


def from_pixel_scale(tracks: pd.DataFrame, player_height_m: float) -> RelativeProjector:
    """Build a RelativeProjector from person bounding boxes (needs y1, y2 columns)."""
    h_px = (tracks["y2"] - tracks["y1"]).where(lambda s: s > 1)
    med = tracks.assign(_h=h_px).groupby("frame")["_h"].median()
    return RelativeProjector(player_height_m / med)


class Projector:
    """Piecewise-constant homographies over frame windows."""

    mode = "pitch"

    def __init__(self, windows: list[tuple[int, np.ndarray]]):
        # windows: sorted [(start_frame, 3x3 H), ...]
        self.windows = windows

    def H(self, frame: int) -> np.ndarray | None:
        h = None
        for start, mat in self.windows:
            if frame >= start:
                h = mat
            else:
                break
        return h

    def project(self, frame: int, x: float, y: float) -> tuple[float, float] | None:
        H = self.H(frame)
        if H is None:
            return None
        v = H @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        px, py = v[0] / v[2], v[1] / v[2]
        # clamp with small tolerance; wildly off-pitch => bad homography for this point
        if -10 <= px <= PITCH_LENGTH_M + 10 and -10 <= py <= PITCH_WIDTH_M + 10:
            return float(np.clip(px, 0, PITCH_LENGTH_M)), float(np.clip(py, 0, PITCH_WIDTH_M))
        return None

    def project_df(self, df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
        out = df.copy()
        pts = [self.project(int(f), x, y) for f, x, y in zip(df["frame"], df[xcol], df[ycol])]
        out["x_m"] = [p[0] if p else np.nan for p in pts]
        out["y_m"] = [p[1] if p else np.nan for p in pts]
        return out


def from_reference_points(ref_json: str | Path) -> Projector:
    """Static homography from 4+ coach-clicked points.

    ref_json: {"points": [{"px": [x,y], "pitch": [x_m, y_m]}, ...]}
    """
    import cv2
    data = json.loads(Path(ref_json).read_text())
    src = np.float32([p["px"] for p in data["points"]])
    dst = np.float32([p["pitch"] for p in data["points"]])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        raise ValueError("Could not compute homography from reference points")
    return Projector([(0, H)])


def from_keypoint_model(video_path: str | Path, weights: str,
                        every_n: int = 30, min_points: int = 4) -> Projector:
    """Moving-camera homography via pitch landmark detection.

    Falls back to the last valid H when too few landmarks are visible.
    Requires a keypoint checkpoint whose keypoint order matches PITCH_LANDMARKS_M.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    landmarks_m = np.array(PITCH_LANDMARKS_M, dtype=np.float32)
    windows: list[tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for f in range(0, n, every_n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        kp = res.keypoints.xy[0].cpu().numpy()
        vis = res.keypoints.conf[0].cpu().numpy() > 0.5
        if vis.sum() < min_points:
            continue
        H, _ = cv2.findHomography(kp[vis], landmarks_m[vis], cv2.RANSAC)
        if H is not None:
            windows.append((f, H))
    cap.release()
    if not windows:
        raise RuntimeError("No homography could be estimated; use reference-point fallback")
    return Projector(windows)


# 32-point pitch landmark template (meters), matching the common
# football-field-detection keypoint ordering: corners, box corners, arcs, center.
_L, _W = PITCH_LENGTH_M, PITCH_WIDTH_M
_BD, _BW = 16.5, 40.3
_GD, _GW = 5.5, 18.32
PITCH_LANDMARKS_M = [
    (0, 0), (0, (_W - _BW) / 2), (0, (_W - _GW) / 2), (0, (_W + _GW) / 2), (0, (_W + _BW) / 2), (0, _W),
    (_GD, (_W - _GW) / 2), (_GD, (_W + _GW) / 2),
    (11.0, _W / 2),
    (_BD, (_W - _BW) / 2), (_BD, (_W + _BW) / 2),
    (_L / 2, 0), (_L / 2, _W / 2 - 9.15), (_L / 2, _W / 2), (_L / 2, _W / 2 + 9.15), (_L / 2, _W),
    (_L - _BD, (_W - _BW) / 2), (_L - _BD, (_W + _BW) / 2),
    (_L - 11.0, _W / 2),
    (_L - _GD, (_W - _GW) / 2), (_L - _GD, (_W + _GW) / 2),
    (_L, 0), (_L, (_W - _BW) / 2), (_L, (_W - _GW) / 2), (_L, (_W + _GW) / 2), (_L, (_W + _BW) / 2), (_L, _W),
    (_BD + 9.15, _W / 2), (_L - _BD - 9.15, _W / 2),
    (_L / 2 - 9.15, _W / 2), (_L / 2 + 9.15, _W / 2),
    (11.0 + 9.15, _W / 2),
]
