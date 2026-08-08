"""Detection + tracking: video -> tracks.parquet + ball.parquet (+ debug video).

Weights are auto-selected by hardware: COCO yolov8x on GPU, yolov8n on CPU
(classes: person, sports ball). For best results set SCOUT_DETECTOR_WEIGHTS to a
football-specific checkpoint trained with classes player/goalkeeper/ball/referee
(e.g. from the Roboflow football-players-detection dataset) — the class mapping
below adapts automatically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scout.config import get_settings

# COCO fallback mapping
COCO_PERSON, COCO_BALL = 0, 32
FOOTBALL_CLASSES = {"player", "goalkeeper", "ball", "referee"}


def run_detection_tracking(video_path: str | Path, out_dir: str | Path,
                           save_debug_video: bool = True, progress=None) -> tuple[Path, Path]:
    """Returns (tracks_parquet, ball_parquet).

    tracks.parquet: frame, track_id, x1, y1, x2, y2, conf, cls  (people)
    ball.parquet:   frame, x, y, conf                            (pixel center)
    """
    import cv2
    import supervision as sv
    from ultralytics import YOLO

    s = get_settings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_pq, ball_pq = out_dir / "tracks.parquet", out_dir / "ball.parquet"

    device = s.resolve_device()
    weights = s.resolve_detector_weights()
    model = YOLO(weights)
    names = {i: n.lower() for i, n in model.names.items()}
    is_football_model = FOOTBALL_CLASSES & set(names.values())
    tracker = sv.ByteTrack(frame_rate=s.target_fps)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer = None
    if save_debug_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / "debug.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), s.target_fps, (w, h))
        box_ann = sv.BoxAnnotator()
        label_ann = sv.LabelAnnotator()

    track_rows, ball_rows = [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=s.detector_conf, verbose=False, device=device)[0]
        det = sv.Detections.from_ultralytics(res)

        if is_football_model:
            person_mask = np.isin(det.class_id,
                                  [i for i, n in names.items() if n in ("player", "goalkeeper", "referee")])
            ball_mask = np.isin(det.class_id, [i for i, n in names.items() if n == "ball"])
        else:
            person_mask = det.class_id == COCO_PERSON
            ball_mask = det.class_id == COCO_BALL

        people = tracker.update_with_detections(det[person_mask])
        for (x1, y1, x2, y2), tid, conf, cid in zip(
                people.xyxy, people.tracker_id, people.confidence, people.class_id):
            track_rows.append((frame_idx, int(tid), float(x1), float(y1), float(x2), float(y2),
                               float(conf), names.get(int(cid), "person")))

        balls = det[ball_mask]
        if len(balls):
            i = int(np.argmax(balls.confidence))
            x1, y1, x2, y2 = balls.xyxy[i]
            ball_rows.append((frame_idx, float((x1 + x2) / 2), float((y1 + y2) / 2),
                              float(balls.confidence[i])))

        if writer is not None:
            labels = [f"#{tid}" for tid in people.tracker_id]
            img = box_ann.annotate(frame.copy(), people)
            img = label_ann.annotate(img, people, labels=labels)
            writer.write(img)
        frame_idx += 1
        if progress and total_frames and frame_idx % 25 == 0:
            progress(min(1.0, frame_idx / total_frames))

    cap.release()
    if writer is not None:
        writer.release()

    pd.DataFrame(track_rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"]
                 ).to_parquet(tracks_pq, index=False)
    pd.DataFrame(ball_rows, columns=["frame", "x", "y", "conf"]).to_parquet(ball_pq, index=False)
    return tracks_pq, ball_pq


def torso_crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Upper-half crop of a player's bbox — where the jersey number lives."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h = y2 - y1
    return frame[y1 + int(0.1 * h): y1 + int(0.55 * h), x1:x2]
