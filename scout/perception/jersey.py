"""Jersey number OCR with temporal voting.

Per-frame OCR on kids' jerseys is unreliable (~60%); voting hundreds of reads
per track with confidence x crop-size weighting gets per-track accuracy >95%.
The voting logic is pure and unit-tested; the OCR engine is pluggable.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Read:
    track_id: int
    number: int
    conf: float
    crop_area: float  # px^2 — small crops lie


def vote_jerseys(reads: list[Read], min_score: float = 1.0,
                 min_margin: float = 0.55) -> dict[int, int]:
    """Weighted majority vote per track.

    weight = conf * sqrt(crop_area). A track gets a number only if the winner's
    score is >= min_score and holds >= min_margin of the total vote mass —
    otherwise stay honest and return nothing for that track.
    """
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for r in reads:
        if 0 < r.number < 100:
            scores[r.track_id][r.number] += r.conf * float(np.sqrt(max(r.crop_area, 1.0)))

    result: dict[int, int] = {}
    for tid, tally in scores.items():
        total = sum(tally.values())
        num, best = max(tally.items(), key=lambda kv: kv[1])
        if best >= min_score and best / total >= min_margin:
            result[tid] = num
    return result


def _ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=True)


def read_jerseys(video_path: str | Path, tracks: pd.DataFrame,
                 sample_every: int = 12) -> dict[int, int]:
    """Run OCR over sampled torso crops, return {track_id: jersey_number}."""
    import cv2

    from scout.perception.detect import torso_crop

    reader = _ocr_reader()
    reads: list[Read] = []
    cap = cv2.VideoCapture(str(video_path))

    for frame_idx, group in tracks[tracks.frame % sample_every == 0].groupby("frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        for _, r in group.iterrows():
            crop = torso_crop(frame, (r.x1, r.y1, r.x2, r.y2))
            if crop.size == 0:
                continue
            for _, text, conf in reader.readtext(crop, allowlist="0123456789"):
                if text.isdigit():
                    reads.append(Read(int(r.track_id), int(text), float(conf),
                                      float(crop.shape[0] * crop.shape[1])))
    cap.release()
    return vote_jerseys(reads)


def join_roster(jerseys: dict[int, int], roster_csv: str | Path | None) -> dict[int, str]:
    """{track_id: player_name} via roster CSV with columns jersey_number,name."""
    if roster_csv is None or not Path(roster_csv).exists():
        return {}
    roster = pd.read_csv(roster_csv)
    by_num = dict(zip(roster["jersey_number"].astype(int), roster["name"]))
    return {tid: by_num[num] for tid, num in jerseys.items() if num in by_num}
