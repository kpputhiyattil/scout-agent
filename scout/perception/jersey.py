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
                 min_margin: float = 0.55, max_number: int = 99,
                 allowed: set[int] | None = None) -> dict[int, int]:
    """Weighted majority vote per track.

    weight = conf * sqrt(crop_area). A track gets a number only if the winner's
    score is >= min_score and holds >= min_margin of the total vote mass —
    otherwise stay honest and return nothing for that track.

    max_number bounds reads to plausible squad numbers, and `allowed` (from a
    roster) restricts them to numbers that actually exist in this team — both
    reject OCR noise such as a "7" on an advertising board read as "76".
    """
    scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for r in reads:
        if not (0 < r.number <= max_number):
            continue
        if allowed is not None and r.number not in allowed:
            continue
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
    from scout.config import get_settings
    return easyocr.Reader(["en"], gpu=get_settings().resolve_device() == "cuda")


def _prepare_crop(crop, min_height: int = 128):
    """Upscale and contrast-boost a torso crop so OCR can resolve the digits.

    Jersey numbers on youth kit are only ~30-60 px tall in 720p footage, well
    under what the OCR detector expects; cubic upscaling plus CLAHE on the
    luminance channel recovers a large share of otherwise-missed reads.
    """
    import cv2
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    if h < min_height:
        f = min_height / h
        crop = cv2.resize(crop, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def roster_numbers(roster_csv: str | Path | None) -> set[int] | None:
    """Squad numbers from the roster, used to reject impossible OCR reads."""
    if roster_csv is None or not Path(roster_csv).exists():
        return None
    nums = pd.read_csv(roster_csv)["jersey_number"].dropna().astype(int)
    return set(nums) or None


def read_jerseys(video_path: str | Path, tracks: pd.DataFrame,
                 sample_every: int | None = None,
                 allowed: set[int] | None = None) -> dict[int, int]:
    """Run OCR over sampled torso crops, return {track_id: jersey_number}."""
    import cv2

    from scout.config import get_settings
    from scout.perception.detect import torso_crop

    s = get_settings()
    sample_every = sample_every or s.ocr_sample_every_n_frames
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
            area = float(crop.shape[0] * crop.shape[1])   # original size: small crops lie
            prepared = _prepare_crop(crop)
            if prepared is None:
                continue
            for _, text, conf in reader.readtext(prepared, allowlist="0123456789"):
                if text.isdigit():
                    reads.append(Read(int(r.track_id), int(text), float(conf), area))
    cap.release()
    return vote_jerseys(reads, max_number=s.max_jersey_number, allowed=allowed)


def merge_map(jerseys: dict[int, int], teams: dict[int, str]) -> dict[int, int]:
    """Collapse track fragments into players: {track_id: canonical_track_id}.

    Trackers lose identity constantly on panning/cut footage — one child becomes
    dozens of short tracks. The jersey number is the only stable anchor we have,
    so every fragment reading the same number for the same team is treated as the
    same player (canonical id = the lowest fragment id). Fragments with no number
    stay separate: guessing would fabricate players.
    """
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for tid, num in jerseys.items():
        groups[(teams.get(tid, "?"), num)].append(int(tid))
    out: dict[int, int] = {}
    for members in groups.values():
        canonical = min(members)
        for tid in members:
            out[tid] = canonical
    return out


def join_roster(jerseys: dict[int, int], roster_csv: str | Path | None) -> dict[int, str]:
    """{track_id: player_name} via roster CSV with columns jersey_number,name."""
    if roster_csv is None or not Path(roster_csv).exists():
        return {}
    roster = pd.read_csv(roster_csv)
    by_num = dict(zip(roster["jersey_number"].astype(int), roster["name"]))
    return {tid: by_num[num] for tid, num in jerseys.items() if num in by_num}
