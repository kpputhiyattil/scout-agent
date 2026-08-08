"""Team assignment: cluster torso-crop colors into two kits, majority-vote per track.

Default features: mean HSV of the torso crop (fast, no extra deps).
If `transformers` + a GPU are available you can switch to SigLIP embeddings by
passing embed_fn — the clustering/voting logic is identical.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _hsv_feature(crop: np.ndarray) -> np.ndarray | None:
    import cv2
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    # drop grass-green pixels so background doesn't pollute the kit color
    not_green = ~((hsv[:, 0] > 35) & (hsv[:, 0] < 85) & (hsv[:, 1] > 60))
    px = hsv[not_green] if not_green.sum() > 20 else hsv
    return np.concatenate([px.mean(axis=0), px.std(axis=0)])


def assign_teams(video_path: str | Path, tracks: pd.DataFrame,
                 sample_every: int = 12,
                 embed_fn: Callable[[np.ndarray], np.ndarray | None] = _hsv_feature,
                 ) -> tuple[dict[int, str], float]:
    """Returns ({track_id: 'A'|'B'}, cluster_separation_score 0-1).

    Low separation (<0.3) => kits too similar; dashboard asks coach to confirm.
    """
    import cv2
    from sklearn.cluster import KMeans

    from scout.perception.detect import torso_crop

    feats, owners = [], []
    cap = cv2.VideoCapture(str(video_path))
    wanted = tracks[tracks.frame % sample_every == 0].groupby("frame")

    for frame_idx, group in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        for _, r in group.iterrows():
            f = embed_fn(torso_crop(frame, (r.x1, r.y1, r.x2, r.y2)))
            if f is not None:
                feats.append(f)
                owners.append(int(r.track_id))
    cap.release()

    if len(feats) < 4:
        return {}, 0.0

    X = np.asarray(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    inter = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])
    intra = np.mean([np.linalg.norm(X[km.labels_ == k] - km.cluster_centers_[k], axis=1).mean()
                     for k in (0, 1)])
    separation = float(inter / (inter + 2 * intra + 1e-6))

    votes: dict[int, list[int]] = defaultdict(list)
    for tid, lab in zip(owners, km.labels_):
        votes[tid].append(int(lab))
    team_of = {tid: ("A" if np.mean(v) < 0.5 else "B") for tid, v in votes.items()}
    return team_of, separation
