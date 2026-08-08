"""Auto-cut highlight clips per player from their key events."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from scout.ingest import ensure_ffmpeg

HIGHLIGHT_TYPES = ("shot", "save", "duel", "interception", "goal")


def cut_highlights(video_path: str | Path, events: pd.DataFrame, fps: float,
                   out_dir: str | Path, track_id: int,
                   pre_s: float = 3.0, post_s: float = 3.0, max_clips: int = 5) -> list[Path]:
    ensure_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor_col = "actor" if "actor" in events.columns else "winner"
    mine = events[(events.type.isin(HIGHLIGHT_TYPES)) & (events.get(actor_col) == track_id)]
    clips = []
    for i, ev in enumerate(mine.head(max_clips).itertuples()):
        start = max(0.0, ev.frame / fps - pre_s)
        out = out_dir / f"p{track_id}_{i}_{ev.type}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(video_path),
             "-t", f"{pre_s + post_s:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-an", str(out)],
            check=True, capture_output=True,
        )
        clips.append(out)
    return clips
