"""Ingest: YouTube URL or local file -> normalized MP4 + Match row.

Normalization makes every downstream model see identical input:
720p, 25 fps, H.264, yuv420p.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from scout.config import get_settings
from scout.db import Match, get_session


def _match_id(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:12]


def _probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (vstream.get("avg_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den or 1)
    return {"fps": fps, "duration_s": float(info["format"].get("duration", 0.0))}


def _download(url: str, dest: Path) -> Path:
    """Download with yt-dlp. Fails fast on DRM/broken URLs before any GPU time."""
    raw = dest / "raw.mp4"
    subprocess.run(
        ["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]", "--merge-output-format", "mp4",
         "-o", str(raw), url],
        check=True,
    )
    return raw


def _normalize(src: Path, dest: Path) -> Path:
    s = get_settings()
    out = dest / "video.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vf", f"scale=-2:{s.target_height},fps={s.target_fps}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
         str(out)],
        check=True, capture_output=True,
    )
    return out


def ingest(source: str) -> str:
    """source = URL or local path. Returns match_id. Idempotent."""
    s = get_settings()
    mid = _match_id(source)
    mdir = s.match_dir(mid)
    video = mdir / "video.mp4"

    if not video.exists():
        if source.startswith(("http://", "https://")):
            raw = _download(source, mdir)
        else:
            raw = Path(source)
            if not raw.exists():
                raise FileNotFoundError(source)
            if raw.resolve() != (mdir / raw.name).resolve():
                shutil.copy(raw, mdir / raw.name)
                raw = mdir / raw.name
        _normalize(raw, mdir)

    meta = _probe(video)
    with get_session() as db:
        if not db.get(Match, mid):
            db.add(Match(id=mid, source=source, video_path=str(video),
                         fps=meta["fps"], duration_s=meta["duration_s"]))
            db.commit()
    (mdir / "meta.json").write_text(json.dumps(meta))
    return mid
