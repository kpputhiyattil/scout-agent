"""Ingest: YouTube URL or local file -> normalized MP4 + Match row.

Normalization makes every downstream model see identical input:
720p, 25 fps, H.264, yuv420p.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

from scout.config import get_settings
from scout.db import Match, get_session


def ensure_ffmpeg() -> None:
    """Make ffmpeg/ffprobe available, preferring the pip-installed static-ffmpeg bundle."""
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if not missing:
        return
    try:
        from static_ffmpeg import add_paths
        add_paths()  # downloads bundled ffmpeg+ffprobe on first use, prepends to PATH
        missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    except ImportError:
        pass
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not found. Easiest fix: `pip install static-ffmpeg` "
            "and retry. Or download ffmpeg from https://www.gyan.dev/ffmpeg/builds/ "
            "(release-essentials zip), extract, and add its bin folder to PATH, "
            "then restart the terminal and Streamlit.")


def _find_executable(name: str, extra_dirs: list[Path] | None = None) -> str | None:
    """Resolve an executable on PATH or in known install locations."""
    found = shutil.which(name)
    if found:
        return found
    candidates: list[Path] = []
    if extra_dirs:
        candidates.extend(extra_dirs)
    home = Path.home()
    if name == "deno":
        candidates.extend([
            home / ".deno" / "bin" / ("deno.exe" if os.name == "nt" else "deno"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "deno" / "bin" / "deno.exe",
        ])
    elif name == "node":
        candidates.extend([
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "node.exe",
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "nodejs" / "node.exe",
        ])
    for path in candidates:
        if path and path.is_file():
            return str(path)
    return None


def _js_runtimes() -> dict[str, dict[str, str]]:
    """Pick a JS runtime for yt-dlp YouTube challenge solving (deno preferred, then node)."""
    deno = _find_executable("deno")
    if deno:
        return {"deno": {"path": deno}}
    node = _find_executable("node")
    if node:
        return {"node": {"path": node}}
    raise RuntimeError(
        "YouTube downloads require a JavaScript runtime (Deno or Node). "
        "Install Deno from https://deno.com (recommended) or Node 22+ from "
        "https://nodejs.org, then restart the terminal and Streamlit.")


def match_id_for(source: str) -> str:
    """Deterministic match id for a source (URL or path) — safe to call before ingest."""
    return hashlib.sha1(source.encode()).hexdigest()[:12]


_match_id = match_id_for


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


def _download(url: str, dest: Path, progress=None) -> Path:
    """Download with yt-dlp (Python API). Fails fast on DRM/broken URLs before any GPU time."""
    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed in this environment. Run `pip install \"yt-dlp[default]\"` "
            '(or `pip install -e ".[perception]"`), then restart Streamlit.')
    if importlib.util.find_spec("yt_dlp_ejs") is None:
        raise RuntimeError(
            "yt-dlp-ejs is missing. Run `pip install \"yt-dlp[default]\"` "
            "(or `pip install yt-dlp-ejs`), then restart Streamlit.")
    ensure_ffmpeg()  # yt-dlp needs ffmpeg to merge separate video/audio streams
    import yt_dlp
    raw = dest / "raw.mp4"

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(raw),
        "progress_hooks": [hook],
        "quiet": True,
        "noprogress": True,
        "js_runtimes": _js_runtimes(),
    }
    s = get_settings()
    if s.ytdlp_cookies:
        opts["cookiefile"] = s.ytdlp_cookies
    elif s.ytdlp_cookies_from_browser:
        opts["cookiesfrombrowser"] = (s.ytdlp_cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc)
        if "No supported JavaScript runtime" in msg or "JS runtime" in msg:
            raise RuntimeError(
                "YouTube download failed: no usable JavaScript runtime. "
                "Install Deno (https://deno.com) or Node 22+ (https://nodejs.org), "
                "then restart Streamlit.") from exc
        if "not a bot" in msg or "Sign in to confirm" in msg:
            raise RuntimeError(
                "YouTube blocked this download as bot traffic — common on cloud/datacenter "
                "IPs (Colab, VPS). Options: (1) download the video on your own machine and "
                "use the local-folder source instead; (2) export cookies.txt from a logged-in "
                "browser and set SCOUT_YTDLP_COOKIES=/path/cookies.txt; (3) locally, set "
                "SCOUT_YTDLP_COOKIES_FROM_BROWSER=chrome.") from exc
        raise
    return raw


def _normalize(src: Path, dest: Path, progress=None) -> Path:
    s = get_settings()
    out = dest / "video.mp4"
    duration = _probe(src)["duration_s"] or 0
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-i", str(src),
         "-vf", f"scale=-2:{s.target_height},fps={s.target_fps}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
         "-progress", "pipe:1", "-nostats", "-loglevel", "error",
         str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:  # ffmpeg emits out_time_ms= lines (microseconds)
        if progress and duration and line.startswith("out_time_ms="):
            try:
                progress(min(1.0, int(line.split("=")[1]) / 1e6 / duration))
            except ValueError:
                pass
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to normalize {src.name}")
    return out


def ingest(source: str, progress=None) -> str:
    """source = URL or local path. Returns match_id. Idempotent.

    progress: optional callable(fraction 0..1) for live status reporting.
    """
    s = get_settings()
    mid = _match_id(source)
    mdir = s.match_dir(mid)
    video = mdir / "video.mp4"
    p = progress or (lambda f: None)

    ensure_ffmpeg()
    if not video.exists():
        if source.startswith(("http://", "https://")):
            # download = first 90% of the stage, normalize = last 10%
            raw = _download(source, mdir, progress=lambda f: p(f * 0.9))
            _normalize(raw, mdir, progress=lambda f: p(0.9 + f * 0.1))
        else:
            raw = Path(source)
            if not raw.exists():
                raise FileNotFoundError(source)
            if raw.resolve() != (mdir / raw.name).resolve():
                shutil.copy(raw, mdir / raw.name)
                raw = mdir / raw.name
            _normalize(raw, mdir, progress=p)

    meta = _probe(video)
    with get_session() as db:
        if not db.get(Match, mid):
            db.add(Match(id=mid, source=source, video_path=str(video),
                         fps=meta["fps"], duration_s=meta["duration_s"]))
            db.commit()
    (mdir / "meta.json").write_text(json.dumps(meta))
    return mid
