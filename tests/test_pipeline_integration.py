"""End-to-end pipeline run with perception stubbed out.

Exercises the real orchestrator — ingest, projection-mode selection, analyze,
rate, report — on a generated video, so stage wiring, artifact formats and DB
writes are verified without needing a GPU or a real match. Perception (YOLO,
OCR, colour clustering) is stubbed: those need real footage and are covered by
their own unit tests.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scout.ingest import ensure_ffmpeg

N_PLAYERS = 12
N_FRAMES = 100          # 4 s at 25 fps — enough to exercise every stage, fast to encode


def _make_video(path: Path, seconds: int = 4) -> Path:
    ensure_ffmpeg()
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True)
    return path


def _fake_tracks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Players drifting across frame + a ball that changes carrier."""
    rng = np.random.default_rng(7)
    rows = []
    for tid in range(1, N_PLAYERS + 1):
        base_x = 40 + tid * 40
        for f in range(N_FRAMES):
            x = base_x + 20 * np.sin(f / 40 + tid)
            y = 180 + 30 * np.cos(f / 55 + tid)
            h = 60 + rng.normal(0, 2)          # bbox height ~ player height
            rows.append((f, tid, x - 12, y - h / 2, x + 12, y + h / 2, 0.9, "person"))
    tracks = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"])

    ball = []
    for f in range(N_FRAMES):
        carrier = 1 + (f // 40) % N_PLAYERS   # ball changes hands every 40 frames
        t = tracks[(tracks.frame == f) & (tracks.track_id == carrier)].iloc[0]
        ball.append((f, (t.x1 + t.x2) / 2 + 3, t.y2 - 2, 0.8))
    return tracks, pd.DataFrame(ball, columns=["frame", "x", "y", "conf"])


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """Isolated data dir + stubbed perception stage."""
    monkeypatch.setenv("SCOUT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCOUT_DEVICE", "cpu")
    # the synthetic clip is 10 s long; keep every track so stage wiring is what's tested
    monkeypatch.setenv("SCOUT_MIN_TRACK_SECONDS", "1")

    import scout.config as config
    import scout.db as db
    config.get_settings.cache_clear()
    db._engine = None            # rebind the SQLite engine to this test's data dir

    tracks, ball = _fake_tracks()

    def fake_detection(video_path, out_dir, save_debug_video=True, progress=None):
        out = Path(out_dir)
        tracks.to_parquet(out / "tracks.parquet", index=False)
        ball.to_parquet(out / "ball.parquet", index=False)
        if progress:
            progress(1.0)
        return out / "tracks.parquet", out / "ball.parquet"

    def fake_teams(video_path, tracks_df):
        return {int(t): ("A" if int(t) <= N_PLAYERS // 2 else "B")
                for t in tracks_df.track_id.unique()}, 0.8

    def fake_jerseys(video_path, tracks_df, **kw):
        return {int(t): int(t) for t in tracks_df.track_id.unique()}

    import scout.perception.detect as detect
    import scout.perception.jersey as jersey
    import scout.perception.team as team
    monkeypatch.setattr(detect, "run_detection_tracking", fake_detection)
    monkeypatch.setattr(team, "assign_teams", fake_teams)
    monkeypatch.setattr(jersey, "read_jerseys", fake_jerseys)
    monkeypatch.setattr(jersey, "join_roster", lambda j, r: {k: f"Player {v}" for k, v in j.items()})

    yield tmp_path
    config.get_settings.cache_clear()
    db._engine = None


def test_full_pipeline_without_reference_points(pipeline_env):
    """The headline requirement: any footage runs to completion, no homography."""
    from scout.config import get_settings
    from scout.db import Player, Rating, get_session
    from scout.pipeline import run

    video = _make_video(pipeline_env / "match.mp4")
    match_id = run(str(video))

    mdir = get_settings().match_dir(match_id)
    assert json.loads((mdir / "projection.json").read_text())["mode"] == "relative"
    for artifact in ("tracks_pitch.parquet", "ball_pitch.parquet", "events.parquet",
                     "roles.parquet", "metrics.parquet"):
        assert (mdir / artifact).exists(), f"missing {artifact}"

    with get_session() as db:
        players = db.query(Player).filter_by(match_id=match_id).all()
        ratings = db.query(Rating).filter_by(match_id=match_id).all()

    assert len(players) == N_PLAYERS
    assert ratings, "no ratings produced"
    assert all(0 <= r.overall <= 100 for r in ratings)
    assert all(r.note for r in ratings), "every player should get a scouting note"


def test_relative_mode_excludes_physical_metrics(pipeline_env):
    from scout.config import PITCH_ONLY_KPIS, get_settings
    from scout.pipeline import run

    video = _make_video(pipeline_env / "match.mp4")
    match_id = run(str(video))
    metrics = pd.read_parquet(get_settings().match_dir(match_id) / "metrics.parquet")

    for kpi in PITCH_ONLY_KPIS:
        assert metrics[kpi].isna().all(), f"{kpi} must not be invented without homography"
    assert metrics.passes_p90.notna().all(), "on-ball metrics must still be computed"


def test_every_player_gets_a_low_confidence_position(pipeline_env):
    from scout.db import Player, get_session
    from scout.pipeline import run

    run(str(_make_video(pipeline_env / "match.mp4")))
    with get_session() as db:
        players = db.query(Player).all()
        roles = [(p.role, p.role_confidence) for p in players]

    assert roles and all(r in {"GK", "DEF", "MID", "ATT"} for r, _ in roles)
    assert all(c <= 0.5 for _, c in roles), \
        "camera-relative positions must be flagged low-confidence for coach review"


def test_fragments_are_excluded_from_ratings(pipeline_env, monkeypatch):
    """Short-lived tracks are fragments, not children — they must not be rated."""
    import json

    from scout.config import get_settings
    from scout.db import Player, get_session
    from scout.pipeline import run

    monkeypatch.setenv("SCOUT_MIN_TRACK_SECONDS", "600")   # nothing can qualify
    get_settings.cache_clear()

    match_id = run(str(_make_video(pipeline_env / "match.mp4")))
    quality = json.loads((get_settings().match_dir(match_id) / "quality.json").read_text())

    assert quality["n_fragments_dropped"] > 0
    # a floor still applies so the coach sees the longest identities, not an empty page
    assert quality["n_rated"] == get_settings().min_rated_players
    with get_session() as db:
        assert db.query(Player).filter_by(match_id=match_id).count() == quality["n_rated"]


def test_quality_report_written(pipeline_env):
    import json

    from scout.config import get_settings
    from scout.pipeline import run

    match_id = run(str(_make_video(pipeline_env / "match.mp4")))
    q = json.loads((get_settings().match_dir(match_id) / "quality.json").read_text())
    assert q["n_raw_tracks"] == N_PLAYERS
    assert q["n_rated"] == N_PLAYERS
    assert q["n_fragments_dropped"] == 0


def test_rerun_is_idempotent(pipeline_env):
    """Second run skips completed stages and does not duplicate players."""
    from scout.db import Player, get_session
    from scout.pipeline import run

    video = _make_video(pipeline_env / "match.mp4")
    first = run(str(video))
    second = run(str(video))
    assert first == second
    with get_session() as db:
        assert db.query(Player).filter_by(match_id=first).count() == N_PLAYERS
