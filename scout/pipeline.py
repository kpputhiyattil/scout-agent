"""Stage orchestrator: idempotent, resumable, keyed by match_id.

Stages: ingest -> perceive -> identify -> project -> analyze -> rate -> report
Each stage writes its artifact; a completed stage is skipped unless --force.
Analytics stages never re-open the video — they run from stored tables.

CLI:
  python -m scout.pipeline --url "https://youtube.com/..." --roster roster.csv
  python -m scout.pipeline --file match.mp4 --ref-points refs.json
"""
from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

from scout.config import get_settings, load_weights
from scout.db import Event, JobStatus, Match, Player, Rating, get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("pipeline")

STAGES = ["ingest", "perceive", "identify", "project", "analyze", "rate", "report"]


def _set_status(match_id: str, stage: str, status: str, detail: str = "") -> None:
    with get_session() as db:
        row = (db.query(JobStatus).filter_by(match_id=match_id, stage=stage).first()
               or JobStatus(match_id=match_id, stage=stage))
        row.status, row.detail = status, detail[:2000]
        db.add(row)
        db.commit()


def _pct_reporter(match_id: str, stage: str):
    """Throttled progress callback: writes 'NN%' to JobStatus.detail on whole-percent change."""
    last = {"p": -1}

    def cb(frac: float) -> None:
        p = int(max(0.0, min(1.0, frac)) * 100)
        if p != last["p"]:
            last["p"] = p
            _set_status(match_id, stage, "running", f"{p}%")
    return cb


def _stage_done(match_id: str, stage: str) -> bool:
    with get_session() as db:
        row = db.query(JobStatus).filter_by(match_id=match_id, stage=stage, status="done").first()
        return row is not None


def run(source: str, roster: str | None = None, ref_points: str | None = None,
        force: bool = False) -> str:
    s = get_settings()

    # ---- ingest ----
    from scout.ingest import ingest, match_id_for
    match_id = match_id_for(source)
    _set_status(match_id, "ingest", "running", "0%")
    try:
        ingest(source, progress=_pct_reporter(match_id, "ingest"))
    except Exception as e:
        _set_status(match_id, "ingest", "failed", f"{e}\n{traceback.format_exc()}")
        raise
    _set_status(match_id, "ingest", "done")
    mdir = s.match_dir(match_id)
    video = mdir / "video.mp4"

    with get_session() as db:
        fps = db.get(Match, match_id).fps

    def stage(name):
        def deco(fn):
            def wrapper():
                if _stage_done(match_id, name) and not force:
                    log.info("skip %s (done)", name)
                    return
                _set_status(match_id, name, "running")
                try:
                    fn()
                    _set_status(match_id, name, "done")
                except Exception as e:
                    _set_status(match_id, name, "failed", f"{e}\n{traceback.format_exc()}")
                    raise
            return wrapper
        return deco

    # ---- perceive: detection + tracking ----
    @stage("perceive")
    def perceive():
        from scout.perception.detect import run_detection_tracking
        run_detection_tracking(video, mdir, progress=_pct_reporter(match_id, "perceive"))

    # ---- identify: teams + jerseys ----
    @stage("identify")
    def identify():
        from scout.perception.jersey import join_roster, read_jerseys
        from scout.perception.team import assign_teams
        tracks = pd.read_parquet(mdir / "tracks.parquet")
        teams, separation = assign_teams(video, tracks)
        jerseys = read_jerseys(video, tracks)
        names = join_roster(jerseys, roster)
        (mdir / "identity.json").write_text(json.dumps({
            "teams": {str(k): v for k, v in teams.items()},
            "jerseys": {str(k): v for k, v in jerseys.items()},
            "names": {str(k): v for k, v in names.items()},
            "team_separation": separation,
        }))
        if separation < 0.3:
            log.warning("Kit colors similar (separation=%.2f) — confirm teams in dashboard", separation)

    # ---- project: homography -> pitch coordinates ----
    @stage("project")
    def project():
        from scout.perception import pitch
        if ref_points:
            proj = pitch.from_reference_points(ref_points)
        else:
            kp_weights = mdir.parent.parent / "models" / "pitch_keypoints.pt"
            if kp_weights.exists():
                proj = pitch.from_keypoint_model(video, str(kp_weights),
                                                 every_n=s.homography_every_n_frames)
            else:
                raise RuntimeError(
                    "No homography source: supply --ref-points refs.json (4 clicked points) "
                    "or place a pitch keypoint model at data/models/pitch_keypoints.pt")
        tracks = pd.read_parquet(mdir / "tracks.parquet")
        tracks["cx"] = (tracks.x1 + tracks.x2) / 2
        tracks["cy"] = tracks.y2  # feet position, not bbox center
        proj.project_df(tracks, "cx", "cy").to_parquet(mdir / "tracks_pitch.parquet", index=False)
        ball = pd.read_parquet(mdir / "ball.parquet")
        proj.project_df(ball, "x", "y").to_parquet(mdir / "ball_pitch.parquet", index=False)

    # ---- analyze: events + roles ----
    @stage("analyze")
    def analyze():
        from scout.analytics import events as E
        from scout.analytics.positions import infer_roles
        identity = json.loads((mdir / "identity.json").read_text())
        tracks = pd.read_parquet(mdir / "tracks_pitch.parquet").dropna(subset=["x_m", "y_m"])
        tracks["team"] = tracks.track_id.astype(str).map(identity["teams"]).fillna("?")
        gk_det = set(tracks[tracks.cls == "goalkeeper"].track_id.unique())

        ball = pd.read_parquet(mdir / "ball_pitch.parquet")
        n_frames = int(tracks.frame.max()) + 1
        ball = E.interpolate_ball(ball[["frame", "x_m", "y_m"]], n_frames,
                                  s.ball_gap_interp_max_frames)

        poss = E.compute_possession(tracks, ball)
        spells = E.possession_spells(poss)

        # provisional roles for GK identification, then attack direction, then final roles
        prov_dir = {"A": +1, "B": -1}
        prov_roles = infer_roles(tracks, prov_dir, gk_det)
        gk_by_team = {}
        for r in prov_roles[prov_roles.role == "GK"].itertuples():
            team = identity["teams"].get(str(r.track_id), "?")
            if team in ("A", "B"):
                gk_by_team[team] = r.track_id
        attack_dir = E.infer_attack_direction(tracks, gk_by_team) or prov_dir
        roles = infer_roles(tracks, attack_dir, gk_det)
        gk_tracks = set(roles[roles.role == "GK"].track_id)

        trans = E.detect_transitions(spells, fps)
        shots = E.detect_shots(ball, poss, fps, attack_dir)
        saves = E.detect_saves(shots, spells, fps, gk_tracks)
        duels = E.detect_duels(spells, tracks, fps)

        all_events = pd.concat([trans, shots, saves, duels], ignore_index=True)
        all_events.to_parquet(mdir / "events.parquet", index=False)
        roles.to_parquet(mdir / "roles.parquet", index=False)
        (mdir / "attack_dir.json").write_text(json.dumps(attack_dir))

    # ---- rate: metrics + ratings -> DB ----
    @stage("rate")
    def rate():
        from scout.analytics.metrics import compute_metrics
        from scout.analytics.rating import rate_players
        identity = json.loads((mdir / "identity.json").read_text())
        attack_dir = json.loads((mdir / "attack_dir.json").read_text())
        tracks = pd.read_parquet(mdir / "tracks_pitch.parquet").dropna(subset=["x_m", "y_m"])
        tracks["team"] = tracks.track_id.astype(str).map(identity["teams"]).fillna("?")
        events = pd.read_parquet(mdir / "events.parquet")
        roles = pd.read_parquet(mdir / "roles.parquet")

        metrics = compute_metrics(events, tracks, fps, attack_dir, roles)
        metrics.to_parquet(mdir / "metrics.parquet", index=False)
        ratings = rate_players(metrics, load_weights())

        with get_session() as db:
            db.query(Rating).filter_by(match_id=match_id).delete()
            db.query(Event).filter_by(match_id=match_id).delete()
            db.query(Player).filter_by(match_id=match_id).delete()
            pid_of = {}
            role_conf = roles.set_index("track_id")["confidence"]
            for _, r in metrics.iterrows():
                tid = int(r.track_id)
                p = Player(match_id=match_id, track_id=tid, team=r.team,
                           jersey=identity["jerseys"].get(str(tid)),
                           name=identity["names"].get(str(tid),
                                f"Unknown #{identity['jerseys'].get(str(tid), '?')}"),
                           role=r.role, role_confidence=float(role_conf.get(tid, 0)),
                           minutes=float(r.minutes))
                db.add(p)
                db.flush()
                pid_of[tid] = p.id
            for _, r in ratings.iterrows():
                db.add(Rating(match_id=match_id, player_id=pid_of[int(r.track_id)],
                              role=r.role, overall=float(r.overall),
                              sub_scores=r.sub_scores, evidence=r.evidence))
            actor_col = events.get("actor")
            if actor_col is not None:
                for _, e in events.iterrows():
                    actor = e.get("actor") if pd.notna(e.get("actor")) else e.get("winner")
                    db.add(Event(match_id=match_id, t=float(e.frame) / fps, type=e.type,
                                 player_id=pid_of.get(int(actor)) if pd.notna(actor) else None,
                                 success=int(e.get("success", 1)) if pd.notna(e.get("success", 1)) else 1))
            db.commit()

    # ---- report: LLM notes + highlight clips ----
    @stage("report")
    def report():
        from scout.report.clips import cut_highlights
        from scout.report.llm import scouting_note
        events = pd.read_parquet(mdir / "events.parquet")
        with get_session() as db:
            for rating in db.query(Rating).filter_by(match_id=match_id).all():
                player = db.get(Player, rating.player_id)
                rating.note = scouting_note(
                    {"role": rating.role, "overall": rating.overall,
                     "sub_scores": rating.sub_scores, "evidence": rating.evidence,
                     "low_sample": player.minutes < 15},
                    jersey=player.jersey or "?")
                cut_highlights(video, events, fps, mdir / "clips", player.track_id)
            db.commit()

    for fn in (perceive, identify, project, analyze, rate, report):
        fn()
    log.info("Match %s complete", match_id)
    return match_id


def main():
    ap = argparse.ArgumentParser(description="ScoutTrainer pipeline")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="YouTube (or other) video URL")
    g.add_argument("--file", help="Local video file")
    ap.add_argument("--roster", help="CSV with jersey_number,name[,age]")
    ap.add_argument("--ref-points", help="JSON with 4+ pixel->pitch reference points")
    ap.add_argument("--force", action="store_true", help="Re-run completed stages")
    a = ap.parse_args()
    run(a.url or a.file, roster=a.roster, ref_points=a.ref_points, force=a.force)


if __name__ == "__main__":
    main()
