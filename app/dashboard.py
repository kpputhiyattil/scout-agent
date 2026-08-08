"""Coach dashboard. Run: streamlit run app/dashboard.py"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

from scout.config import get_settings
from scout.db import JobStatus, Match, Override, Player, Rating, apply_overrides, get_session
from scout.ingest import match_id_for

st.set_page_config(page_title="ScoutTrainer", page_icon="⚽", layout="wide")
s = get_settings()

ROLE_LABEL = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "ATT": "Attacker"}


def _run_pipeline(source, roster, ref_points):
    from scout.pipeline import run
    try:
        run(source, roster=roster, ref_points=ref_points)
    except Exception:
        pass  # status table already records failure detail


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
STAGE_LABEL = {"ingest": "downloading / normalizing video", "perceive": "detecting & tracking players",
               "identify": "reading jerseys & teams", "project": "mapping pitch coordinates",
               "analyze": "detecting events & positions", "rate": "computing ratings",
               "report": "writing scouting notes & clips"}


def _request_start():
    """URL field on_change: pressing Enter starts the analysis."""
    st.session_state["start_requested"] = True


# ---------- sidebar: new match ----------
with st.sidebar:
    st.title("⚽ ScoutTrainer")
    st.subheader("Analyze a match")

    mode = st.radio("Video source", ["YouTube URL", "Local folder"], horizontal=True)
    source = None
    if mode == "YouTube URL":
        url = st.text_input("Video URL (YouTube etc.)", key="yt_url",
                            placeholder="https://youtube.com/watch?v=...",
                            on_change=_request_start,
                            help="Press Enter or click Start analysis")
        source = url.strip() or None
    else:
        folder = Path(st.text_input("Videos folder", value=str(s.videos_dir)))
        if folder.is_dir():
            videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
            if videos:
                pick = st.selectbox("Video file", [v.name for v in videos])
                source = str(folder / pick)
            else:
                st.caption(f"No videos found — drop a match video into `{folder}`.")
        else:
            st.caption("Folder not found.")

    roster_up = st.file_uploader("Roster CSV (jersey_number,name)", type=["csv"])
    refs_up = st.file_uploader("Pitch reference points JSON (optional)", type=["json"])

    clicked = st.button("Start analysis", type="primary", disabled=not source)
    enter_pressed = st.session_state.pop("start_requested", False)
    started = st.session_state.setdefault("started_sources", set())

    if (clicked or enter_pressed) and source:
        with get_session() as db:
            has_failed = db.query(JobStatus).filter_by(
                match_id=match_id_for(source), status="failed").first() is not None
        if source in started and not has_failed:
            st.info("This video is already being processed — progress shows below.")
        else:
            updir = s.data_dir / "uploads"
            updir.mkdir(parents=True, exist_ok=True)
            roster = None
            if roster_up:
                rp = updir / "roster.csv"
                rp.write_bytes(roster_up.getvalue())
                roster = str(rp)
            refs = None
            if refs_up:
                fp = updir / "refs.json"
                fp.write_bytes(refs_up.getvalue())
                refs = str(fp)
            threading.Thread(target=_run_pipeline, args=(source, roster, refs), daemon=True).start()
            started.add(source)
            st.success("Processing started — streaming/downloading the video now. "
                       "Progress below updates automatically.")

# ---------- main: match browser ----------
with get_session() as db:
    matches = db.query(Match).order_by(Match.created_at.desc()).all()
    match_opts = {f"{m.id} — {m.source[:60]}": m.id for m in matches}
    match_src = {m.id: m.source for m in matches}

# ---------- job progress (main screen, auto-refreshing) ----------
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
st.session_state.setdefault("_n_matches", len(match_opts))


def _job_progress():
    # only show jobs started from this dashboard session — old runs stay out of view
    session_mids = {match_id_for(src) for src in st.session_state.get("started_sources", set())}
    if not session_mids:
        return
    with get_session() as db:
        jobs = pd.read_sql(db.query(JobStatus).statement, db.bind)
        n_matches = db.query(Match).count()
    if n_matches > st.session_state["_n_matches"]:
        # a new match landed while processing — refresh the whole page
        st.session_state["_n_matches"] = n_matches
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()
    jobs = jobs[jobs.match_id.isin(session_mids)] if not jobs.empty else jobs
    if jobs.empty:
        st.info("⏳ Starting up — streaming/downloading the video…")
        return
    running = jobs[jobs.status == "running"]
    failed = jobs[jobs.status == "failed"]
    if running.empty and failed.empty and match_opts:
        return  # nothing in flight — keep the main screen clean
    st.subheader("Job progress")
    for _, j in running.iterrows():
        done_stages = int((jobs[jobs.match_id == j["match_id"]].status == "done").sum())
        label = f"stage {done_stages + 1}/7 — {j['stage']}: {STAGE_LABEL.get(j['stage'], 'running')}"
        m = re.match(r"(\d+)%", str(j.get("detail") or ""))
        if m:
            pct = int(m.group(1))
            st.progress(pct / 100, text=f"⏳ {label} — {pct}%")
        else:
            st.info(f"⏳ {label}…")
    for _, j in failed.iterrows():
        st.error(f"❌ {j['stage']} failed: {str(j['detail']).splitlines()[0][:200]}")
    st.dataframe(jobs[["match_id", "stage", "status"]], hide_index=True, height=220)


if _fragment:
    _fragment(run_every=3)(_job_progress)()
else:
    _job_progress()
    if st.button("Refresh"):
        st.rerun()

if not match_opts:
    if not st.session_state.get("started_sources"):
        st.info("No matches yet. Paste a YouTube URL or pick a video from your folder in the sidebar.")
    st.stop()

mid = match_opts[st.selectbox("Match", list(match_opts))]
mdir = s.match_dir(mid)

with get_session() as db:
    players = [apply_overrides(db, p) for p in db.query(Player).filter_by(match_id=mid).all()]
    ratings = {r.player_id: r for r in db.query(Rating).filter_by(match_id=mid).all()}

if not players:
    src = match_src[mid]
    if src in st.session_state.get("started_sources", set()):
        st.info("Analysis in progress — the Job progress panel above updates automatically.")
    else:
        st.info("The video is ingested but not analyzed yet.")
        if st.button("▶ Start / resume analysis", type="primary"):
            threading.Thread(target=_run_pipeline, args=(src, None, None), daemon=True).start()
            st.session_state.setdefault("started_sources", set()).add(src)
            st.rerun()
    vid = mdir / "video.mp4"
    if vid.exists():
        st.caption("Match video")
        st.video(str(vid))
    st.stop()

if (mdir / "projection.json").exists():
    if json.loads((mdir / "projection.json").read_text()).get("mode") == "relative":
        st.warning(
            "**Camera-relative mode** — this footage has no usable pitch reference "
            "(moving camera or partial pitch view), so ratings use event and technical "
            "KPIs only: touches, passing, duels, possession won/lost, shots. Distance, "
            "speed and pitch-position metrics are excluded, and positions are a rough "
            "guess — correct them below. For full metrics, analyze fixed wide-angle "
            "footage with pitch reference points.")

if (mdir / "quality.json").exists():
    q = json.loads((mdir / "quality.json").read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked identities", q["n_raw_tracks"])
    c2.metric("Jersey numbers read", q["n_jerseys_read"])
    c3.metric("Players rated", q["n_rated"])
    c4.metric("Median minutes", round(q["median_minutes"], 1))
    if q["n_raw_tracks"] > 4 * max(q["n_rated"], 1):
        st.error(
            f"**Heavy identity fragmentation** — {q['n_raw_tracks']} separate tracks were "
            f"created for ~{q['n_rated']} players. Camera cuts, panning and occlusion make the "
            "tracker lose people, so one child becomes many short tracks. Ratings here are "
            "unreliable. Fixes, in order of impact: use continuous footage (no edited "
            "highlights), a fixed wide-angle camera, and a roster CSV so jersey numbers can "
            "stitch fragments back together.")
    elif q["n_jerseys_read"] < q["n_rated"]:
        st.info("Some players have no jersey number — OCR needs numbers to be visible and "
                "reasonably large. Assign them manually in Coach corrections below.")

identity = {}
if (mdir / "identity.json").exists():
    identity = json.loads((mdir / "identity.json").read_text())
    if identity.get("team_separation", 1) < 0.3:
        st.warning("Kit colors look similar — team assignment may need manual correction below.")

# squad table
rows = []
for p in players:
    r = ratings.get(p.id)
    rows.append({"Player": p.name, "Jersey": p.jersey or "?", "Team": p.team,
                 "Position": ROLE_LABEL.get(p.role, p.role),
                 "Pos. confidence": round(p.role_confidence or 0, 2),
                 "Minutes": round(p.minutes),
                 "Rating": r.overall if r else None,
                 **({g: v for g, v in (r.sub_scores or {}).items()} if r else {})})
df = pd.DataFrame(rows).sort_values("Rating", ascending=False, na_position="last")

# best player per position
st.subheader("Best by position")
best_cols = st.columns(len(ROLE_LABEL))
for col, (role, label) in zip(best_cols, ROLE_LABEL.items()):
    cands = [(p, ratings[p.id]) for p in players
             if p.role == role and p.id in ratings and ratings[p.id].overall is not None]
    with col:
        st.caption(label)
        if cands:
            bp, br = max(cands, key=lambda x: x[1].overall)
            st.metric(f"{bp.name} (#{bp.jersey or '?'})", round(br.overall))
        else:
            st.metric("—", "—")

st.subheader("Squad ratings")
st.dataframe(df, hide_index=True, use_container_width=True)
st.download_button("Export CSV", df.to_csv(index=False), f"ratings_{mid}.csv")

# ---------- player detail ----------
st.divider()
sel = st.selectbox("Player detail", [f"{p.name} (#{p.jersey or '?'})" for p in players])
p = players[[f"{q.name} (#{q.jersey or '?'})" for q in players].index(sel)]
r = ratings.get(p.id)

c1, c2 = st.columns([1, 1])
with c1:
    st.metric(f"{ROLE_LABEL.get(p.role, p.role)} rating", r.overall if r else "—")
    if r and r.sub_scores:
        try:
            import plotly.graph_objects as go
            cats = list(r.sub_scores)
            fig = go.Figure(go.Scatterpolar(r=[r.sub_scores[c] for c in cats] + [r.sub_scores[cats[0]]],
                                            theta=cats + [cats[0]], fill="toself"))
            fig.update_layout(polar={"radialaxis": {"range": [0, 100]}}, height=350,
                              margin={"l": 40, "r": 40, "t": 30, "b": 30})
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.json(r.sub_scores)
    if r and r.note:
        st.markdown(f"**Scouting note**\n\n{r.note}")
    if r and r.evidence:
        with st.expander("Evidence (raw metrics)"):
            st.json(r.evidence)

with c2:
    # heatmap from pitch tracks
    tp = mdir / "tracks_pitch.parquet"
    if tp.exists():
        t = pd.read_parquet(tp)
        t = t[t.track_id == p.track_id].dropna(subset=["x_m", "y_m"])
        if not t.empty:
            try:
                from mplsoccer import Pitch
                pitch = Pitch(pitch_type="custom", pitch_length=100, pitch_width=64,
                              line_color="grey")
                fig, ax = pitch.draw(figsize=(6, 4))
                pitch.kdeplot(t.x_m, t.y_m, ax=ax, fill=True, cmap="Reds", levels=50)
                st.pyplot(fig)
            except ImportError:
                st.scatter_chart(t.rename(columns={"x_m": "x", "y_m": "y"})[["x", "y"]], x="x", y="y")
    clips = sorted((mdir / "clips").glob(f"p{p.track_id}_*.mp4")) if (mdir / "clips").exists() else []
    if clips:
        st.caption("Key moments")
        for c in clips[:3]:
            st.video(str(c))

# ---------- corrections ----------
st.divider()
with st.expander("Coach corrections (jersey / name / team / position)"):
    f1, f2, f3, f4 = st.columns(4)
    new_jersey = f1.text_input("Jersey", value=str(p.jersey or ""))
    new_name = f2.text_input("Name", value=p.name)
    new_team = f3.selectbox("Team", ["A", "B"], index=0 if p.team == "A" else 1)
    new_role = f4.selectbox("Position", list(ROLE_LABEL),
                            index=list(ROLE_LABEL).index(p.role) if p.role in ROLE_LABEL else 1)
    if st.button("Save corrections"):
        with get_session() as db:
            for field, old, new in [("jersey", str(p.jersey or ""), new_jersey),
                                    ("name", p.name, new_name),
                                    ("team", p.team, new_team), ("role", p.role, new_role)]:
                if new and new != old:
                    db.add(Override(player_id=p.id, field=field, value=new))
            db.commit()
        st.success("Saved. Re-run the rating stage to refresh scores: "
                   f"`python -m scout.pipeline --file {mdir / 'video.mp4'} --force`")
