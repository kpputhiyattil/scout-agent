# ⚽ ScoutTrainer

AI football scouting for youth players: give it any match video (YouTube URL or file), it detects and tracks every kid, identifies them by jersey number, infers their position, and rates each one 0–100 **against the expectations of their own position** — attacker, midfielder, defender, or goalkeeper — with evidence-backed sub-scores and an AI-written scouting note for the coach.

Design: **CV turns pixels into facts, rules turn facts into scores, an LLM turns scores into words.**

## How it works

```
video ─► detect (YOLOv8) ─► track (ByteTrack) ─► teams (color clustering)
                                              ─► jerseys (OCR + temporal voting)
      ─► homography (pitch keypoints or 4 clicked points) ─► pitch coordinates
      ─► events (rule engine: possession, passes, shots, duels, saves)
      ─► roles (heatmap centroid: GK/DEF/MID/ATT)
      ─► metrics (role-specific KPIs, per-90)
      ─► ratings (squad-percentile x coach-editable weights.yaml)
      ─► dashboard (Streamlit) + LLM scouting notes (Claude, metric-grounded)
```

Ratings are transparent by design: every score comes with the raw KPI evidence, weights live in `weights.yaml` (edit + re-run rating stage only, no video reprocessing), and coach corrections (jersey/team/position) are stored as overrides — pipeline output is never mutated.

## Prerequisites

- **ffmpeg** — handled automatically: if no ffmpeg/ffprobe is on PATH, the pip-installed `static-ffmpeg` bundle is used (downloaded on first run). A system ffmpeg on PATH is used if present.
- **Deno or Node 22+** (required for YouTube) — yt-dlp solves YouTube JS challenges with a JS runtime; without one downloads fail. Installers: https://deno.com (recommended) or https://nodejs.org. Also install `yt-dlp[default]` so the EJS challenge scripts are present.
- NVIDIA GPU recommended for the perception stage (CPU works, ~6× slower)

## Quick start

```bash
# Option A: Docker (GPU)
docker compose up                      # dashboard at http://localhost:8501

# Option B: local install
pip install -e ".[all]"
streamlit run app/dashboard.py

# Process a match from CLI (drop your video into data/videos/ first)
python -m scout.pipeline --file data/videos/match.mp4 \
    --roster examples/roster_example.csv \
    --ref-points examples/refs_example.json

# Tests (analytics engine is fully unit-tested, no GPU needed)
pytest -q
```

## Inputs

- **Video** — two ways, in the dashboard sidebar or CLI:
  - **YouTube URL** — pasted directly; the pipeline downloads and normalizes it.
  - **Local folder** — drop files into `data/videos/` (or point the sidebar at any folder) and pick one from the list.
  Best results: elevated wide-angle view of the whole pitch.
- **Roster CSV** (optional) — `jersey_number,name,age` to map numbers to kids' names.
- **Pitch reference points** (optional, unlocks full metrics) — 4 clicked pixel→meter points (`examples/refs_example.json`); or drop a pitch-keypoint model at `data/models/pitch_keypoints.pt` for moving-camera homography. Without either, the pipeline runs in **camera-relative mode** (see below) instead of failing.
- **Detector weights** — auto-selected by hardware: `yolov8x.pt` (accurate) on GPU, `yolov8n.pt` (fast) on CPU. Override with `SCOUT_DETECTOR_WEIGHTS` — ideally a football-specific checkpoint (player/GK/ball/referee classes) for best results.

## Footage quality drives everything

Tracking quality, not model size, is what limits results. A tracker loses a player at every hard cut, fast pan or occlusion and issues a new id — an edited highlights reel produced **1239 track ids for ~12 children**, i.e. fragments of less than a second each, from which no honest rating can be computed.

Only players the coach can actually identify are rated. A score attached to "some child we tracked for four seconds" is worse than no score, so three filters run before rating:

| Filter | Setting | Why |
|---|---|---|
| Must have a jersey number | `SCOUT_REQUIRE_JERSEY_FOR_RATING` (default on) | a rating no one can attribute to a child is unusable |
| Must be observed ≥ 20 s | `SCOUT_MIN_TRACK_SECONDS` | shorter identities are tracking fragments |
| Squad-size cap per team | `SCOUT_MAX_PLAYERS_PER_TEAM` (default 16) | more than a squad means duplicate identities |

Jersey numbers are additionally bounded by `SCOUT_MAX_JERSEY_NUMBER` (default 30), and a roster CSV restricts OCR to numbers that actually exist in the squad — the single most effective accuracy improvement available. If no number is readable anywhere, the pipeline still rates the longest tracks rather than returning nothing, and says so.

The pipeline defends against fragmentation in three further ways:

- **Fragment merging** — tracks reading the same jersey number for the same team are stitched into one player (`identity.json → merge`). Jersey OCR is therefore not a nice-to-have: it is the identity anchor.
- **Fragment filtering** — identities observed for less than `SCOUT_MIN_TRACK_SECONDS` (default 20 s) are excluded from ratings rather than presented as children with four seconds of match time.
- **Quality reporting** — `quality.json` records raw tracks, jerseys read, players rated and fragments dropped; the dashboard shows these and warns loudly when fragmentation is severe.

Best footage: one continuous take, fixed elevated wide-angle camera, whole pitch in view, jersey numbers legible. Worst: edited highlights with cuts and zooms.

## Two measurement modes

Not every clip is a fixed wide-angle full-pitch recording, so the pipeline adapts instead of refusing to run:

| | **Pitch mode** | **Camera-relative mode** |
|---|---|---|
| Needs | reference points or keypoint model | nothing |
| Footage | fixed wide-angle, whole pitch visible | phone video, panning, highlights |
| Scale from | homography → true metres | player bbox height ≈ 1.45 m |
| Ratings use | all KPIs | on-ball KPIs only |
| Excluded | — | distance, speed, sprints, progressive carries, final-third touches, forward-pass %, shots on target, save % |
| Positions | heatmap centroid on pitch | rough guess from frame position — **coach should correct** |

In camera-relative mode unmeasurable KPIs are stored as NaN rather than as invented numbers; the rating engine drops them and renormalizes the remaining weights, so a score always reflects only what was actually observed. The dashboard shows a banner, and scouting notes are told not to comment on fitness or positioning.

## Exports

Every match produces two artifacts for sharing:

- **`ratings_<match>.csv`** — one row per player: identity, position, minutes, rating, sub-scores and every raw KPI.
- **`scouting_report_<match>.docx`** — coach-facing Word summary: what the footage could and couldn't measure, standouts by position, squad table (top 30) and per-player notes.

```bash
python -m scout.report.export --latest              # or --match <id>
python -m scout.report.export --match <id> --out reports/
```

Also available as download buttons in the dashboard, and as a cell in the Colab notebook. Players without a roster name are labelled by team and number (`B #14`), which a coach can match to a child on the pitch.

## Configuration

Environment variables (or `.env`):

```
SCOUT_DEVICE=cuda                # cuda | cpu
SCOUT_ANTHROPIC_API_KEY=...      # optional; template notes if unset
SCOUT_DETECTOR_WEIGHTS=...       # optional override (default: auto — yolov8x on GPU, yolov8n on CPU)
```

## Privacy

This system analyzes footage of minors. All video and names stay on the local machine; only anonymized numeric metrics (jersey number, no name) are sent to the LLM API for the scouting note, and that step is optional.

## Project layout

```
scout/perception/   detection, tracking, team + jersey ID, homography
scout/analytics/    event rules, role inference, metrics, rating engine
scout/report/       LLM scouting notes, highlight clip cutting
scout/pipeline.py   idempotent, resumable stage orchestrator
app/dashboard.py    Streamlit coach UI (YouTube URL or local-folder video source)
data/videos/        drop local match videos here — the dashboard lists them
weights.yaml        coach-editable rating weights per role
tests/              synthetic-scenario unit tests (20 tests)
```
