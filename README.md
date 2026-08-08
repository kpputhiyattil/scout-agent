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
- **Pitch reference points** (recommended for MVP) — 4 clicked pixel→meter points (`examples/refs_example.json`); or drop a pitch-keypoint model at `data/models/pitch_keypoints.pt` for moving-camera homography.
- **Detector weights** — auto-selected by hardware: `yolov8x.pt` (accurate) on GPU, `yolov8n.pt` (fast) on CPU. Override with `SCOUT_DETECTOR_WEIGHTS` — ideally a football-specific checkpoint (player/GK/ball/referee classes) for best results.

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
