"""Central configuration: paths, device, pitch geometry, rating weights."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard youth 11v11 pitch (meters). Coordinates are normalized to this.
PITCH_LENGTH_M = 100.0
PITCH_WIDTH_M = 64.0
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.3

# Physical sanity bounds for kids
MAX_PLAUSIBLE_SPEED_KMH = 30.0
SPRINT_SPEED_KMH = 18.0

# KPIs that require true pitch coordinates. In camera-relative mode these are
# suppressed (NaN) and the rating engine renormalizes the remaining weights.
PITCH_ONLY_KPIS = (
    "distance_km_p90", "top_speed_kmh", "sprints_p90",
    "progressive_carries_p90", "final_third_touches_p90", "forward_pass_pct",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCOUT_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    device: str = "cuda"  # "cuda" | "cpu"

    # Detection
    detector_weights: str = ""  # empty = auto: yolov8x on GPU, yolov8n on CPU.
    # Set explicitly to override — ideally a football-specific checkpoint
    # (players/GK/ball/referee classes) for best results.
    detector_conf: float = 0.3
    detect_batch_size: int = 16
    target_fps: int = 25
    target_height: int = 720

    # Camera-relative fallback (no homography): typical player height, used to
    # convert pixels to approximate metres from bounding-box size.
    player_height_m: float = 1.45

    # A tracked identity shorter than this is a fragment, not a player: trackers
    # lose people behind occlusions and camera cuts. Fragments are excluded from
    # ratings rather than presented as children with 4 seconds of match time.
    min_track_seconds: float = 20.0
    min_rated_players: int = 4   # if the filter leaves fewer, keep the longest instead

    # Sampling
    ocr_sample_every_n_frames: int = 12   # ~2x/sec at 25fps
    homography_every_n_frames: int = 30

    # Event rules (meters / seconds)
    possession_radius_m: float = 2.5
    possession_hysteresis_frames: int = 3
    ball_gap_interp_max_frames: int = 25
    shot_speed_ms: float = 8.0
    save_window_s: float = 1.5
    duel_radius_m: float = 2.0

    # YouTube download auth (needed on cloud/datacenter IPs like Colab, which YouTube
    # challenges with "Sign in to confirm you're not a bot")
    ytdlp_cookies: str = ""               # path to a cookies.txt file
    ytdlp_cookies_from_browser: str = ""  # e.g. "chrome", "firefox", "edge"

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5"

    def resolve_device(self) -> str:
        """Requested device, downgraded to cpu when CUDA isn't actually available."""
        if self.device == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return self.device
        except ImportError:
            pass
        import logging
        logging.getLogger("scout").warning(
            "CUDA requested but not available — falling back to CPU (slower). "
            "Set SCOUT_DEVICE=cpu to silence this.")
        return "cpu"

    def resolve_detector_weights(self) -> str:
        """Explicit weights if set, else pick by hardware: yolov8x (GPU) / yolov8n (CPU)."""
        if self.detector_weights:
            return self.detector_weights
        return "yolov8x.pt" if self.resolve_device() == "cuda" else "yolov8n.pt"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'scout.db'}"

    @property
    def videos_dir(self) -> Path:
        """Folder the dashboard scans for local match videos."""
        d = self.data_dir / "videos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def match_dir(self, match_id: str) -> Path:
        d = self.data_dir / "matches" / match_id
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


@lru_cache
def load_weights(path: str | None = None) -> dict:
    """Load role->KPI->{weight, group} rating weights. Coach-editable YAML."""
    p = Path(path) if path else PROJECT_ROOT / "weights.yaml"
    with open(p, encoding="utf-8") as f:
        w = yaml.safe_load(f)
    for role, kpis in w.items():
        for kpi, spec in kpis.items():
            if "weight" not in spec or "group" not in spec:
                raise ValueError(f"weights.yaml: {role}.{kpi} needs 'weight' and 'group'")
    return w
