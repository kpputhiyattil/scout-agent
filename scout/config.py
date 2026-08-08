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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCOUT_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    device: str = "cuda"  # "cuda" | "cpu"

    # Detection
    detector_weights: str = "yolov8x.pt"  # COCO fallback (person + sports ball).
    # Set to a football-specific checkpoint (players/GK/ball/referee) for best results.
    detector_conf: float = 0.3
    detect_batch_size: int = 16
    target_fps: int = 25
    target_height: int = 720

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

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5"

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
