"""Camera-relative mode: no homography available.

Covers the three behaviours that keep ratings honest on phone/highlight footage:
scale estimation from bounding boxes, roles inferred from frame position, and
unmeasurable KPIs being dropped (not invented) by the rating engine.
"""
import numpy as np
import pandas as pd
import pytest

from scout.analytics.metrics import compute_metrics
from scout.analytics.positions import infer_roles_relative
from scout.analytics.rating import rate_players
from scout.config import PITCH_ONLY_KPIS
from scout.perception.pitch import from_pixel_scale

PLAYER_H_M = 1.45


def _boxes(frame_heights: dict[int, list[float]]) -> pd.DataFrame:
    """frame -> list of bbox pixel heights."""
    rows = []
    for f, hs in frame_heights.items():
        for i, h in enumerate(hs):
            rows.append((f, i, 100.0, 200.0, 120.0, 200.0 + h))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])


class TestPixelScale:
    def test_scale_matches_player_height(self):
        # 100px-tall players => 1.45m / 100px = 0.0145 m per pixel
        proj = from_pixel_scale(_boxes({0: [100.0, 100.0]}), PLAYER_H_M)
        assert proj.scale_at(0) == pytest.approx(PLAYER_H_M / 100)

    def test_zoom_in_shrinks_metres_per_pixel(self):
        # players appear bigger (camera closer) => each pixel spans less ground
        proj = from_pixel_scale(_boxes({0: [50.0], 10: [200.0]}), PLAYER_H_M)
        assert proj.scale_at(10) < proj.scale_at(0)

    def test_projection_converts_pixels_to_metres(self):
        proj = from_pixel_scale(_boxes({0: [100.0]}), PLAYER_H_M)
        df = pd.DataFrame({"frame": [0], "x": [200.0], "y": [400.0]})
        out = proj.project_df(df, "x", "y")
        assert out.x_m.iat[0] == pytest.approx(200 * PLAYER_H_M / 100)
        assert out.y_m.iat[0] == pytest.approx(400 * PLAYER_H_M / 100)

    def test_unknown_frame_falls_back_to_median(self):
        proj = from_pixel_scale(_boxes({0: [100.0], 1: [100.0]}), PLAYER_H_M)
        assert proj.scale_at(999) == pytest.approx(proj.median)

    def test_mode_flag(self):
        assert from_pixel_scale(_boxes({0: [100.0]}), PLAYER_H_M).mode == "relative"


class TestRelativeRoles:
    def _tracks(self):
        # team A spread across the frame: deep, middle, advanced
        rows = []
        for tid, x in [(1, 10.0), (2, 50.0), (3, 90.0)]:
            for f in range(10):
                rows.append((f, tid, "A", x, 30.0))
        return pd.DataFrame(rows, columns=["frame", "track_id", "team", "x_m", "y_m"])

    def test_orders_players_across_the_frame(self):
        roles = infer_roles_relative(self._tracks()).set_index("track_id")["role"]
        assert roles[1] == "DEF" and roles[2] == "MID" and roles[3] == "ATT"

    def test_confidence_stays_low(self):
        # a panning camera makes this a hint, never a measurement
        roles = infer_roles_relative(self._tracks())
        assert (roles.confidence <= 0.5).all()

    def test_detector_goalkeeper_respected(self):
        roles = infer_roles_relative(self._tracks(), detector_gk={1}).set_index("track_id")
        assert roles.loc[1, "role"] == "GK"


class TestRelativeMetricsAndRating:
    def _inputs(self):
        rows, events = [], []
        for tid in range(1, 9):
            for f in range(0, 250, 10):
                rows.append((f, tid, "A" if tid <= 4 else "B", 10.0 + tid, 20.0))
            events.append((tid * 3, "pass", tid, (tid % 8) + 1))
        tracks = pd.DataFrame(rows, columns=["frame", "track_id", "team", "x_m", "y_m"])
        ev = pd.DataFrame(events, columns=["frame", "type", "actor", "target"])
        roles = pd.DataFrame({"track_id": list(range(1, 9)),
                              "role": ["DEF", "MID", "ATT", "GK"] * 2,
                              "confidence": [0.4] * 8})
        return ev, tracks, roles

    def test_pitch_only_kpis_are_nan_not_zero(self):
        ev, tracks, roles = self._inputs()
        m = compute_metrics(ev, tracks, 25.0, {}, roles, mode="relative")
        for kpi in PITCH_ONLY_KPIS:
            assert m[kpi].isna().all(), f"{kpi} should be unmeasurable in relative mode"

    def test_on_ball_kpis_still_computed(self):
        ev, tracks, roles = self._inputs()
        m = compute_metrics(ev, tracks, 25.0, {}, roles, mode="relative")
        assert m.passes_p90.notna().all()
        assert m.pass_completion_pct.notna().all()

    def test_pitch_mode_keeps_physical_kpis(self):
        ev, tracks, roles = self._inputs()
        m = compute_metrics(ev, tracks, 25.0, {"A": 1, "B": -1}, roles, mode="pitch")
        assert m.distance_km_p90.notna().any()

    def test_rating_drops_unavailable_kpis(self):
        ev, tracks, roles = self._inputs()
        m = compute_metrics(ev, tracks, 25.0, {}, roles, mode="relative")
        weights = {"MID": {"passes_p90": {"weight": 0.5, "group": "technical"},
                           "distance_km_p90": {"weight": 0.5, "group": "physical"}}}
        out = rate_players(m, weights)
        assert not out.empty
        # physical KPI was all-NaN: excluded from evidence and from the sub-scores
        assert all("distance_km_p90" not in e for e in out.evidence)
        assert all("physical" not in s for s in out.sub_scores)
        assert out.overall.between(0, 100).all()

    def test_rating_still_produced_when_only_one_kpi_survives(self):
        ev, tracks, roles = self._inputs()
        m = compute_metrics(ev, tracks, 25.0, {}, roles, mode="relative")
        weights = {"MID": {"top_speed_kmh": {"weight": 1.0, "group": "physical"},
                           "duel_win_pct": {"weight": 1.0, "group": "defending"}}}
        out = rate_players(m, weights)
        assert out.overall.notna().all()

    def test_shots_without_attack_direction_have_unknown_on_target(self):
        from scout.analytics.events import detect_shots
        ball = pd.DataFrame({"frame": range(6),
                             "x_m": [0, 0.1, 0.2, 5, 10, 15], "y_m": [0.0] * 6})
        poss = pd.DataFrame({"frame": range(6), "owner": [1] * 6, "team": ["A"] * 6})
        shots = detect_shots(ball, poss, 25.0, attack_dir=None)
        assert len(shots) >= 1
        assert shots.on_target.isna().all()
