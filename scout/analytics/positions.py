"""Role inference: GK / DEF / MID / ATT per player for the match.

GK: detector class (if football model) OR >60% dwell time in own penalty box.
Outfield: heatmap centroid along the team's attack axis, split into terciles.
Youth formations are fluid — low-confidence assignments are flagged and the
coach can override in the dashboard.
"""
from __future__ import annotations

import pandas as pd

from scout.config import PENALTY_BOX_DEPTH_M, PENALTY_BOX_WIDTH_M, PITCH_LENGTH_M, PITCH_WIDTH_M


def _in_own_box(x: pd.Series, y: pd.Series, attack_dir: int) -> pd.Series:
    y_ok = ((PITCH_WIDTH_M - PENALTY_BOX_WIDTH_M) / 2 <= y) & (y <= (PITCH_WIDTH_M + PENALTY_BOX_WIDTH_M) / 2)
    x_ok = (x <= PENALTY_BOX_DEPTH_M) if attack_dir > 0 else (x >= PITCH_LENGTH_M - PENALTY_BOX_DEPTH_M)
    return x_ok & y_ok


def infer_roles_relative(tracks: pd.DataFrame,
                         detector_gk: set[int] | None = None) -> pd.DataFrame:
    """Role inference without pitch coordinates (camera-relative footage).

    Uses each player's average position along the camera's horizontal axis
    relative to their own team: the team defends toward one side of the frame,
    so the ordering still separates deep players from advanced ones. Confidence
    is capped low — a panning camera makes this a hint, not a measurement, and
    the coach is expected to correct it in the dashboard.
    """
    detector_gk = detector_gk or set()
    rows = []
    for _, tg in tracks[tracks.team.isin(["A", "B"])].groupby("team"):
        per = tg.groupby("track_id")
        mean_x = per["x_m"].mean()
        if mean_x.empty:
            continue
        # orient so that "low" = the side this team's players sit behind on average
        team_axis = mean_x.rank(pct=True)
        spread = (per["x_m"].std().fillna(0) / (mean_x.abs().mean() or 1)).clip(0, 1)

        for tid in per.groups:
            if tid in detector_gk:
                rows.append((int(tid), "GK", 0.5))
                continue
            a = float(team_axis[tid])
            role = "DEF" if a <= 1 / 3 else ("MID" if a <= 2 / 3 else "ATT")
            conf = float(max(0.15, 0.5 - 0.3 * float(spread.get(tid, 0))))
            rows.append((int(tid), role, round(conf, 2)))
    return pd.DataFrame(rows, columns=["track_id", "role", "confidence"])


def infer_roles(tracks: pd.DataFrame, attack_dir: dict[str, int],
                detector_gk: set[int] | None = None) -> pd.DataFrame:
    """tracks: frame, track_id, team, x_m, y_m (pitch coords).

    Returns: track_id, role, confidence (0-1).
    """
    detector_gk = detector_gk or set()
    rows = []
    for team, tg in tracks[tracks.team.isin(["A", "B"])].groupby("team"):
        d = attack_dir.get(team, +1)
        per = tg.groupby("track_id")

        # normalized advance: 0 = own goal, 1 = opponent goal
        adv = per["x_m"].mean()
        adv = adv / PITCH_LENGTH_M if d > 0 else 1 - adv / PITCH_LENGTH_M
        spread = per["x_m"].std().fillna(0) / PITCH_LENGTH_M

        box_dwell = tg.assign(inbox=_in_own_box(tg.x_m, tg.y_m, d)).groupby("track_id")["inbox"].mean()

        gk_ids = {tid for tid in per.groups if tid in detector_gk or box_dwell.get(tid, 0) > 0.6}
        outfield = adv.drop(index=[t for t in gk_ids if t in adv.index])

        # terciles of advance among this team's outfield players
        if len(outfield) >= 3:
            q1, q2 = outfield.quantile([1 / 3, 2 / 3])
        else:
            q1, q2 = 1 / 3, 2 / 3

        for tid in per.groups:
            if tid in gk_ids:
                rows.append((int(tid), "GK", float(min(1.0, box_dwell.get(tid, 0) + 0.3))))
                continue
            a = float(adv[tid])
            role = "DEF" if a <= q1 else ("MID" if a <= q2 else "ATT")
            # roamers (huge positional spread) get low confidence
            conf = float(max(0.2, 1.0 - 2.5 * float(spread[tid])))
            rows.append((int(tid), role, round(conf, 2)))
    return pd.DataFrame(rows, columns=["track_id", "role", "confidence"])
