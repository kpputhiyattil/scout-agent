"""Rule-based event detection over pitch-coordinate tracks.

Input DataFrames (pitch meters, attack direction NOT yet normalized):
  tracks: frame, track_id, team, x_m, y_m
  ball:   frame, x_m, y_m

All rules are pure functions -> unit-testable with synthetic scenarios.
Honesty rule: ball gaps > max_interp frames => possession 'unknown', never guessed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scout.config import PITCH_LENGTH_M, PITCH_WIDTH_M, get_settings

UNKNOWN = -1


def interpolate_ball(ball: pd.DataFrame, n_frames: int, max_gap: int) -> pd.DataFrame:
    """Reindex ball to every frame; linear-fill only gaps whose FULL length
    <= max_gap. Longer gaps stay NaN entirely — never partially guessed."""
    full = ball.set_index("frame").reindex(range(n_frames))
    isna = full["x_m"].isna()
    gap_id = (isna != isna.shift()).cumsum()
    gap_len = isna.groupby(gap_id).transform("sum")
    fillable = isna & (gap_len <= max_gap)
    interp = full.interpolate(limit_area="inside")
    out = full.copy()
    out[fillable] = interp[fillable]
    return out.reset_index()


def compute_possession(tracks: pd.DataFrame, ball: pd.DataFrame,
                       radius_m: float | None = None,
                       hysteresis: int | None = None) -> pd.DataFrame:
    """Per-frame possession: nearest player within radius, with hysteresis
    (a new owner must be nearest for `hysteresis` consecutive frames to take over).

    Returns: frame, owner (track_id or UNKNOWN), team.
    """
    s = get_settings()
    radius_m = radius_m or s.possession_radius_m
    hysteresis = hysteresis if hysteresis is not None else s.possession_hysteresis_frames

    team_of = tracks.groupby("track_id")["team"].agg(lambda x: x.mode().iat[0]).to_dict()
    ball_by_frame = ball.set_index("frame")[["x_m", "y_m"]]

    rows = []
    current, candidate, streak = UNKNOWN, UNKNOWN, 0
    for frame, grp in tracks.groupby("frame"):
        if frame not in ball_by_frame.index or ball_by_frame.loc[frame].isna().any():
            current, candidate, streak = UNKNOWN, UNKNOWN, 0
            rows.append((frame, UNKNOWN, "?"))
            continue
        bx, by = ball_by_frame.loc[frame, "x_m"], ball_by_frame.loc[frame, "y_m"]
        d = np.hypot(grp["x_m"] - bx, grp["y_m"] - by)
        nearest_i = d.idxmin()
        nearest, dist = int(grp.loc[nearest_i, "track_id"]), float(d.loc[nearest_i])

        if dist > radius_m:
            candidate, streak = UNKNOWN, 0
            # keep current owner briefly (ball in flight) — ownership persists
        elif nearest == current:
            candidate, streak = UNKNOWN, 0
        elif nearest == candidate:
            streak += 1
            if streak >= hysteresis:
                current, candidate, streak = nearest, UNKNOWN, 0
        else:
            candidate, streak = nearest, 1
            if hysteresis <= 1:
                current, candidate, streak = nearest, UNKNOWN, 0
        rows.append((frame, current, team_of.get(current, "?")))
    return pd.DataFrame(rows, columns=["frame", "owner", "team"])


def possession_spells(possession: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-frame ownership into spells: owner, team, start, end."""
    p = possession[possession.owner != UNKNOWN].copy()
    if p.empty:
        return pd.DataFrame(columns=["owner", "team", "start", "end"])
    p["spell"] = (p.owner != p.owner.shift()).cumsum()
    g = p.groupby("spell").agg(owner=("owner", "first"), team=("team", "first"),
                               start=("frame", "min"), end=("frame", "max"))
    return g.reset_index(drop=True)


def detect_transitions(spells: pd.DataFrame, fps: float, max_gap_s: float = 2.0) -> pd.DataFrame:
    """Possession changes -> pass (same team) / loss+interception (other team).

    Returns events: frame, type, actor, target, success.
    """
    rows = []
    for prev, nxt in zip(spells.itertuples(), spells.iloc[1:].itertuples()):
        gap_s = (nxt.start - prev.end) / fps
        if gap_s > max_gap_s or prev.owner == nxt.owner:
            continue
        if prev.team == nxt.team and prev.team != "?":
            rows.append((prev.end, "pass", prev.owner, nxt.owner, 1))
        elif prev.team != "?" and nxt.team != "?":
            rows.append((prev.end, "loss", prev.owner, nxt.owner, 0))
            rows.append((nxt.start, "interception", nxt.owner, prev.owner, 1))
    return pd.DataFrame(rows, columns=["frame", "type", "actor", "target", "success"])


def detect_shots(ball: pd.DataFrame, possession: pd.DataFrame, fps: float,
                 attack_dir: dict[str, int] | None,
                 shot_speed_ms: float | None = None) -> pd.DataFrame:
    """Shot = ball leaves a player's control at high speed, moving toward the
    goal his team attacks, from inside the attacking 40% of the pitch.

    attack_dir: {'A': +1|-1, 'B': ...} — +1 means attacking toward x = L.
    Pass None for camera-relative footage: without a pitch frame of reference the
    zone and goal-line checks are meaningless, so every high-speed release counts
    as a strike attempt and on_target is left unknown (NaN).
    """
    s = get_settings()
    shot_speed_ms = shot_speed_ms or s.shot_speed_ms
    b = ball.dropna(subset=["x_m", "y_m"]).copy()
    if len(b) < 3:
        return pd.DataFrame(columns=["frame", "type", "actor", "on_target"])
    b["vx"] = b.x_m.diff() * fps / b.frame.diff()
    b["vy"] = b.y_m.diff() * fps / b.frame.diff()
    b["speed"] = np.hypot(b.vx, b.vy)
    b["prev_speed"] = b.speed.shift()
    poss = possession.set_index("frame")

    rows, cooldown = [], -999
    goal_y = (PITCH_WIDTH_M / 2 - 3.66, PITCH_WIDTH_M / 2 + 3.66)
    for r in b.itertuples():
        if r.frame - cooldown < fps:  # 1s between shots
            continue
        # kick moment = speed rising edge: ball was slow, now fast.
        # A ball already in flight stays above threshold and never re-triggers.
        if not (r.speed >= shot_speed_ms > (r.prev_speed if np.isfinite(r.prev_speed) else 0.0)):
            continue
        if r.frame not in poss.index:
            continue
        owner, team = poss.loc[r.frame, "owner"], poss.loc[r.frame, "team"]
        if owner == UNKNOWN:
            continue
        if attack_dir is None:  # camera-relative: no goal to aim at, on_target unknown
            rows.append((int(r.frame), "shot", int(owner), np.nan))
            cooldown = r.frame
            continue
        if team not in attack_dir:
            continue
        d = attack_dir[team]
        in_attacking_zone = (r.x_m > 0.6 * PITCH_LENGTH_M) if d > 0 else (r.x_m < 0.4 * PITCH_LENGTH_M)
        toward_goal = (r.vx * d) > abs(r.vy)
        if in_attacking_zone and toward_goal:
            goal_x = PITCH_LENGTH_M if d > 0 else 0.0
            t_to_line = (goal_x - r.x_m) / r.vx if r.vx else np.inf
            y_at_line = r.y_m + r.vy * t_to_line if np.isfinite(t_to_line) and t_to_line > 0 else np.inf
            on_target = int(goal_y[0] - 1 <= y_at_line <= goal_y[1] + 1)
            rows.append((int(r.frame), "shot", int(owner), on_target))
            cooldown = r.frame
    return pd.DataFrame(rows, columns=["frame", "type", "actor", "on_target"])


def detect_saves(shots: pd.DataFrame, spells: pd.DataFrame, fps: float,
                 gk_tracks: set[int], window_s: float | None = None) -> pd.DataFrame:
    """Save = GK gains possession within window after an on-target shot by the opponent."""
    window = (window_s or get_settings().save_window_s) * fps
    rows = []
    for shot in shots[shots.on_target == 1].itertuples():
        after = spells[(spells.start > shot.frame) & (spells.start <= shot.frame + window)]
        gk = after[after.owner.isin(gk_tracks)]
        if not gk.empty:
            g = gk.iloc[0]
            rows.append((int(g.start), "save", int(g.owner), int(shot.actor)))
    return pd.DataFrame(rows, columns=["frame", "type", "actor", "shooter"])


def detect_duels(spells: pd.DataFrame, tracks: pd.DataFrame, fps: float,
                 radius_m: float | None = None) -> pd.DataFrame:
    """Duel = possession flips between opposing players who were within radius
    of each other at the flip. Winner = new owner."""
    radius_m = radius_m or get_settings().duel_radius_m
    pos = tracks.set_index(["frame", "track_id"])[["x_m", "y_m"]].sort_index()

    def at(frame, track_id):
        """Position of one player in one frame, or None. Tolerates duplicate rows."""
        try:
            row = pos.loc[(frame, track_id)]
        except KeyError:
            return None
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row

    rows = []
    for prev, nxt in zip(spells.itertuples(), spells.iloc[1:].itertuples()):
        if prev.team == nxt.team or "?" in (prev.team, nxt.team):
            continue
        f = nxt.start
        a, b = at(f, prev.owner), at(f, nxt.owner)
        if a is None or b is None:
            continue
        if np.hypot(a.x_m - b.x_m, a.y_m - b.y_m) <= radius_m:
            rows.append((int(f), "duel", int(nxt.owner), int(prev.owner)))
    return pd.DataFrame(rows, columns=["frame", "type", "winner", "loser"])


def infer_attack_direction(tracks: pd.DataFrame, gk_tracks_by_team: dict[str, int]) -> dict[str, int]:
    """A team attacks away from its own GK's average x. Recompute per half upstream
    if the camera crosses the halfway line."""
    out = {}
    mean_x = tracks.groupby("track_id")["x_m"].mean()
    for team, gk in gk_tracks_by_team.items():
        if gk in mean_x.index:
            out[team] = +1 if mean_x[gk] < PITCH_LENGTH_M / 2 else -1
    # fallback: opposite of the other team
    for t in ("A", "B"):
        if t not in out and ("A" if t == "B" else "B") in out:
            out[t] = -out["A" if t == "B" else "B"]
    return out
