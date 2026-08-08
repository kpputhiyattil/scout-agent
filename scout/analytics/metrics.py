"""Per-player KPI computation from events + tracks. All *_p90 normalized.

Small samples (<15 min played) are flagged low_sample=True — per-90 numbers
from 10 minutes of play are noisy and the rating engine discloses it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scout.config import (
    MAX_PLAUSIBLE_SPEED_KMH,
    PITCH_LENGTH_M,
    PITCH_ONLY_KPIS,
    SPRINT_SPEED_KMH,
)


def physical_metrics(tracks: pd.DataFrame, fps: float) -> pd.DataFrame:
    """distance_km, top_speed_kmh, sprints per track. Speeds above the plausible
    bound for kids are treated as tracking glitches and dropped."""
    rows = []
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        dt = g.frame.diff() / fps
        dist = np.hypot(g.x_m.diff(), g.y_m.diff())
        speed_kmh = (dist / dt * 3.6).replace([np.inf, -np.inf], np.nan)
        ok = speed_kmh < MAX_PLAUSIBLE_SPEED_KMH
        speed_kmh = speed_kmh.where(ok)
        dist = dist.where(ok)

        sprinting = speed_kmh > SPRINT_SPEED_KMH
        sprints = int((sprinting & ~sprinting.shift(fill_value=False)).sum())
        minutes = float((g.frame.max() - g.frame.min()) / fps / 60)
        rows.append((int(tid), float(dist.sum(skipna=True) / 1000),
                     float(speed_kmh.max(skipna=True) or 0), sprints, minutes))
    return pd.DataFrame(rows, columns=["track_id", "distance_km", "top_speed_kmh", "sprints", "minutes"])


def _p90(count: float, minutes: float) -> float:
    return float(count * 90 / minutes) if minutes > 0 else 0.0


def compute_metrics(events: pd.DataFrame, tracks: pd.DataFrame, fps: float,
                    attack_dir: dict[str, int], roles: pd.DataFrame,
                    mode: str = "pitch") -> pd.DataFrame:
    """events: frame, type, actor(track_id), plus type-specific cols.
    Returns one row per track_id with every KPI weights.yaml can reference.

    mode='relative' (no homography): KPIs that depend on true pitch coordinates
    are set to NaN rather than reported as bogus numbers — the rating engine
    drops them and renormalizes the remaining weights.
    """
    phys = physical_metrics(tracks, fps).set_index("track_id")
    team_of = tracks.groupby("track_id")["team"].agg(lambda x: x.mode().iat[0])
    pos_mean = tracks.groupby("track_id")[["x_m", "y_m"]].mean()
    role_of = roles.set_index("track_id")["role"]

    def ev(t):
        if "type" not in events.columns:
            return events.iloc[0:0]
        return events[events.type == t]

    def col(df, name, default=np.nan):
        """Event column that only exists once that event type occurred."""
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    passes, losses = ev("pass"), ev("loss")
    inter, duels = ev("interception"), ev("duel")
    shots, saves = ev("shot"), ev("save")

    rows = []
    for tid in phys.index:
        m = phys.loc[tid, "minutes"]
        team = team_of.get(tid, "?")
        d = attack_dir.get(team, +1)

        p_made = passes[col(passes, "actor", -1) == tid]
        p_lost = losses[col(losses, "actor", -1) == tid]
        n_pass_att = len(p_made) + len(p_lost)

        # forward pass: receiver is further along attack axis than passer at pass frame
        fwd = 0
        pos_at = tracks.set_index(["frame", "track_id"])
        for r in p_made.itertuples():
            try:
                dx = pos_at.loc[(r.frame, int(r.target)), "x_m"] - pos_at.loc[(r.frame, tid), "x_m"]
                fwd += int(dx * d > 2)
            except KeyError:
                pass

        my_shots = shots[col(shots, "actor", -1) == tid]
        opp_shots_on_t = shots[(col(shots, "on_target") == 1)
                               & (col(shots, "actor", -1).map(team_of) != team)]
        my_saves = saves[col(saves, "actor", -1) == tid]
        duels_w = duels[duels.winner == tid] if "winner" in duels else duels.iloc[0:0]
        duels_l = duels[duels.loser == tid] if "loser" in duels else duels.iloc[0:0]

        in_final_third = tracks[(tracks.track_id == tid)]
        ft = ((in_final_third.x_m > 2 / 3 * PITCH_LENGTH_M) if d > 0
              else (in_final_third.x_m < PITCH_LENGTH_M / 3)).mean()

        n_duels = len(duels_w) + len(duels_l)
        rows.append({
            "track_id": int(tid), "team": team, "role": role_of.get(tid, "?"),
            "minutes": round(m, 1), "low_sample": m < 15,
            "passes_p90": _p90(len(p_made), m),
            "pass_completion_pct": 100 * len(p_made) / n_pass_att if n_pass_att else 0.0,
            "forward_pass_pct": 100 * fwd / len(p_made) if len(p_made) else 0.0,
            "possession_won_p90": _p90(len(inter[inter.actor == tid]) + len(duels_w), m),
            "possession_lost_p90": _p90(len(p_lost) + len(duels_l), m),
            "duel_win_pct": 100 * len(duels_w) / n_duels if n_duels else 0.0,
            "interceptions_p90": _p90(len(inter[inter.actor == tid]), m),
            "clearances_p90": _p90(len(ev("clearance")[ev("clearance").actor == tid]) if len(ev("clearance")) else 0, m),
            "shots_p90": _p90(len(my_shots), m),
            "shots_on_target_p90": _p90(float(my_shots.on_target.sum(skipna=True)) if len(my_shots) else 0, m),
            "goals_p90": _p90(len(ev("goal")[ev("goal").actor == tid]) if len(ev("goal")) else 0, m),
            "chances_created_p90": _p90(sum(1 for r in p_made.itertuples()
                                            if len(shots[(shots.actor == r.target)
                                                         & (shots.frame - r.frame < 5 * fps)
                                                         & (shots.frame > r.frame)])), m),
            "dribble_success_pct": 100 * len(duels_w) / n_duels if n_duels else 0.0,
            "progressive_carries_p90": _p90(_progressive_carries(tracks, tid, d), m),
            "final_third_touches_p90": _p90(float(ft) * len(p_made) if len(p_made) else 0, m),
            "saves_p90": _p90(len(my_saves), m),
            "save_pct": 100 * len(my_saves) / len(opp_shots_on_t) if len(opp_shots_on_t) else 0.0,
            "goals_conceded_p90": _p90(max(0, len(opp_shots_on_t) - len(my_saves)), m)
                                   if role_of.get(tid) == "GK" else 0.0,
            "sweeper_actions_p90": _p90(len(inter[inter.actor == tid]), m)
                                    if role_of.get(tid) == "GK" else 0.0,
            "distance_km_p90": _p90(phys.loc[tid, "distance_km"], m),
            "top_speed_kmh": round(float(phys.loc[tid, "top_speed_kmh"]), 1),
            "sprints_p90": _p90(phys.loc[tid, "sprints"], m),
        })
    df = pd.DataFrame(rows)
    if mode == "relative":
        # camera motion makes these unmeasurable; NaN is honest, a number is not
        for kpi in PITCH_ONLY_KPIS:
            if kpi in df.columns:
                df[kpi] = np.nan
        if "shots_on_target_p90" in df.columns:
            df["shots_on_target_p90"] = np.nan   # no goal line to test against
        if "save_pct" in df.columns:
            df["save_pct"] = np.nan
    return df


def _progressive_carries(tracks: pd.DataFrame, tid: int, d: int,
                         min_gain_m: float = 10.0, window_s: float = 4.0, fps: float = 25.0) -> int:
    """Count windows where the player advanced >= min_gain_m toward goal."""
    g = tracks[tracks.track_id == tid].sort_values("frame")
    if len(g) < 2:
        return 0
    x = g.x_m.to_numpy() * d
    f = g.frame.to_numpy()
    n, count, i = len(x), 0, 0
    w = int(window_s * fps)
    while i < n - 1:
        j = np.searchsorted(f, f[i] + w, side="right") - 1
        if j > i and x[j] - x[i] >= min_gain_m:
            count += 1
            i = j
        else:
            i += 1
    return count
