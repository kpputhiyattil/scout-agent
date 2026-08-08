"""Rating engine: role-weighted percentile scoring -> 0-100 + sub-scores.

Why percentiles: absolute benchmarks don't exist for kids. Each KPI is ranked
within the squad's role peers (fallback: all outfield players when the peer
group is too small), then combined with coach-editable weights from weights.yaml.
Negative weights (e.g. possession_lost_p90) invert the percentile.
"""
from __future__ import annotations

import pandas as pd

MIN_PEER_GROUP = 6


def _percentile_rank(series: pd.Series) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series(50.0, index=series.index)
    return series.rank(pct=True) * 100


def rate_players(metrics: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """metrics: one row per track_id incl. 'role' column.
    weights: role -> kpi -> {weight, group} (from weights.yaml).

    KPIs that are entirely unavailable (all-NaN — e.g. distance and speed on
    camera-relative footage) are dropped and the remaining weights renormalize,
    so a rating always reflects only what was actually measured. `measured_kpis`
    records what fed each score.

    Returns: track_id, role, overall (0-100), sub_scores {group: 0-100},
             evidence {kpi: raw value}, low_sample, measured_kpis.
    """
    out = []
    outfield = metrics[metrics.role.isin(["DEF", "MID", "ATT"])]

    for role, spec in weights.items():
        players = metrics[metrics.role == role]
        if players.empty:
            continue
        peers = players if len(players) >= MIN_PEER_GROUP else (
            outfield if role != "GK" and len(outfield) >= MIN_PEER_GROUP else players)

        kpis = [k for k in spec if k in metrics.columns and not metrics[k].isna().all()]
        pct = {k: _percentile_rank(peers[k]) for k in kpis}

        for _, p in players.iterrows():
            tid = p.track_id
            total_w, score = 0.0, 0.0
            groups: dict[str, list[tuple[float, float]]] = {}
            used = []
            for k in kpis:
                if pd.isna(p[k]):                    # missing for this player only
                    continue
                used.append(k)
                w = spec[k]["weight"]
                raw = pct[k].get(p.name, 50.0)
                v = float(raw) if pd.notna(raw) else 50.0
                v = 100 - v if w < 0 else v          # negative weight => lower is better
                aw = abs(w)
                score += aw * v
                total_w += aw
                groups.setdefault(spec[k]["group"], []).append((aw, v))
            overall = round(score / total_w, 1) if total_w else 50.0
            subs = {g: round(sum(w * v for w, v in items) / sum(w for w, _ in items), 1)
                    for g, items in groups.items()}
            out.append({
                "track_id": int(tid), "role": role, "overall": overall,
                "sub_scores": subs,
                "evidence": {k: round(float(p[k]), 2) for k in used},
                "low_sample": bool(p.get("low_sample", False)),
                "measured_kpis": len(used),
            })
    return pd.DataFrame(out).sort_values("overall", ascending=False).reset_index(drop=True)
