import pandas as pd

from scout.analytics.rating import rate_players

WEIGHTS = {
    "ATT": {
        "goals_p90": {"weight": 0.6, "group": "attacking"},
        "possession_lost_p90": {"weight": -0.4, "group": "technical"},
    },
    "GK": {"saves_p90": {"weight": 1.0, "group": "defending"}},
}


def _metrics():
    return pd.DataFrame([
        {"track_id": 1, "role": "ATT", "goals_p90": 2.0, "possession_lost_p90": 1.0, "low_sample": False},
        {"track_id": 2, "role": "ATT", "goals_p90": 0.5, "possession_lost_p90": 5.0, "low_sample": False},
        {"track_id": 3, "role": "ATT", "goals_p90": 1.0, "possession_lost_p90": 3.0, "low_sample": True},
        {"track_id": 4, "role": "GK", "saves_p90": 4.0, "goals_p90": 0, "possession_lost_p90": 0,
         "low_sample": False},
    ])


def test_better_player_rates_higher():
    r = rate_players(_metrics(), WEIGHTS).set_index("track_id")
    assert r.loc[1, "overall"] > r.loc[2, "overall"]


def test_negative_weight_inverts():
    # player 2 loses the ball most -> that KPI should drag him down, not up
    r = rate_players(_metrics(), WEIGHTS).set_index("track_id")
    assert r.loc[2, "overall"] == r["overall"].min()


def test_scores_bounded_and_subscores_present():
    r = rate_players(_metrics(), WEIGHTS)
    assert r.overall.between(0, 100).all()
    assert all("attacking" in s for s in r[r.role == "ATT"].sub_scores)


def test_single_gk_gets_neutral_score():
    r = rate_players(_metrics(), WEIGHTS).set_index("track_id")
    assert r.loc[4, "overall"] == 50.0  # no peers -> percentile 50


def test_low_sample_flag_propagates():
    r = rate_players(_metrics(), WEIGHTS).set_index("track_id")
    assert bool(r.loc[3, "low_sample"]) is True
