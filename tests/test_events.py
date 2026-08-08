"""Synthetic-scenario tests for every event rule."""
import numpy as np
import pandas as pd

from scout.analytics import events as E
from tests.conftest import make_tracks, straight_run


def _ball_at(points):
    return pd.DataFrame(points, columns=["frame", "x_m", "y_m"])


def test_possession_nearest_within_radius(fps):
    tracks = make_tracks({
        1: ("A", [(f, 50.0, 32.0) for f in range(50)]),   # at the ball
        2: ("B", [(f, 80.0, 32.0) for f in range(50)]),   # far away
    })
    ball = _ball_at([(f, 50.5, 32.0) for f in range(50)])
    poss = E.compute_possession(tracks, ball, radius_m=2.5, hysteresis=3)
    assert (poss.owner.iloc[10:] == 1).all()


def test_possession_hysteresis_blocks_flicker(fps):
    # ball ping-pongs near player 2 for only 2 frames — shouldn't switch (hysteresis=3)
    tracks = make_tracks({
        1: ("A", [(f, 50.0, 32.0) for f in range(30)]),
        2: ("B", [(f, 53.5, 32.0) for f in range(30)]),
    })
    pts = [(f, 50.5, 32.0) for f in range(20)] + [(20, 53.4, 32.0), (21, 53.4, 32.0)] \
        + [(f, 50.5, 32.0) for f in range(22, 30)]
    poss = E.compute_possession(tracks, _ball_at(pts), radius_m=2.5, hysteresis=3)
    assert (poss[poss.frame >= 25].owner == 1).all()


def test_ball_gap_beyond_max_interp_is_unknown(fps):
    tracks = make_tracks({1: ("A", [(f, 50.0, 32.0) for f in range(100)])})
    ball = _ball_at([(f, 50.5, 32.0) for f in range(10)] + [(f, 50.5, 32.0) for f in range(80, 100)])
    ball_full = E.interpolate_ball(ball, 100, max_gap=25)
    # 70-frame gap > 25 => NaN in the middle
    assert ball_full.loc[45, ["x_m", "y_m"]].isna().all()
    poss = E.compute_possession(tracks, ball_full, radius_m=2.5, hysteresis=1)
    assert (poss[(poss.frame > 12) & (poss.frame < 78)].owner == E.UNKNOWN).all()


def test_pass_same_team_and_interception_other_team(fps):
    spells = pd.DataFrame([
        {"owner": 1, "team": "A", "start": 0, "end": 10},
        {"owner": 2, "team": "A", "start": 15, "end": 30},   # pass A->A
        {"owner": 3, "team": "B", "start": 33, "end": 50},   # loss + interception
    ])
    ev = E.detect_transitions(spells, fps)
    assert set(ev.type) == {"pass", "loss", "interception"}
    p = ev[ev.type == "pass"].iloc[0]
    assert p.actor == 1 and p.target == 2 and p.success == 1
    assert ev[ev.type == "interception"].iloc[0].actor == 3


def test_no_pass_across_long_gap(fps):
    spells = pd.DataFrame([
        {"owner": 1, "team": "A", "start": 0, "end": 10},
        {"owner": 2, "team": "A", "start": 10 + int(3 * fps), "end": 200},  # 3s gap > 2s max
    ])
    assert E.detect_transitions(spells, fps).empty


def test_shot_detected_toward_goal(fps):
    # player 1 (team A attacking +x) at x=80; ball accelerates toward goal at x=100
    tracks = make_tracks({1: ("A", [(f, 80.0, 32.0) for f in range(60)])})
    ball_pts = [(f, 80.0, 32.0) for f in range(30)] + \
               straight_run(range(30, 60), 80.0, 32.0, vx=15.0)  # 15 m/s
    ball = _ball_at(ball_pts)
    poss = E.compute_possession(tracks, ball, radius_m=2.5, hysteresis=1)
    shots = E.detect_shots(ball, poss, fps, {"A": +1, "B": -1}, shot_speed_ms=8.0)
    assert len(shots) == 1
    assert shots.iloc[0].actor == 1
    assert shots.iloc[0].on_target == 1  # straight at goal center


def test_slow_ball_is_not_a_shot(fps):
    tracks = make_tracks({1: ("A", [(f, 80.0, 32.0) for f in range(60)])})
    ball = _ball_at(straight_run(range(60), 80.0, 32.0, vx=3.0))  # 3 m/s dribble
    poss = E.compute_possession(tracks, ball, radius_m=2.5, hysteresis=1)
    assert E.detect_shots(ball, poss, fps, {"A": +1}).empty


def test_save_gk_gains_ball_after_on_target_shot(fps):
    shots = pd.DataFrame([{"frame": 100, "type": "shot", "actor": 9, "on_target": 1}])
    spells = pd.DataFrame([
        {"owner": 9, "team": "A", "start": 60, "end": 99},
        {"owner": 1, "team": "B", "start": 120, "end": 200},  # GK 0.8s later
    ])
    saves = E.detect_saves(shots, spells, fps, gk_tracks={1})
    assert len(saves) == 1 and saves.iloc[0].actor == 1


def test_duel_requires_proximity(fps):
    spells = pd.DataFrame([
        {"owner": 1, "team": "A", "start": 0, "end": 49},
        {"owner": 2, "team": "B", "start": 50, "end": 100},
    ])
    close = make_tracks({1: ("A", [(50, 40.0, 30.0)]), 2: ("B", [(50, 41.0, 30.0)])})
    far = make_tracks({1: ("A", [(50, 40.0, 30.0)]), 2: ("B", [(50, 60.0, 30.0)])})
    assert len(E.detect_duels(spells, close, fps, radius_m=2.0)) == 1
    assert E.detect_duels(spells, far, fps, radius_m=2.0).empty


def test_attack_direction_from_gk_position():
    tracks = make_tracks({
        1: ("A", [(f, 8.0, 32.0) for f in range(10)]),    # GK A near x=0
        2: ("B", [(f, 92.0, 32.0) for f in range(10)]),   # GK B near x=100
    })
    d = E.infer_attack_direction(tracks, {"A": 1, "B": 2})
    assert d == {"A": +1, "B": -1}
