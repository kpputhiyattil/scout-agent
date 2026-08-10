"""Goal and own-goal detection.

A goal is the ball crossing a goal line between the posts. Credit goes to the last
player in possession; if that player's team defends the goal that was scored in, it
is an own goal — recorded, but never counted as a goal for the child.
"""
import numpy as np
import pandas as pd
import pytest

from scout.analytics.events import detect_goals
from scout.config import PITCH_LENGTH_M, PITCH_WIDTH_M

FPS = 25.0
MID_Y = PITCH_WIDTH_M / 2
# A attacks toward x=L (defends x=0); B attacks toward x=0 (defends x=L)
ATTACK_DIR = {"A": +1, "B": -1}
TEAM_OF = {1: "A", 2: "B"}


def ball_crossing(from_x, to_x, y=MID_Y, start_frame=100):
    return pd.DataFrame({"frame": [start_frame, start_frame + 1],
                         "x_m": [from_x, to_x], "y_m": [y, y]})


def spell(owner, end_frame=99):
    return pd.DataFrame({"owner": [owner], "team": [TEAM_OF[owner]],
                         "start": [end_frame - 20], "end": [end_frame]})


class TestGoalDetection:
    def test_attacker_scoring_in_opponent_goal_is_a_goal(self):
        # A attacks toward x=L, ball crosses that line, A player last touched it
        goals = detect_goals(ball_crossing(PITCH_LENGTH_M - 2, PITCH_LENGTH_M + 0.2),
                             spell(1), FPS, ATTACK_DIR, TEAM_OF)
        assert len(goals) == 1
        assert goals.type.iat[0] == "goal"
        assert goals.actor.iat[0] == 1
        assert goals.scoring_team.iat[0] == "A"

    def test_ball_wide_of_the_post_is_not_a_goal(self):
        wide = MID_Y + 10          # outside the goal mouth
        goals = detect_goals(ball_crossing(PITCH_LENGTH_M - 2, PITCH_LENGTH_M + 0.2, y=wide),
                             spell(1), FPS, ATTACK_DIR, TEAM_OF)
        assert goals.empty

    def test_ball_near_the_line_without_crossing_is_not_a_goal(self):
        goals = detect_goals(ball_crossing(PITCH_LENGTH_M - 5, PITCH_LENGTH_M - 2),
                             spell(1), FPS, ATTACK_DIR, TEAM_OF)
        assert goals.empty

    def test_goal_at_the_other_end_credits_the_other_team(self):
        # crossing x=0 means the team attacking toward 0 (B) scored
        goals = detect_goals(ball_crossing(2, -0.2), spell(2), FPS, ATTACK_DIR, TEAM_OF)
        assert goals.type.iat[0] == "goal"
        assert goals.scoring_team.iat[0] == "B"

    def test_one_goal_is_not_double_counted(self):
        """The ball sits in the net for several frames after crossing."""
        b = pd.DataFrame({"frame": range(100, 140),
                          "x_m": [PITCH_LENGTH_M - 2] + [PITCH_LENGTH_M + 0.3] * 39,
                          "y_m": [MID_Y] * 40})
        assert len(detect_goals(b, spell(1), FPS, ATTACK_DIR, TEAM_OF)) == 1


class TestOwnGoal:
    def test_player_scoring_in_the_goal_his_team_defends_is_an_own_goal(self):
        # player 1 is team A, which DEFENDS x=0; ball crosses x=0 off player 1
        goals = detect_goals(ball_crossing(2, -0.2), spell(1), FPS, ATTACK_DIR, TEAM_OF)
        assert len(goals) == 1
        assert goals.type.iat[0] == "own_goal"
        assert goals.actor.iat[0] == 1

    def test_own_goal_credits_the_opposing_team_with_the_score(self):
        goals = detect_goals(ball_crossing(2, -0.2), spell(1), FPS, ATTACK_DIR, TEAM_OF)
        assert goals.scoring_team.iat[0] == "B", "the goal counts for the other team"

    def test_own_goal_is_not_counted_as_a_goal_for_the_child(self):
        from scout.analytics.metrics import compute_metrics
        events = pd.DataFrame([(100, "own_goal", 1, "B")],
                              columns=["frame", "type", "actor", "scoring_team"])
        tracks = pd.DataFrame({"frame": list(range(0, 500, 10)) * 1,
                               "track_id": 1, "team": "A", "x_m": 50.0, "y_m": 30.0})
        roles = pd.DataFrame({"track_id": [1], "role": ["DEF"], "confidence": [0.8]})
        m = compute_metrics(events, tracks, FPS, ATTACK_DIR, roles)
        assert m.goals_p90.iat[0] == 0.0, "an own goal must never read as a goal scored"
        assert m.own_goals.iat[0] == 1.0, "but it is still recorded for context"


class TestCameraRelativeMode:
    def test_no_attack_direction_means_no_goals_claimed(self):
        """Without a pitch reference there is no goal line — report nothing."""
        goals = detect_goals(ball_crossing(PITCH_LENGTH_M - 2, PITCH_LENGTH_M + 0.2),
                             spell(1), FPS, attack_dir={}, team_of=TEAM_OF)
        assert goals.empty

    def test_own_goals_are_unweighted_in_ratings(self):
        """Development reports must not punish a child for an own goal."""
        import yaml
        weights = yaml.safe_load(open("weights.yaml"))
        for role, kpis in weights.items():
            assert "own_goals" not in kpis, f"{role} must not weight own_goals"
