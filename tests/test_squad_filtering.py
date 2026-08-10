"""Only rate players a coach can actually identify, in plausible squad numbers.

A 5-minute U8 clip produced 1239 tracked identities and 137 "defenders" for one
team. A real squad is ~7-16 children. Three filters bring output back to reality:
a readable jersey number, a minimum observed time, and a squad-size cap.
"""
import pytest

from scout.perception.jersey import Read, roster_numbers, vote_jerseys

from tests.test_pipeline_integration import pipeline_env  # noqa: F401


class TestJerseyPlausibility:
    def test_numbers_beyond_squad_range_are_rejected(self):
        # "76" on a U8 shirt is almost always a misread of 7 or 6
        reads = [Read(1, 76, 0.9, 400.0) for _ in range(5)]
        assert vote_jerseys(reads, max_number=30) == {}

    def test_numbers_inside_range_survive(self):
        reads = [Read(1, 16, 0.9, 400.0) for _ in range(5)]
        assert vote_jerseys(reads, max_number=30) == {1: 16}

    def test_roster_restricts_to_real_squad_numbers(self):
        reads = [Read(1, 9, 0.9, 400.0) for _ in range(5)]
        assert vote_jerseys(reads, allowed={7, 10, 11}) == {}
        assert vote_jerseys(reads, allowed={7, 9, 11}) == {1: 9}

    def test_roster_numbers_read_from_csv(self, tmp_path):
        p = tmp_path / "roster.csv"
        p.write_text("jersey_number,name\n7,Alex\n10,Sam\n")
        assert roster_numbers(p) == {7, 10}

    def test_missing_roster_means_no_restriction(self):
        assert roster_numbers(None) is None


class TestSquadFiltering:
    """End-to-end: the rate stage must drop unidentifiable and surplus identities."""

    @pytest.fixture
    def env(self, pipeline_env):  # noqa: F811
        return pipeline_env

    def test_unidentified_players_are_not_rated(self, env, monkeypatch):
        import json

        import scout.perception.jersey as jersey
        from scout.config import get_settings
        from scout.db import Player, get_session
        from scout.pipeline import run
        from tests.test_pipeline_integration import _make_video

        # only 3 of the 12 tracked identities get a jersey number
        monkeypatch.setattr(jersey, "read_jerseys",
                            lambda v, t, **kw: {1: 7, 2: 9, 3: 11})
        mid = run(str(_make_video(env / "match.mp4")))

        q = json.loads((get_settings().match_dir(mid) / "quality.json").read_text())
        assert q["n_unidentified_dropped"] == 9
        assert q["n_rated"] == 3
        with get_session() as db:
            players = db.query(Player).filter_by(match_id=mid).all()
        assert {p.jersey for p in players} == {7, 9, 11}

    def test_squad_size_cap_applies(self, env, monkeypatch):
        import json

        from scout.config import get_settings
        from scout.pipeline import run
        from tests.test_pipeline_integration import _make_video

        monkeypatch.setenv("SCOUT_MAX_PLAYERS_PER_TEAM", "2")
        get_settings.cache_clear()
        mid = run(str(_make_video(env / "match.mp4")))

        q = json.loads((get_settings().match_dir(mid) / "quality.json").read_text())
        assert q["n_over_squad_size_dropped"] > 0
        assert q["n_rated"] <= 4, "2 teams x 2 players"

    def test_run_still_produces_output_when_no_jersey_is_readable(self, env, monkeypatch):
        """Degrade rather than return an empty report — the coach still sees something."""
        import scout.perception.jersey as jersey
        from scout.db import Rating, get_session
        from scout.pipeline import run
        from tests.test_pipeline_integration import _make_video

        monkeypatch.setattr(jersey, "read_jerseys", lambda v, t, **kw: {})
        mid = run(str(_make_video(env / "match.mp4")))
        with get_session() as db:
            assert db.query(Rating).filter_by(match_id=mid).count() > 0
