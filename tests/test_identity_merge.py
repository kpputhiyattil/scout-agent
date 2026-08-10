"""Fragment merging: turning many short tracks back into few players.

Trackers lose identity on panning/cut footage — a 5-minute clip produced 1239
track ids for ~12 visible players. The jersey number is the only stable anchor,
so fragments reading the same number for the same team are one player.
"""
import pandas as pd

from scout.perception.jersey import Read, merge_map, vote_jerseys


class TestMergeMap:
    def test_same_number_same_team_merges(self):
        jerseys = {1: 7, 5: 7, 9: 7}
        teams = {1: "A", 5: "A", 9: "A"}
        assert set(merge_map(jerseys, teams).values()) == {1}, "all fragments -> lowest id"

    def test_same_number_different_teams_stay_apart(self):
        # both squads have a #7 — merging them would invent a player
        m = merge_map({1: 7, 2: 7}, {1: "A", 2: "B"})
        assert m[1] != m[2]

    def test_unnumbered_fragments_are_not_merged(self):
        # no jersey read => no evidence => leave alone rather than guess
        assert merge_map({}, {1: "A", 2: "A"}) == {}

    def test_distinct_numbers_never_merge(self):
        m = merge_map({1: 7, 2: 9}, {1: "A", 2: "A"})
        assert m[1] != m[2]

    def test_canonical_id_is_stable(self):
        a = merge_map({9: 4, 3: 4, 6: 4}, {9: "B", 3: "B", 6: "B"})
        b = merge_map({3: 4, 6: 4, 9: 4}, {3: "B", 6: "B", 9: "B"})
        assert a == b, "merge must not depend on dict ordering"


class TestMergeAppliedToTracks:
    def test_fragments_accumulate_into_one_player(self):
        from scout.pipeline import _apply_merge
        # same child tracked as 1 (frames 0-99) then 2 (frames 100-199)
        tracks = pd.DataFrame({
            "frame": list(range(100)) + list(range(100, 200)),
            "track_id": [1] * 100 + [2] * 100,
            "x_m": 1.0, "y_m": 1.0,
        })
        identity = {"merge": {"1": 1, "2": 1}}
        out = _apply_merge(tracks, identity)
        assert out.track_id.nunique() == 1
        span = out.frame.max() - out.frame.min()
        assert span == 199, "merged player spans both fragments"

    def test_no_merge_map_is_a_no_op(self):
        from scout.pipeline import _apply_merge
        tracks = pd.DataFrame({"frame": [0, 1], "track_id": [3, 4]})
        assert _apply_merge(tracks, {}).track_id.tolist() == [3, 4]

    def test_overlapping_fragments_collapse_to_one_row_per_frame(self):
        """Two fragments with the same number can coexist in a frame; if both survive,
        every (frame, track_id) lookup downstream returns two rows and crashes."""
        from scout.pipeline import _apply_merge
        tracks = pd.DataFrame({
            "frame": [0, 0, 1, 1],
            "track_id": [1, 2, 1, 2],
            "x1": [0, 0, 0, 0], "y1": [0, 0, 0, 0],
            "x2": [10, 30, 10, 30], "y2": [10, 30, 10, 30],   # id 2 has the larger box
            "x_m": [1.0, 2.0, 1.0, 2.0], "y_m": [1.0, 2.0, 1.0, 2.0],
        })
        out = _apply_merge(tracks, {"merge": {"1": 1, "2": 1}})
        assert not out.duplicated(subset=["frame", "track_id"]).any()
        assert len(out) == 2
        assert out.x_m.tolist() == [2.0, 2.0], "keeps the largest detection"

    def test_duel_detection_survives_merged_ids(self):
        """Regression: merged tracks used to make detect_duels raise
        'The truth value of a Series is ambiguous'."""
        from scout.analytics.events import detect_duels
        tracks = pd.DataFrame({
            "frame": [10, 10, 10],
            "track_id": [1, 1, 2],          # duplicate id 1 in the same frame
            "x_m": [5.0, 5.1, 6.0], "y_m": [5.0, 5.0, 5.0],
        })
        spells = pd.DataFrame({"start": [0, 10], "owner": [1, 2], "team": ["A", "B"]})
        duels = detect_duels(spells, tracks, fps=25.0)
        assert len(duels) == 1 and duels.winner.iat[0] == 2


class TestVotingUnchanged:
    def test_weak_evidence_still_yields_nothing(self):
        # score = conf * sqrt(area) = 0.2 * 3 = 0.6, below min_score
        assert vote_jerseys([Read(1, 7, 0.2, 9.0)], min_score=1.0) == {}

    def test_split_vote_yields_nothing(self):
        # two numbers read equally often: no winner clears the margin
        reads = [Read(1, 7, 0.9, 400.0), Read(1, 8, 0.9, 400.0)]
        assert vote_jerseys(reads) == {}

    def test_consistent_reads_win(self):
        reads = [Read(1, 7, 0.9, 400.0) for _ in range(5)] + [Read(1, 1, 0.3, 100.0)]
        assert vote_jerseys(reads) == {1: 7}
