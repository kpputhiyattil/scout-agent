"""CSV and Word export, plus the naming rule that makes players identifiable."""
import pytest

from tests.test_pipeline_integration import _make_video, pipeline_env  # noqa: F401


@pytest.fixture
def exported_match(pipeline_env):  # noqa: F811
    from scout.pipeline import run
    return run(str(_make_video(pipeline_env / "match.mp4")))


class TestPlayerNaming:
    def test_unnamed_players_use_team_and_jersey(self, pipeline_env, monkeypatch):  # noqa: F811
        """'B #14' tells a coach who to look at; 'Unknown #14' does not."""
        import scout.perception.jersey as jersey
        monkeypatch.setattr(jersey, "join_roster", lambda j, r: {})   # no roster supplied
        from scout.db import Player, get_session
        from scout.pipeline import run
        mid = run(str(_make_video(pipeline_env / "noroster.mp4")))
        with get_session() as db:
            names = [p.name for p in db.query(Player).filter_by(match_id=mid).all()]
        assert names and not any(n.startswith("Unknown") for n in names)
        assert all(n.startswith(("A ", "B ")) for n in names), names[:3]

    def test_named_players_keep_roster_name(self, pipeline_env, monkeypatch):  # noqa: F811
        import scout.perception.jersey as jersey
        monkeypatch.setattr(jersey, "join_roster", lambda j, r: {k: "Alex Smith" for k in j})
        from scout.db import Player, get_session
        from scout.pipeline import run
        mid = run(str(_make_video(pipeline_env / "m2.mp4")))
        with get_session() as db:
            names = {p.name for p in db.query(Player).filter_by(match_id=mid).all()}
        assert "Alex Smith" in names


class TestCsvExport:
    def test_csv_has_a_row_per_player_and_kpi_columns(self, exported_match, tmp_path):
        import pandas as pd

        from scout.db import Player, get_session
        from scout.report.export import export_csv
        out = export_csv(exported_match, tmp_path / "r.csv")
        df = pd.read_csv(out)
        with get_session() as db:
            n = db.query(Player).filter_by(match_id=exported_match).count()
        assert len(df) == n
        assert {"Team", "Position", "Player", "Minutes", "Rating"} <= set(df.columns)
        assert any(c.startswith("KPI: ") for c in df.columns), "raw evidence must be exported"

    def test_sorted_by_rating_descending(self, exported_match):
        from scout.report.export import ratings_frame
        r = ratings_frame(exported_match).rating.dropna()
        assert list(r) == sorted(r, reverse=True)

    def test_grouped_by_team_then_position(self, exported_match):
        """The file should read top-down: each team's goalkeepers, then defenders, etc."""
        from scout.report.export import ROLE_LABEL, ROLE_ORDER, report_frame
        df = report_frame(exported_match)
        order = {ROLE_LABEL[r]: i for i, r in enumerate(ROLE_ORDER)}
        for team, block in df.groupby("Team", sort=False):
            seq = [order[p] for p in block.Position]
            assert seq == sorted(seq), f"team {team} positions out of order"
        # teams appear as contiguous blocks, not interleaved
        teams = df.Team.tolist()
        assert len(set(teams)) == len([k for k, _ in __import__("itertools").groupby(teams)])

    def test_best_in_each_position_is_rank_one(self, exported_match):
        from scout.report.export import report_frame
        df = report_frame(exported_match)
        for _, block in df.groupby(["Team", "Position"], sort=False):
            assert block["Rank In Position"].iloc[0] == 1
            ratings = block.Rating.dropna().tolist()
            assert ratings == sorted(ratings, reverse=True), "best player must be listed first"


class TestDocxExport:
    def test_document_is_written_and_readable(self, exported_match, tmp_path):
        docx = pytest.importorskip("docx")
        from scout.report.export import export_docx
        out = export_docx(exported_match, tmp_path / "report.docx")
        assert out.exists() and out.stat().st_size > 5000

        text = "\n".join(p.text for p in docx.Document(str(out)).paragraphs)
        assert "Match scouting report" in text
        assert "How to read this report" in text
        assert "percentiles within this squad" in text, "the rating caveat must survive"

    def test_camera_relative_limits_are_stated(self, exported_match, tmp_path):
        """A coach must not read on-ball-only scores as full performance data."""
        docx = pytest.importorskip("docx")
        from scout.report.export import export_docx
        out = export_docx(exported_match, tmp_path / "report.docx")
        text = "\n".join(p.text for p in docx.Document(str(out)).paragraphs)
        assert "Distance covered, speed and positioning were NOT measured" in text

    def test_tables_are_capped_and_grouped(self, exported_match, tmp_path):
        docx = pytest.importorskip("docx")
        from scout.report.export import MAX_ROWS_PER_BLOCK, export_docx
        out = export_docx(exported_match, tmp_path / "report.docx")
        d = docx.Document(str(out))
        assert len(d.tables) >= 2, "expected one table per team-and-position block"
        for t in d.tables:
            assert len(t.rows) <= MAX_ROWS_PER_BLOCK + 1   # + header
        headings = [p.text for p in d.paragraphs if p.style.name.startswith("Heading")]
        assert any(h.startswith("Team ") for h in headings)

    def test_export_all_writes_both(self, exported_match, tmp_path):
        pytest.importorskip("docx")
        from scout.report.export import export_all
        paths = export_all(exported_match, tmp_path)
        assert paths["csv"].exists() and paths["docx"].exists()
