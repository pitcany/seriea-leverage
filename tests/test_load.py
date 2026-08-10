"""Tests for parsing and validating raw season CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

from seriea.data.load import load_all, load_season, season_start_year
from seriea.data.teams import normalise_name, season_appearances, teams_in_season
from seriea.data.download import raw_path

HEADER = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,B365H,B365D,B365A"
)


def write_season(directory: Path, season: str, rows: list[str]) -> None:
    """Write a synthetic season CSV in the source format."""
    path = raw_path(season, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([HEADER, *rows]) + "\n")


def valid_rows() -> list[str]:
    return [
        "I1,25/08/07,Fiorentina,Juventus,2,0,H,14,9,6,3,7,4,2.10,3.40,3.60",
        "I1,26/08/07,Napoli,Fiorentina,1,1,D,11,12,4,5,5,6,2.50,3.30,2.90",
        "I1,02/09/07,Juventus,Napoli,0,1,A,10,13,3,7,4,8,1.95,3.50,4.00",
    ]


def test_season_start_year_maps_codes_to_calendar_years() -> None:
    assert season_start_year("0708") == 2007
    assert season_start_year("2526") == 2025


def test_season_start_year_rejects_malformed_codes() -> None:
    with pytest.raises(ValueError, match="four digits"):
        season_start_year("07-08")


def test_normalise_name_collapses_whitespace() -> None:
    assert normalise_name("  Hellas   Verona ") == "Hellas Verona"


def test_normalise_name_rejects_blank() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalise_name("   ")


def test_load_season_produces_the_canonical_schema(tmp_path: Path) -> None:
    write_season(tmp_path, "0708", valid_rows())
    frame = load_season("0708", tmp_path)

    assert len(frame) == 3
    assert frame["season"].unique().tolist() == ["0708"]
    assert frame["season_start_year"].unique().tolist() == [2007]
    for column in ("date", "home", "away", "home_goals", "away_goals", "outcome"):
        assert column in frame.columns
    assert frame["date"].is_monotonic_increasing


def test_load_season_parses_two_digit_and_four_digit_dates(tmp_path: Path) -> None:
    write_season(
        tmp_path,
        "1819",
        ["I1,18/08/2018,Fiorentina,Juventus,1,0,H,10,8,4,2,5,3,2.4,3.2,3.1"],
    )
    frame = load_season("1819", tmp_path)
    assert frame.loc[0, "date"].year == 2018
    assert frame.loc[0, "date"].month == 8


def test_load_season_reads_odds_columns(tmp_path: Path) -> None:
    write_season(tmp_path, "0708", valid_rows())
    frame = load_season("0708", tmp_path)
    assert frame.loc[0, "b365_h"] == pytest.approx(2.10)
    # Pinnacle columns are absent from this era and must come through as null.
    assert frame["psc_h"].isna().all()


def test_load_season_rejects_a_result_that_contradicts_the_score(tmp_path: Path) -> None:
    write_season(
        tmp_path,
        "0708",
        ["I1,25/08/07,Fiorentina,Juventus,2,0,A,14,9,6,3,7,4,2.10,3.40,3.60"],
    )
    with pytest.raises(ValueError, match="disagrees with the scoreline"):
        load_season("0708", tmp_path)


def test_load_season_rejects_unknown_outcome_codes(tmp_path: Path) -> None:
    write_season(
        tmp_path,
        "0708",
        ["I1,25/08/07,Fiorentina,Juventus,2,0,X,14,9,6,3,7,4,2.10,3.40,3.60"],
    )
    with pytest.raises(ValueError, match="unrecognised outcome"):
        load_season("0708", tmp_path)


def test_load_season_rejects_a_team_playing_itself(tmp_path: Path) -> None:
    write_season(
        tmp_path,
        "0708",
        ["I1,25/08/07,Fiorentina,Fiorentina,1,1,D,14,9,6,3,7,4,2.10,3.40,3.60"],
    )
    with pytest.raises(ValueError, match="identical teams"):
        load_season("0708", tmp_path)


def test_load_season_raises_when_not_downloaded(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        load_season("9999", tmp_path)


def test_load_all_enforces_a_complete_fixture_list(tmp_path: Path) -> None:
    write_season(tmp_path, "0708", valid_rows())
    with pytest.raises(ValueError, match="expected 380"):
        load_all(("0708",), tmp_path)


def test_load_all_admits_incomplete_seasons_when_asked(tmp_path: Path) -> None:
    write_season(tmp_path, "0708", valid_rows())
    frame = load_all(("0708",), tmp_path, require_complete=False)
    assert len(frame) == 3


def test_teams_in_season_lists_clubs_alphabetically(tmp_path: Path) -> None:
    write_season(tmp_path, "0708", valid_rows())
    frame = load_all(("0708",), tmp_path, require_complete=False)
    assert teams_in_season(frame, "0708") == ("Fiorentina", "Juventus", "Napoli")


def test_season_appearances_counts_distinct_seasons(tmp_path: Path) -> None:
    write_season(tmp_path, "0708", valid_rows())
    write_season(
        tmp_path,
        "0809",
        ["I1,30/08/08,Fiorentina,Milan,3,1,H,12,7,5,2,6,3,2.0,3.3,3.8"],
    )
    frame = load_all(("0708", "0809"), tmp_path, require_complete=False)
    counts = season_appearances(frame)
    assert counts["Fiorentina"] == 2
    assert counts["Milan"] == 1
