"""Parsing of raw season CSVs into one canonical, validated match frame.

The source files drift over time: the date format switches from ``dd/mm/yy`` to
``dd/mm/yyyy`` at 2018-19, Pinnacle prices appear from 2012-13, and closing
prices appear later still. All of that is absorbed here so downstream modules
see a single stable schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seriea.config import MATCHES_PER_SEASON, OUTCOMES, RAW_DIR, SEASONS
from seriea.data.download import raw_path
from seriea.data.teams import normalise_name

#: Source column -> canonical column, for fields present in every season.
_CORE_COLUMNS: dict[str, str] = {
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "outcome",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
}

#: Source column -> canonical column, for price fields that may be absent.
#: ``b365_*`` are Bet365 opening prices (all seasons); ``ps_*`` are Pinnacle
#: opening and ``psc_*`` Pinnacle closing prices (2012-13 onward).
_ODDS_COLUMNS: dict[str, str] = {
    "B365H": "b365_h",
    "B365D": "b365_d",
    "B365A": "b365_a",
    "PSH": "ps_h",
    "PSD": "ps_d",
    "PSA": "ps_a",
    "PSCH": "psc_h",
    "PSCD": "psc_d",
    "PSCA": "psc_a",
}

_COUNT_COLUMNS: tuple[str, ...] = (
    "home_goals", "away_goals", "home_shots", "away_shots",
    "home_sot", "away_sot", "home_corners", "away_corners",
)


def season_start_year(season: str) -> int:
    """Convert a season code to the calendar year the season began.

    Args:
        season: Four-character code such as ``"0708"`` or ``"2526"``.

    Returns:
        The starting year, e.g. 2007 for ``"0708"``.

    Raises:
        ValueError: If the code is not four digits.
    """
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"Season code must be four digits, got {season!r}.")
    return 2000 + int(season[:2])


def load_season(season: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse one season's CSV into the canonical schema.

    Args:
        season: Season code such as ``"2425"``.
        raw_dir: Directory holding raw downloads. Defaults to ``data/raw``.

    Returns:
        A frame with one row per match, sorted by date then home team.

    Raises:
        FileNotFoundError: If the season has not been downloaded.
        ValueError: If required columns are missing or the data fails
            validation.
    """
    path = raw_path(season, raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Season {season} not downloaded. Run seriea.data.download.download_all() first."
        )

    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    missing = sorted(set(_CORE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Season {season} is missing required columns: {missing}")

    frame = raw[list(_CORE_COLUMNS)].rename(columns=_CORE_COLUMNS).copy()

    # dayfirst covers both dd/mm/yy and dd/mm/yyyy; the format genuinely varies
    # across seasons so an explicit single format would reject half the corpus.
    frame["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")

    for source_column, canonical in _ODDS_COLUMNS.items():
        frame[canonical] = (
            pd.to_numeric(raw[source_column], errors="coerce")
            if source_column in raw.columns
            else pd.NA
        )
        frame[canonical] = frame[canonical].astype("Float64")

    frame["home"] = frame["home"].map(normalise_name)
    frame["away"] = frame["away"].map(normalise_name)
    frame["outcome"] = frame["outcome"].astype(str).str.strip().str.upper()

    for column in _COUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    frame["season"] = season
    frame["season_start_year"] = season_start_year(season)

    ordered = frame.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    _validate_season(ordered, season)
    return ordered


def _validate_season(frame: pd.DataFrame, season: str) -> None:
    """Fail fast on structurally impossible season data.

    Args:
        frame: Canonical frame for a single season.
        season: Season code, used only in error messages.

    Raises:
        ValueError: If the outcome codes are unrecognised, the recorded outcome
            disagrees with the goals, or a team meets itself.
    """
    unknown = set(frame["outcome"]) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Season {season} has unrecognised outcome codes: {sorted(unknown)}")

    derived = pd.Series("D", index=frame.index, dtype=object)
    derived[frame["home_goals"] > frame["away_goals"]] = "H"
    derived[frame["home_goals"] < frame["away_goals"]] = "A"
    disagreements = int((derived != frame["outcome"]).sum())
    if disagreements:
        raise ValueError(
            f"Season {season} has {disagreements} matches where the recorded result "
            "disagrees with the scoreline."
        )

    self_matches = int((frame["home"] == frame["away"]).sum())
    if self_matches:
        raise ValueError(f"Season {season} has {self_matches} matches with identical teams.")


def load_all(
    seasons: tuple[str, ...] = SEASONS,
    raw_dir: Path | None = None,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Parse and concatenate every requested season.

    Args:
        seasons: Season codes to load, in chronological order.
        raw_dir: Directory holding raw downloads.
        require_complete: Raise if any season does not hold the full 380-match
            fixture list. Set False to admit a season still in progress.

    Returns:
        A single frame sorted by date, with a fresh integer index.

    Raises:
        ValueError: If ``require_complete`` is set and a season is incomplete.
    """
    frames = []
    for season in seasons:
        frame = load_season(season, raw_dir)
        if require_complete and len(frame) != MATCHES_PER_SEASON:
            raise ValueError(
                f"Season {season} holds {len(frame)} matches, expected {MATCHES_PER_SEASON}. "
                "Pass require_complete=False to admit an in-progress season."
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def default_raw_dir() -> Path:
    """Return the project's raw-data directory.

    Returns:
        Path to ``data/raw``.
    """
    return RAW_DIR
