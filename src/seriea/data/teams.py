"""Team-name normalisation and season-membership helpers.

football-data.co.uk is internally consistent for Serie A: across 2007-08 to
2025-26 every club is spelled the same way in every season it appears, so this
module deliberately does *not* invent an alias table. It normalises whitespace,
validates against the observed roster, and exposes membership helpers used by
the season simulator.
"""

from __future__ import annotations

import pandas as pd


def normalise_name(name: str) -> str:
    """Collapse whitespace in a raw team name.

    Args:
        name: Team name as it appears in the source CSV.

    Returns:
        The name with leading/trailing whitespace stripped and internal runs of
        whitespace collapsed to single spaces.

    Raises:
        ValueError: If the name is empty after normalisation.
    """
    cleaned = " ".join(str(name).split())
    if not cleaned:
        raise ValueError("Team name is empty after normalisation.")
    return cleaned


def teams_in_season(matches: pd.DataFrame, season: str) -> tuple[str, ...]:
    """List the clubs contesting one season, alphabetically.

    Args:
        matches: Canonical match frame from :mod:`seriea.data.load`.
        season: Season code such as ``"2425"``.

    Returns:
        Sorted tuple of club names.
    """
    subset = matches[matches["season"] == season]
    names = set(subset["home"]) | set(subset["away"])
    return tuple(sorted(names))


def season_appearances(matches: pd.DataFrame) -> pd.Series:
    """Count how many seasons each club appears in.

    Promotion and relegation mean many clubs appear only once or twice, which is
    the practical motivation for partial pooling of team-strength parameters.

    Args:
        matches: Canonical match frame.

    Returns:
        Series indexed by club name, holding the season count, sorted
        descending.
    """
    per_season = (
        pd.concat(
            [
                matches[["season", "home"]].rename(columns={"home": "team"}),
                matches[["season", "away"]].rename(columns={"away": "team"}),
            ]
        )
        .drop_duplicates()
        .groupby("team")
        .size()
    )
    return per_season.sort_values(ascending=False)
