"""Retrieval of raw season CSVs from football-data.co.uk.

Downloads are cached on disk: a season already present is never re-fetched
unless the caller explicitly forces it. This keeps backtests reproducible and
avoids hammering a free data source.
"""

from __future__ import annotations

from pathlib import Path

import requests

from seriea.config import DIVISION, RAW_DIR, SEASONS, SOURCE_URL_TEMPLATE

#: Seasons whose CSVs are complete and stable. A short timeout is fine because
#: each file is well under a megabyte.
_REQUEST_TIMEOUT_SECONDS: int = 30

#: Below this size a response is assumed to be an error page rather than a CSV.
_MIN_PLAUSIBLE_BYTES: int = 10_000


def raw_path(season: str, raw_dir: Path | None = None) -> Path:
    """Return the on-disk location for one season's raw CSV.

    Args:
        season: Season code such as ``"2425"``.
        raw_dir: Directory holding raw downloads. Defaults to the project's
            ``data/raw``.

    Returns:
        Path to the CSV, which may not exist yet.
    """
    directory = RAW_DIR if raw_dir is None else raw_dir
    return directory / f"{DIVISION}_{season}.csv"


def download_season(
    season: str,
    raw_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Fetch one season's CSV, returning the cached copy when present.

    Args:
        season: Season code such as ``"2425"``.
        raw_dir: Destination directory. Defaults to the project's ``data/raw``.
        force: Re-download even when a cached copy exists.

    Returns:
        Path to the downloaded CSV.

    Raises:
        requests.HTTPError: If the source returns a non-success status.
        ValueError: If the response is too small to be a plausible season CSV,
            which indicates an error page rather than data.
    """
    destination = raw_path(season, raw_dir)
    if destination.exists() and not force:
        return destination

    url = SOURCE_URL_TEMPLATE.format(season=season, division=DIVISION)
    response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    if len(response.content) < _MIN_PLAUSIBLE_BYTES:
        raise ValueError(
            f"Response for season {season} was only {len(response.content)} bytes; "
            "expected a full-season CSV. The source may have moved or rate-limited."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def download_all(
    seasons: tuple[str, ...] = SEASONS,
    raw_dir: Path | None = None,
    *,
    force: bool = False,
) -> tuple[Path, ...]:
    """Fetch every requested season.

    Args:
        seasons: Season codes to retrieve.
        raw_dir: Destination directory.
        force: Re-download even when cached copies exist.

    Returns:
        Paths to the downloaded CSVs, in the order requested.
    """
    return tuple(download_season(s, raw_dir, force=force) for s in seasons)
