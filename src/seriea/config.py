"""Project-wide constants and paths.

Centralises every hard-coded value so that no other module embeds a season code,
URL, or filesystem path directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

#: football-data.co.uk division code for the Italian Serie A.
DIVISION: Final[str] = "I1"

#: URL template for a single season's results CSV.
SOURCE_URL_TEMPLATE: Final[str] = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"

#: Season codes in chronological order. "0708" denotes the 2007-08 campaign.
SEASONS: Final[tuple[str, ...]] = (
    "0708", "0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516",
    "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425",
    "2526",
)

#: Match outcomes in their natural ordinal sequence. The order matters: the
#: Ranked Probability Score treats these as ordered categories, so H-D-A (home
#: win, draw, away win) must never be permuted.
OUTCOMES: Final[tuple[str, str, str]] = ("H", "D", "A")

#: Serie A awards three points for a win and one for a draw.
POINTS_WIN: Final[int] = 3
POINTS_DRAW: Final[int] = 1
POINTS_LOSS: Final[int] = 0

#: Teams per season and the resulting fixture count.
TEAMS_PER_SEASON: Final[int] = 20
MATCHES_PER_SEASON: Final[int] = 380

#: The club this analysis is oriented around.
FOCUS_TEAM: Final[str] = "Fiorentina"
