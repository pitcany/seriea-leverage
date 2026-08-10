"""Monte Carlo simulation of a Serie A season.

Match-level probabilities are only useful to a club once they are aggregated
into the quantities that actually drive decisions: the chance of reaching
Europe, the chance of going down, the chance of the title. That aggregation is
what turns a forecast into a plan.

Scorelines rather than outcomes are sampled, using the full Dixon-Coles score
grid, so goal difference emerges naturally and can break ties. Serie A's first
tiebreaker is head-to-head record; this simulator applies the simpler
points-then-goal-difference-then-goals-scored order, which agrees with the
official ordering in the large majority of cases and is standard practice for
season simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seriea.config import POINTS_DRAW, POINTS_WIN

#: Default number of simulated seasons.
DEFAULT_SIMULATIONS: int = 20_000

#: Seed for reproducibility.
DEFAULT_SEED: int = 20260810


@dataclass(frozen=True)
class SeasonSimulation:
    """Results of simulating a season to its conclusion.

    Attributes:
        teams: Club names, indexing the columns of the arrays below.
        points: Final points, shape ``(n_simulations, n_teams)``.
        positions: Final league positions where 1 is champion, same shape.
        home_goals: Sampled home goals per fixture, shape
            ``(n_simulations, n_fixtures)``.
        away_goals: Sampled away goals per fixture, same shape.
    """

    teams: tuple[str, ...]
    points: np.ndarray
    positions: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray

    def position_probability(self, team: str, best: int, worst: int) -> float:
        """Probability a club finishes within a position band.

        Args:
            team: Club name.
            best: Best position in the band, inclusive (1 is champion).
            worst: Worst position in the band, inclusive.

        Returns:
            Share of simulations finishing in ``[best, worst]``.

        Raises:
            KeyError: If the club did not appear in the simulation.
        """
        if team not in self.teams:
            raise KeyError(f"{team!r} is not in this simulation.")
        column = self.teams.index(team)
        finishing = self.positions[:, column]
        return float(((finishing >= best) & (finishing <= worst)).mean())

    def expected_points(self, team: str) -> float:
        """Mean simulated points total for a club.

        Args:
            team: Club name.

        Returns:
            Mean final points across simulations.

        Raises:
            KeyError: If the club did not appear in the simulation.
        """
        if team not in self.teams:
            raise KeyError(f"{team!r} is not in this simulation.")
        return float(self.points[:, self.teams.index(team)].mean())


def current_standings(played: pd.DataFrame, teams: tuple[str, ...]) -> pd.DataFrame:
    """Build the league table from completed matches.

    Args:
        played: Canonical frame of matches already played this season.
        teams: Every club contesting the season, including any yet to play.

    Returns:
        Frame indexed by club with ``points``, ``goals_for``, ``goals_against``
        and ``goal_difference`` columns.
    """
    table = pd.DataFrame(
        0,
        index=list(teams),
        columns=["points", "goals_for", "goals_against"],
        dtype=int,
    )

    for _, match in played.iterrows():
        home, away = match["home"], match["away"]
        home_goals, away_goals = int(match["home_goals"]), int(match["away_goals"])

        table.loc[home, "goals_for"] += home_goals
        table.loc[home, "goals_against"] += away_goals
        table.loc[away, "goals_for"] += away_goals
        table.loc[away, "goals_against"] += home_goals

        if home_goals > away_goals:
            table.loc[home, "points"] += POINTS_WIN
        elif home_goals < away_goals:
            table.loc[away, "points"] += POINTS_WIN
        else:
            table.loc[home, "points"] += POINTS_DRAW
            table.loc[away, "points"] += POINTS_DRAW

    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    return table


def sample_scorelines(
    score_grids: np.ndarray, n_simulations: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw scorelines for each fixture from its joint score distribution.

    Args:
        score_grids: Array of shape ``(n_fixtures, g, g)`` where entry
            ``[i, x, y]`` is the probability fixture ``i`` finishes ``x-y``.
        n_simulations: Number of seasons to simulate.
        rng: Random generator.

    Returns:
        Tuple of home and away goal arrays, each of shape
        ``(n_simulations, n_fixtures)``.
    """
    n_fixtures, grid_size = score_grids.shape[0], score_grids.shape[1]
    flattened = score_grids.reshape(n_fixtures, -1)
    cumulative = np.cumsum(flattened, axis=1)
    # Guard against floating-point drift leaving the final bin below 1.
    cumulative[:, -1] = 1.0

    draws = rng.random((n_simulations, n_fixtures))
    # searchsorted per fixture: each column has its own cumulative distribution.
    cells = np.empty((n_simulations, n_fixtures), dtype=np.int64)
    for fixture in range(n_fixtures):
        cells[:, fixture] = np.searchsorted(cumulative[fixture], draws[:, fixture], side="right")
    cells = np.clip(cells, 0, grid_size * grid_size - 1)

    return cells // grid_size, cells % grid_size


def simulate_season(
    standings: pd.DataFrame,
    fixtures: pd.DataFrame,
    score_grids: np.ndarray,
    *,
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> SeasonSimulation:
    """Play out the remaining fixtures many times and record final tables.

    Args:
        standings: Current table, indexed by club.
        fixtures: Remaining fixtures with ``home`` and ``away`` columns, aligned
            row-for-row with ``score_grids``.
        score_grids: Joint score distributions for those fixtures.
        n_simulations: Number of seasons to simulate.
        seed: Random seed.

    Returns:
        The completed simulation.

    Raises:
        ValueError: If the fixtures and score grids disagree in length, or a
            fixture names a club absent from the standings.
    """
    if len(fixtures) != score_grids.shape[0]:
        raise ValueError(
            f"fixtures has {len(fixtures)} rows but score_grids has {score_grids.shape[0]}."
        )

    teams = tuple(standings.index)
    position_of = {team: index for index, team in enumerate(teams)}

    unknown = (set(fixtures["home"]) | set(fixtures["away"])) - set(teams)
    if unknown:
        raise ValueError(f"Fixtures reference clubs absent from the standings: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    home_goals, away_goals = sample_scorelines(score_grids, n_simulations, rng)

    points = np.tile(standings["points"].to_numpy(dtype=float), (n_simulations, 1))
    goals_for = np.tile(standings["goals_for"].to_numpy(dtype=float), (n_simulations, 1))
    goals_against = np.tile(standings["goals_against"].to_numpy(dtype=float), (n_simulations, 1))

    home_index = np.array([position_of[team] for team in fixtures["home"]])
    away_index = np.array([position_of[team] for team in fixtures["away"]])

    home_win = home_goals > away_goals
    away_win = home_goals < away_goals
    draw = home_goals == away_goals

    for fixture in range(len(fixtures)):
        h, a = home_index[fixture], away_index[fixture]
        points[:, h] += home_win[:, fixture] * POINTS_WIN + draw[:, fixture] * POINTS_DRAW
        points[:, a] += away_win[:, fixture] * POINTS_WIN + draw[:, fixture] * POINTS_DRAW
        goals_for[:, h] += home_goals[:, fixture]
        goals_against[:, h] += away_goals[:, fixture]
        goals_for[:, a] += away_goals[:, fixture]
        goals_against[:, a] += home_goals[:, fixture]

    positions = _rank_tables(points, goals_for - goals_against, goals_for)
    return SeasonSimulation(teams, points, positions, home_goals, away_goals)


def _rank_tables(
    points: np.ndarray, goal_difference: np.ndarray, goals_for: np.ndarray
) -> np.ndarray:
    """Convert simulated totals into finishing positions.

    Ties are broken on goal difference then goals scored; any residual tie is
    resolved arbitrarily but deterministically, which affects a negligible
    fraction of simulations.

    Args:
        points: Final points, shape ``(n_simulations, n_teams)``.
        goal_difference: Final goal difference, same shape.
        goals_for: Final goals scored, same shape.

    Returns:
        Integer positions where 1 is champion, same shape.
    """
    # A single lexicographic key: points dominate, then goal difference, then
    # goals scored. The multipliers are far larger than any attainable value in
    # the lower-order terms, so the ordering is exact.
    key = points * 1e6 + goal_difference * 1e3 + goals_for
    order = np.argsort(-key, axis=1, kind="stable")

    positions = np.empty_like(order)
    ranks = np.arange(1, points.shape[1] + 1)
    np.put_along_axis(positions, order, np.tile(ranks, (points.shape[0], 1)), axis=1)
    return positions
