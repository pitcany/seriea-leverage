"""Tests for the season simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seriea.simulation.season import current_standings, simulate_season

TEAMS = ("Fiorentina", "Juventus", "Napoli")
GRID_SIZE = 4


def certain_scoreline(home_goals: int, away_goals: int) -> np.ndarray:
    """Build a score grid placing all probability on one scoreline."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[home_goals, away_goals] = 1.0
    return grid


def empty_standings() -> pd.DataFrame:
    return pd.DataFrame(
        0, index=list(TEAMS), columns=["points", "goals_for", "goals_against"], dtype=int
    ).assign(goal_difference=0)


def test_current_standings_awards_three_for_a_win_and_one_for_a_draw() -> None:
    played = pd.DataFrame(
        [
            {"home": "Fiorentina", "away": "Juventus", "home_goals": 2, "away_goals": 0},
            {"home": "Napoli", "away": "Fiorentina", "home_goals": 1, "away_goals": 1},
        ]
    )
    table = current_standings(played, TEAMS)

    assert table.loc["Fiorentina", "points"] == 4
    assert table.loc["Juventus", "points"] == 0
    assert table.loc["Napoli", "points"] == 1
    assert table.loc["Fiorentina", "goal_difference"] == 2
    assert table.loc["Juventus", "goal_difference"] == -2


def test_current_standings_includes_teams_with_no_matches() -> None:
    played = pd.DataFrame(
        [{"home": "Fiorentina", "away": "Juventus", "home_goals": 1, "away_goals": 0}]
    )
    table = current_standings(played, TEAMS)
    assert table.loc["Napoli", "points"] == 0


def test_deterministic_grids_produce_the_exact_final_table() -> None:
    fixtures = pd.DataFrame(
        [
            {"home": "Fiorentina", "away": "Juventus"},
            {"home": "Fiorentina", "away": "Napoli"},
        ]
    )
    grids = np.stack([certain_scoreline(2, 0), certain_scoreline(3, 0)])

    simulation = simulate_season(
        empty_standings(), fixtures, grids, n_simulations=50, seed=1
    )

    assert simulation.expected_points("Fiorentina") == pytest.approx(6.0)
    assert simulation.expected_points("Juventus") == pytest.approx(0.0)
    assert simulation.position_probability("Fiorentina", 1, 1) == pytest.approx(1.0)
    # Juventus lost by two, Napoli by three, so Juventus finishes above on goal difference.
    assert simulation.position_probability("Juventus", 2, 2) == pytest.approx(1.0)
    assert simulation.position_probability("Napoli", 3, 3) == pytest.approx(1.0)


def test_positions_are_a_permutation_in_every_simulation() -> None:
    fixtures = pd.DataFrame(
        [
            {"home": "Fiorentina", "away": "Juventus"},
            {"home": "Napoli", "away": "Fiorentina"},
        ]
    )
    grids = np.stack([np.full((GRID_SIZE, GRID_SIZE), 1 / GRID_SIZE**2)] * 2)

    simulation = simulate_season(
        empty_standings(), fixtures, grids, n_simulations=200, seed=7
    )
    expected = np.arange(1, len(TEAMS) + 1)
    for row in simulation.positions:
        assert np.array_equal(np.sort(row), expected)


def test_simulation_is_reproducible_under_a_fixed_seed() -> None:
    fixtures = pd.DataFrame([{"home": "Fiorentina", "away": "Juventus"}])
    grids = np.stack([np.full((GRID_SIZE, GRID_SIZE), 1 / GRID_SIZE**2)])

    first = simulate_season(empty_standings(), fixtures, grids, n_simulations=100, seed=42)
    second = simulate_season(empty_standings(), fixtures, grids, n_simulations=100, seed=42)
    assert np.array_equal(first.points, second.points)


def test_simulate_season_rejects_misaligned_grids() -> None:
    fixtures = pd.DataFrame([{"home": "Fiorentina", "away": "Juventus"}])
    grids = np.stack([certain_scoreline(1, 0), certain_scoreline(2, 0)])
    with pytest.raises(ValueError, match="score_grids"):
        simulate_season(empty_standings(), fixtures, grids, n_simulations=10)


def test_simulate_season_rejects_unknown_clubs() -> None:
    fixtures = pd.DataFrame([{"home": "Fiorentina", "away": "Atalanta"}])
    grids = np.stack([certain_scoreline(1, 0)])
    with pytest.raises(ValueError, match="absent from the standings"):
        simulate_season(empty_standings(), fixtures, grids, n_simulations=10)


def test_position_probability_rejects_unknown_club() -> None:
    fixtures = pd.DataFrame([{"home": "Fiorentina", "away": "Juventus"}])
    grids = np.stack([certain_scoreline(1, 0)])
    simulation = simulate_season(empty_standings(), fixtures, grids, n_simulations=10)
    with pytest.raises(KeyError):
        simulation.position_probability("Milan", 1, 1)
