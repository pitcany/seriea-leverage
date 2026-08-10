"""Tests for match-leverage computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seriea.simulation.leverage import match_leverage, position_objective
from seriea.simulation.season import simulate_season

TEAMS = ("Fiorentina", "Juventus", "Napoli")
GRID_SIZE = 4


def coin_flip_grid() -> np.ndarray:
    """A grid giving the home side a 1-0 win or a 0-1 loss with equal chance."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[1, 0] = 0.5
    grid[0, 1] = 0.5
    return grid


def empty_standings() -> pd.DataFrame:
    return pd.DataFrame(
        0, index=list(TEAMS), columns=["points", "goals_for", "goals_against"], dtype=int
    ).assign(goal_difference=0)


def test_position_objective_selects_the_band() -> None:
    objective = position_objective(1, 4)
    assert objective(np.array([1, 4, 5, 20])).tolist() == [True, True, False, False]


def test_position_objective_rejects_an_empty_band() -> None:
    with pytest.raises(ValueError, match="Invalid position band"):
        position_objective(5, 2)


def test_leverage_is_zero_when_the_objective_is_certain() -> None:
    """With three clubs, finishing in the top three is guaranteed for everyone.

    No result can change an outcome that is already assured, so every fixture
    must carry exactly zero leverage.
    """
    fixtures = pd.DataFrame(
        [
            {"home": "Fiorentina", "away": "Juventus", "date": pd.Timestamp("2026-01-01")},
            {"home": "Napoli", "away": "Fiorentina", "date": pd.Timestamp("2026-01-08")},
        ]
    )
    grids = np.stack([coin_flip_grid()] * 2)
    simulation = simulate_season(empty_standings(), fixtures, grids, n_simulations=2000, seed=3)

    table = match_leverage(simulation, fixtures, "Fiorentina", position_objective(1, 3))
    assert np.allclose(table["leverage"].to_numpy(), 0.0)


def test_leverage_is_positive_when_results_decide_the_title() -> None:
    fixtures = pd.DataFrame(
        [
            {"home": "Fiorentina", "away": "Juventus", "date": pd.Timestamp("2026-01-01")},
            {"home": "Napoli", "away": "Fiorentina", "date": pd.Timestamp("2026-01-08")},
        ]
    )
    grids = np.stack([coin_flip_grid()] * 2)
    simulation = simulate_season(empty_standings(), fixtures, grids, n_simulations=4000, seed=5)

    table = match_leverage(simulation, fixtures, "Fiorentina", position_objective(1, 1))
    assert (table["leverage"] > 0).all()
    assert (table["leverage"] <= 1.0).all()
    assert (table["p_objective_if_win"] > table["p_objective_if_loss"]).all()


def test_leverage_covers_only_the_club_s_own_fixtures() -> None:
    fixtures = pd.DataFrame(
        [
            {"home": "Fiorentina", "away": "Juventus", "date": pd.Timestamp("2026-01-01")},
            {"home": "Napoli", "away": "Juventus", "date": pd.Timestamp("2026-01-08")},
        ]
    )
    grids = np.stack([coin_flip_grid()] * 2)
    simulation = simulate_season(empty_standings(), fixtures, grids, n_simulations=1000, seed=9)

    table = match_leverage(simulation, fixtures, "Fiorentina", position_objective(1, 1))
    assert len(table) == 1
    assert table.loc[0, "opponent"] == "Juventus"
    assert table.loc[0, "venue"] == "H"


def test_leverage_records_venue_for_away_fixtures() -> None:
    fixtures = pd.DataFrame(
        [{"home": "Napoli", "away": "Fiorentina", "date": pd.Timestamp("2026-01-08")}]
    )
    grids = np.stack([coin_flip_grid()])
    simulation = simulate_season(empty_standings(), fixtures, grids, n_simulations=1000, seed=11)

    table = match_leverage(simulation, fixtures, "Fiorentina", position_objective(1, 1))
    assert table.loc[0, "venue"] == "A"
    assert table.loc[0, "opponent"] == "Napoli"


def test_leverage_rejects_unknown_club() -> None:
    fixtures = pd.DataFrame([{"home": "Fiorentina", "away": "Juventus"}])
    grids = np.stack([coin_flip_grid()])
    simulation = simulate_season(empty_standings(), fixtures, grids, n_simulations=100, seed=1)
    with pytest.raises(KeyError):
        match_leverage(simulation, fixtures, "Milan", position_objective(1, 1))
