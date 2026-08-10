"""Match leverage: how much a single fixture moves a club's season.

Not every match matters equally. A fixture is *high leverage* when the club's
season objective — Europe, survival, the title — hinges on it, and low leverage
when the objective is safe or gone regardless. Formally, leverage is the swing
in the probability of achieving the objective between winning and losing:

.. math:: L_m = P(\\text{objective} \\mid \\text{win } m) - P(\\text{objective} \\mid \\text{lose } m)

This is the quantity a sporting director actually needs. It tells you which
fixtures justify spending a tiring key player and which are the ones to rotate
through — a decision no 1X2 probability answers on its own.

Because the simulator samples every fixture independently, conditioning on a
fixture's simulated result after the fact is equivalent to forcing that result
and resampling the rest, so a single simulation run supports every fixture's
leverage at once.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from seriea.simulation.season import SeasonSimulation

#: Minimum simulations behind a conditional probability before it is reported.
#: Below this the Monte Carlo error swamps the estimate.
_MIN_CONDITIONAL_SAMPLES: int = 50


def position_objective(best: int, worst: int) -> Callable[[np.ndarray], np.ndarray]:
    """Build an objective testing whether a finishing position falls in a band.

    Args:
        best: Best position in the band, inclusive (1 is champion).
        worst: Worst position in the band, inclusive.

    Returns:
        A predicate mapping an array of finishing positions to booleans.

    Raises:
        ValueError: If the band is empty or positions are non-positive.
    """
    if best < 1 or worst < best:
        raise ValueError(f"Invalid position band [{best}, {worst}].")

    def objective(positions: np.ndarray) -> np.ndarray:
        return (positions >= best) & (positions <= worst)

    return objective


def match_leverage(
    simulation: SeasonSimulation,
    fixtures: pd.DataFrame,
    team: str,
    objective: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Rank a club's remaining fixtures by how much each swings its objective.

    Args:
        simulation: A completed season simulation.
        fixtures: The remaining fixtures fed to that simulation, aligned
            row-for-row with its sampled scorelines.
        team: Club whose season objective is at stake.
        objective: Predicate on finishing positions, as built by
            :func:`position_objective`.

    Returns:
        Frame with one row per fixture involving the club, carrying the
        conditional objective probabilities given a win, draw and loss, the
        leverage, and the Monte Carlo standard error on that leverage. Sorted by
        leverage, descending.

    Raises:
        KeyError: If the club is absent from the simulation.
        ValueError: If the fixtures and simulation disagree in length.
    """
    if team not in simulation.teams:
        raise KeyError(f"{team!r} is not in this simulation.")
    if len(fixtures) != simulation.home_goals.shape[1]:
        raise ValueError(
            f"fixtures has {len(fixtures)} rows but the simulation holds "
            f"{simulation.home_goals.shape[1]}."
        )

    column = simulation.teams.index(team)
    achieved = objective(simulation.positions[:, column])

    involved = fixtures.reset_index(drop=True)
    involved = involved[(involved["home"] == team) | (involved["away"] == team)]

    records = []
    for position, fixture in involved.iterrows():
        at_home = fixture["home"] == team
        scored = simulation.home_goals[:, position] if at_home else simulation.away_goals[:, position]
        conceded = simulation.away_goals[:, position] if at_home else simulation.home_goals[:, position]

        won, drew, lost = scored > conceded, scored == conceded, scored < conceded
        probability_won = _conditional_mean(achieved, won)
        probability_lost = _conditional_mean(achieved, lost)

        records.append(
            {
                "date": fixture.get("date"),
                "opponent": fixture["away"] if at_home else fixture["home"],
                "venue": "H" if at_home else "A",
                "p_objective_if_win": probability_won,
                "p_objective_if_draw": _conditional_mean(achieved, drew),
                "p_objective_if_loss": probability_lost,
                "leverage": probability_won - probability_lost,
                "leverage_std_error": _difference_standard_error(achieved, won, lost),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "date", "opponent", "venue", "p_objective_if_win",
                "p_objective_if_draw", "p_objective_if_loss", "leverage",
                "leverage_std_error",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values("leverage", ascending=False)
        .reset_index(drop=True)
    )


def _conditional_mean(achieved: np.ndarray, condition: np.ndarray) -> float:
    """Estimate P(objective | condition) from simulation draws.

    Args:
        achieved: Boolean array flagging simulations meeting the objective.
        condition: Boolean array selecting the conditioning event.

    Returns:
        The conditional probability, or NaN if too few draws satisfy the
        condition to support an estimate.
    """
    count = int(condition.sum())
    if count < _MIN_CONDITIONAL_SAMPLES:
        return float("nan")
    return float(achieved[condition].mean())


def _difference_standard_error(
    achieved: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    """Monte Carlo standard error on a difference of conditional probabilities.

    The two conditioning events are disjoint, so the estimates are independent
    and their variances add.

    Args:
        achieved: Boolean array flagging simulations meeting the objective.
        first: Boolean array selecting the first conditioning event.
        second: Boolean array selecting the second.

    Returns:
        The standard error, or NaN if either event is too rare.
    """
    first_count, second_count = int(first.sum()), int(second.sum())
    if first_count < _MIN_CONDITIONAL_SAMPLES or second_count < _MIN_CONDITIONAL_SAMPLES:
        return float("nan")

    first_rate = achieved[first].mean()
    second_rate = achieved[second].mean()
    variance = (
        first_rate * (1.0 - first_rate) / first_count
        + second_rate * (1.0 - second_rate) / second_count
    )
    return float(np.sqrt(variance))
