"""Fiorentina match leverage, as the tool would have run mid-season.

Takes a decision date inside 2025-26, fits the model on everything before it,
simulates the remainder of the season, and ranks Fiorentina's remaining fixtures
by how much each one moves the club's survival probability.

Because that season is now complete, the output can be checked against what
actually happened — which is the point. A leverage table is only worth acting on
if the season probabilities behind it were honest at the time.

Run: ``python scripts/fiorentina_leverage.py [--as-of 2026-03-01]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from seriea.config import FOCUS_TEAM, REPORTS_DIR
from seriea.data.load import load_all
from seriea.data.teams import teams_in_season
from seriea.models.dixon_coles import DixonColesForecaster
from seriea.simulation.leverage import match_leverage, position_objective
from seriea.simulation.season import current_standings, simulate_season

SEASON = "2526"
DEFAULT_AS_OF = "2026-03-01"

#: Serie A relegates the bottom three of twenty.
RELEGATION_BAND: tuple[int, int] = (18, 20)
SURVIVAL_BAND: tuple[int, int] = (1, 17)
EUROPE_BAND: tuple[int, int] = (1, 6)

#: Training history for the mid-season fit, in days.
TRAINING_WINDOW_DAYS: int = 1_460

FALLBACK_DECAY: float = 0.002


def load_decay_rate() -> float:
    """Read the validated decay rate, falling back to a sensible default."""
    path = Path(REPORTS_DIR) / "decay_tuning.json"
    if not path.exists():
        return FALLBACK_DECAY
    return float(json.loads(path.read_text())["best"]["decay_rate"])


def main() -> None:
    """Produce the leverage table and season projection for Fiorentina."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF, help="Decision date, YYYY-MM-DD.")
    parser.add_argument("--simulations", type=int, default=20_000)
    arguments = parser.parse_args()

    as_of = pd.Timestamp(arguments.as_of)
    matches = load_all()
    season = matches[matches["season"] == SEASON]
    teams = teams_in_season(matches, SEASON)

    played = season[season["date"] < as_of]
    remaining = season[season["date"] >= as_of].reset_index(drop=True)
    if remaining.empty:
        raise SystemExit(f"No fixtures remain on or after {as_of.date()}.")

    print(f"Decision date {as_of.date()}: {len(played)} played, {len(remaining)} remaining\n")

    history = matches[
        (matches["date"] < as_of)
        & (matches["date"] >= as_of - pd.Timedelta(days=TRAINING_WINDOW_DAYS))
    ]
    forecaster = DixonColesForecaster(decay_rate=load_decay_rate()).fit(history, as_of)

    standings = current_standings(played, teams)
    grids = forecaster.predict_score_grid(remaining)
    simulation = simulate_season(
        standings, remaining, grids, n_simulations=arguments.simulations
    )

    # ---- where the club stood ------------------------------------------------
    table = standings.sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False
    )
    table.insert(0, "pos", range(1, len(table) + 1))
    print("Table at the decision date:")
    print(table.to_string())

    # ---- season projection ---------------------------------------------------
    own_remaining = remaining[
        (remaining["home"] == FOCUS_TEAM) | (remaining["away"] == FOCUS_TEAM)
    ]
    print(
        f"\n{FOCUS_TEAM} projection with {len(own_remaining)} of its own fixtures left "
        f"({len(remaining)} across the division):"
    )
    print(f"  expected final points : {simulation.expected_points(FOCUS_TEAM):.1f}")
    for label, (best, worst) in (
        ("survival (1-17)", SURVIVAL_BAND),
        ("relegation (18-20)", RELEGATION_BAND),
        ("Europe (1-6)", EUROPE_BAND),
    ):
        probability = simulation.position_probability(FOCUS_TEAM, best, worst)
        print(f"  P({label:<19}) = {probability:6.1%}")

    # ---- leverage ------------------------------------------------------------
    leverage = match_leverage(
        simulation, remaining, FOCUS_TEAM, position_objective(*SURVIVAL_BAND)
    )
    print(f"\n{FOCUS_TEAM} remaining fixtures ranked by leverage on SURVIVAL:")
    display = leverage.assign(date=lambda frame: frame["date"].dt.date)
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- what actually happened ---------------------------------------------
    final = current_standings(season, teams).sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False
    )
    final.insert(0, "pos", range(1, len(final) + 1))
    actual_position = int(final.loc[FOCUS_TEAM, "pos"])
    actual_points = int(final.loc[FOCUS_TEAM, "points"])
    print(
        f"\nActual outcome: {FOCUS_TEAM} finished {actual_position} on {actual_points} points."
    )

    output = Path(REPORTS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    leverage.to_csv(output / "fiorentina_leverage.csv", index=False)
    print(f"Wrote {output / 'fiorentina_leverage.csv'}")


if __name__ == "__main__":
    main()
