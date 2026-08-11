"""Does a shot-based signal add anything the goals model and the market miss?

The headline result of this project is that a goals-based Dixon-Coles model
earns zero weight alongside a de-vigged closing price. That is a finding about
*goals*, and goals are the noisiest thing a football match produces. This
experiment re-runs the same question with shots on target as the target
variable — the pre-xG proxy for chance creation — using the identical
walk-forward protocol.

Three pools are fitted, all on the validation period only:

* market + shots — does the shot signal beat the closing price on its own terms?
* Dixon-Coles + shots — does it add to the goals model, ignoring the market?
* market + Dixon-Coles + shots — the joint test, which can surface information
  that pairwise pooling hides.

Run: ``python scripts/run_shots_experiment.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from seriea.config import OUTCOMES, REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.backtest import forecast_matrix, rolling_origin_forecasts
from seriea.evaluation.inference import bootstrap_mean, paired_bootstrap_difference
from seriea.evaluation.metrics import accuracy, log_loss, ranked_probability_score
from seriea.models.blend import fit_pool_weights, logarithmic_pool_many
from seriea.models.dixon_coles import DixonColesForecaster
from seriea.models.market import market_probabilities
from seriea.models.shots import ShotsForecaster

BACKTEST_START = pd.Timestamp("2013-08-01")
TEST_START = pd.Timestamp("2019-08-01")
FALLBACK_DECAY: float = 0.002


def load_decay_rate() -> float:
    """Read the validated decay rate, falling back to the shipped default."""
    path = Path(REPORTS_DIR) / "decay_tuning.json"
    if not path.exists():
        return FALLBACK_DECAY
    return float(json.loads(path.read_text())["best"]["decay_rate"])


def score(name: str, probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, object]:
    """Score one forecaster with a bootstrap interval on its RPS."""
    rps = ranked_probability_score(probabilities, outcomes)
    interval = bootstrap_mean(rps)
    return {
        "model": name,
        "rps": float(rps.mean()),
        "rps_lower": interval.lower,
        "rps_upper": interval.upper,
        "log_loss": float(log_loss(probabilities, outcomes).mean()),
        "accuracy": accuracy(probabilities, outcomes),
    }


def main() -> None:
    """Run the walk-forward comparison and write results to ``reports/``."""
    decay = load_decay_rate()
    matches = load_all()
    print(f"decay rate: {decay}\nbuilding walk-forward forecasts (this takes a few minutes)\n")

    goals_forecasts = rolling_origin_forecasts(
        matches, lambda: DixonColesForecaster(decay_rate=decay), BACKTEST_START
    )
    shots_forecasts = rolling_origin_forecasts(
        matches, lambda: ShotsForecaster(decay_rate=decay), BACKTEST_START
    )

    keys = ["date", "home", "away"]
    merged = goals_forecasts.merge(
        shots_forecasts[keys + [f"p_{o}" for o in OUTCOMES]],
        on=keys,
        suffixes=("", "_shots"),
    )

    market = market_probabilities(merged)
    complete = np.isfinite(market).all(axis=1)
    merged, market = merged[complete].reset_index(drop=True), market[complete]

    goals = forecast_matrix(merged)
    shots = merged[[f"p_{o}_shots" for o in OUTCOMES]].to_numpy(dtype=float)
    outcomes = merged["outcome"].to_numpy(dtype=object)
    is_test = (merged["date"] >= TEST_START).to_numpy()

    print(f"validation matches: {int((~is_test).sum())}   test matches: {int(is_test.sum())}\n")

    # ---- fit every pool on validation only -----------------------------------
    pools = {
        "market + shots": [market, shots],
        "Dixon-Coles + shots": [goals, shots],
        "market + Dixon-Coles + shots": [market, goals, shots],
    }
    fitted: dict[str, np.ndarray] = {}
    print("=" * 78)
    print("POOL WEIGHTS (fitted on validation 2013-19)")
    print("=" * 78)
    for name, members in pools.items():
        weights = fit_pool_weights(
            [member[~is_test] for member in members], outcomes[~is_test]
        )
        fitted[name] = weights
        labels = name.split(" + ")
        print(f"{name:<32} " + "  ".join(f"{l}={w:.2f}" for l, w in zip(labels, weights)))

    # ---- score everything on test -------------------------------------------
    candidates = {
        "Shots on target": shots,
        "Dixon-Coles (goals)": goals,
        "Market (Shin de-vigged)": market,
    }
    for name, members in pools.items():
        candidates[f"Pool: {name}"] = logarithmic_pool_many(members, fitted[name])

    rows = [score(n, v[is_test], outcomes[is_test]) for n, v in candidates.items()]
    table = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)

    print("\n" + "=" * 78)
    print("TEST PERIOD 2019-26 — lower RPS is better")
    print("=" * 78)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 78)
    print("PAIRED BOOTSTRAP vs MARKET (negative = beats the market)")
    print("=" * 78)
    market_rps = ranked_probability_score(market[is_test], outcomes[is_test])
    comparisons = []
    for name, values in candidates.items():
        if name.startswith("Market ("):
            continue
        difference = paired_bootstrap_difference(
            ranked_probability_score(values[is_test], outcomes[is_test]), market_rps
        )
        comparisons.append(
            {
                "model": name,
                "rps_difference": difference.estimate,
                "lower": difference.lower,
                "upper": difference.upper,
                "significant": difference.excludes_zero(),
            }
        )
    comparison_table = pd.DataFrame(comparisons)
    print(comparison_table.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    output = Path(REPORTS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "shots_scores.csv", index=False)
    comparison_table.to_csv(output / "shots_vs_market.csv", index=False)
    (output / "shots_pool_weights.json").write_text(
        json.dumps({name: list(map(float, w)) for name, w in fitted.items()}, indent=2)
    )
    print(f"\nWrote results to {output}")


if __name__ == "__main__":
    main()
