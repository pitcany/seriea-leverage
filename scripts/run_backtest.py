"""Main experiment: walk-forward evaluation against the market benchmark.

Protocol, stated up front so nothing is chosen after seeing the test set:

* **Validation**, 2013-14 to 2018-19 — used to pick the Dixon-Coles decay rate
  (in ``tune_decay.py``), the pooling weight, and the calibration temperature.
* **Test**, 2019-20 to 2025-26 — touched once, for the numbers reported.

Every model forecasts the same matches, so comparisons use a paired bootstrap.
The market is the benchmark that matters: a structural model that cannot beat a
de-vigged closing price has not demonstrated incremental information, whatever
its accuracy looks like in isolation.

Run: ``python scripts/run_backtest.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from seriea.config import OUTCOMES, REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.backtest import forecast_matrix, rolling_origin_forecasts
from seriea.evaluation.calibration import calibration_slope_intercept, multiclass_calibration_error
from seriea.evaluation.calibrators import TemperatureScaling
from seriea.evaluation.inference import bootstrap_mean, paired_bootstrap_difference
from seriea.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss,
    outcomes_to_indicator,
    ranked_probability_score,
)
from seriea.models.blend import MarketModelPool
from seriea.models.dixon_coles import DixonColesForecaster
from seriea.models.market import market_probabilities
from seriea.models.naive import BaseRateForecaster, UniformForecaster

BACKTEST_START = pd.Timestamp("2013-08-01")
TEST_START = pd.Timestamp("2019-08-01")

#: Fallback if the tuning sweep has not been run.
FALLBACK_DECAY: float = 0.0065


def load_decay_rate() -> float:
    """Read the validated decay rate, falling back to the published value."""
    path = Path(REPORTS_DIR) / "decay_tuning.json"
    if not path.exists():
        print(f"! {path} not found; falling back to decay={FALLBACK_DECAY}")
        return FALLBACK_DECAY
    return float(json.loads(path.read_text())["best"]["decay_rate"])


def summarise(name: str, probabilities: np.ndarray, outcomes: np.ndarray) -> dict[str, object]:
    """Score one forecaster on one sample."""
    rps = ranked_probability_score(probabilities, outcomes)
    interval = bootstrap_mean(rps)
    return {
        "model": name,
        "n": int(len(outcomes)),
        "rps": float(rps.mean()),
        "rps_lower": interval.lower,
        "rps_upper": interval.upper,
        "log_loss": float(log_loss(probabilities, outcomes).mean()),
        "brier": float(brier_score(probabilities, outcomes).mean()),
        "accuracy": accuracy(probabilities, outcomes),
    }


def main() -> None:
    """Run the walk-forward comparison and write results to ``reports/``."""
    decay = load_decay_rate()
    print(f"Dixon-Coles decay rate (validated): {decay}\n")

    matches = load_all()
    forecasts = rolling_origin_forecasts(
        matches, lambda: DixonColesForecaster(decay_rate=decay), BACKTEST_START
    )

    # The market is a lookup, not a fit, so it is attached to the same rows.
    market = market_probabilities(forecasts)
    complete = np.isfinite(market).all(axis=1)
    forecasts = forecasts[complete].reset_index(drop=True)
    market = market[complete]
    model = forecast_matrix(forecasts)
    outcomes = forecasts["outcome"].to_numpy(dtype=object)

    is_test = (forecasts["date"] >= TEST_START).to_numpy()
    print(f"validation matches: {int((~is_test).sum())}   test matches: {int(is_test.sum())}\n")

    # ---- fit everything that needs fitting, on validation only ----------------
    pool = MarketModelPool().fit(market[~is_test], model[~is_test], outcomes[~is_test])
    pooled = pool.transform(market, model)

    scaler = TemperatureScaling().fit(model[~is_test], outcomes[~is_test])
    calibrated_model = scaler.transform(model)

    print(f"pooling weight on Dixon-Coles: {pool.weight:.4f}")
    print(f"temperature for Dixon-Coles:   {scaler.temperature:.4f}\n")

    # ---- baselines fitted the same walk-forward way --------------------------
    base_rate = rolling_origin_forecasts(
        matches, BaseRateForecaster, BACKTEST_START, refit_days=365
    )
    uniform = rolling_origin_forecasts(matches, UniformForecaster, BACKTEST_START, refit_days=365)
    keys = ["date", "home", "away"]
    lookup = forecasts[keys].merge(
        base_rate[keys + [f"p_{o}" for o in OUTCOMES]], on=keys, how="left"
    )
    base_rate_matrix = lookup[[f"p_{o}" for o in OUTCOMES]].to_numpy(dtype=float)
    uniform_matrix = np.full_like(base_rate_matrix, 1 / 3)
    del uniform

    candidates = {
        "Uniform (2016 benchmark)": uniform_matrix,
        "Base rate": base_rate_matrix,
        "Dixon-Coles": model,
        "Dixon-Coles (calibrated)": calibrated_model,
        "Market (Shin de-vigged)": market,
        "Market + Dixon-Coles pool": pooled,
    }

    rows = [summarise(name, values[is_test], outcomes[is_test]) for name, values in candidates.items()]
    table = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)

    print("=" * 96)
    print("TEST PERIOD 2019-20 to 2025-26 — lower RPS is better")
    print("=" * 96)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- paired comparisons against the market ------------------------------
    print("\n" + "=" * 96)
    print("PAIRED BOOTSTRAP vs MARKET (mean RPS difference; negative = beats the market)")
    print("=" * 96)
    market_rps = ranked_probability_score(market[is_test], outcomes[is_test])
    comparisons = []
    for name, values in candidates.items():
        if name.startswith("Market (") :
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

    # ---- calibration --------------------------------------------------------
    print("\n" + "=" * 96)
    print("CALIBRATION ON TEST PERIOD")
    print("=" * 96)
    indicator = outcomes_to_indicator(outcomes[is_test])
    calibration_rows = []
    for name, values in candidates.items():
        if name.startswith("Uniform"):
            continue
        errors = multiclass_calibration_error(values[is_test], outcomes[is_test])
        home_fit = calibration_slope_intercept(values[is_test][:, 0], indicator[:, 0])
        calibration_rows.append(
            {
                "model": name,
                "ECE_H": errors["H"],
                "ECE_D": errors["D"],
                "ECE_A": errors["A"],
                "ECE_macro": errors["macro"],
                "home_slope": home_fit.slope,
                "home_intercept": home_fit.intercept,
            }
        )
    calibration_table = pd.DataFrame(calibration_rows)
    print(calibration_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    output = Path(REPORTS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "backtest_scores.csv", index=False)
    comparison_table.to_csv(output / "backtest_vs_market.csv", index=False)
    calibration_table.to_csv(output / "backtest_calibration.csv", index=False)
    forecasts.assign(
        p_market_H=market[:, 0], p_market_D=market[:, 1], p_market_A=market[:, 2],
        p_pool_H=pooled[:, 0], p_pool_D=pooled[:, 1], p_pool_A=pooled[:, 2],
        is_test=is_test,
    ).to_parquet(output / "forecasts.parquet", index=False)

    (output / "backtest_settings.json").write_text(
        json.dumps(
            {
                "decay_rate": decay,
                "pool_weight": pool.weight,
                "temperature": scaler.temperature,
                "backtest_start": str(BACKTEST_START.date()),
                "test_start": str(TEST_START.date()),
                "n_validation": int((~is_test).sum()),
                "n_test": int(is_test.sum()),
            },
            indent=2,
        )
    )
    print(f"\nWrote results to {output}")


if __name__ == "__main__":
    main()
