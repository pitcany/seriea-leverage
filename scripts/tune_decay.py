"""Select the Dixon-Coles time-decay rate on a held-out validation period.

Select daily decay using walk-forward validation on 2013/14-2018/19.
The later historical evaluation period has already been inspected in revision.

Run: ``python scripts/tune_decay.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seriea.config import REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.backtest import forecast_matrix, rolling_origin_forecasts
from seriea.evaluation.metrics import log_loss, ranked_probability_score
from seriea.models.dixon_coles import DixonColesForecaster

#: Candidate decay rates per day. Zero means no decay at all; 0.0065 is the rate
#: retained from the original daily grid (not the published half-week units); the upper end is near
#: five weeks.
DECAY_GRID: tuple[float, ...] = (0.0, 0.002, 0.004, 0.0065, 0.009, 0.012, 0.018)

VALIDATION_START = pd.Timestamp("2013-08-01")
VALIDATION_END = pd.Timestamp("2019-07-01")


def main() -> None:
    """Sweep the decay grid and write the results to ``reports/decay_tuning.json``."""
    matches = load_all()
    validation = matches[matches["date"] < VALIDATION_END]

    results = []
    for decay in DECAY_GRID:
        forecasts = rolling_origin_forecasts(
            validation,
            lambda d=decay: DixonColesForecaster(decay_rate=d),
            VALIDATION_START,
        )
        probabilities = forecast_matrix(forecasts)
        outcomes = forecasts["outcome"].to_numpy(dtype=object)

        record = {
            "decay_rate": decay,
            "half_life_days": (float("inf") if decay == 0 else round(0.6931 / decay, 1)),
            "n_matches": int(len(forecasts)),
            "rps": float(ranked_probability_score(probabilities, outcomes).mean()),
            "log_loss": float(log_loss(probabilities, outcomes).mean()),
        }
        results.append(record)
        print(
            f"decay={decay:<7} half-life={record['half_life_days']:<8} "
            f"RPS={record['rps']:.5f}  logloss={record['log_loss']:.5f}",
            flush=True,
        )

    best = min(results, key=lambda row: row["rps"])
    print(f"\nBest decay rate by RPS: {best['decay_rate']} (half-life {best['half_life_days']}d)")

    output = Path(REPORTS_DIR) / "decay_tuning.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"grid": results, "best": best}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
