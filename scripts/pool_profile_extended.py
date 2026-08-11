"""Trace the pooling log-loss profile over negative as well as positive weights.

The headline profile in ``paper_supplements.py`` scans the weight over the
admissible pooling range ``[0, 1]``, where a logarithmic pool is a genuine
mixture of two opinions. That scan establishes that the constrained optimum is
at the left boundary, but not that the *unconstrained* optimum is: a referee can
reasonably ask whether the loss would keep falling for a weight below zero, in
which case the structural model would be anti-informative given the price rather
than merely uninformative.

This extends the scan to negative weights. A negative weight is not a mixture --
it tilts the market forecast *away* from the model in log-odds space -- so it is
a diagnostic, not a forecast anyone would issue. That is why the range is opened
here rather than in ``seriea.models.blend``, whose ``[0, 1]`` guard is correct
for the pooling the paper actually reports.

Run: ``python scripts/pool_profile_extended.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seriea.config import OUTCOMES, REPORTS_DIR
from seriea.evaluation.metrics import log_loss

#: Floor matching ``seriea.models.blend`` so the two agree on ``[0, 1]``.
_PROBABILITY_FLOOR: float = 1e-12


def pool(market: np.ndarray, model: np.ndarray, weight: float) -> np.ndarray:
    """Pool two forecast sets, admitting weights outside ``[0, 1]``.

    Identical arithmetic to :func:`seriea.models.blend.logarithmic_pool`, with
    the range check dropped.

    Args:
        market: Market forecasts, shape ``(n, 3)``.
        model: Structural forecasts, shape ``(n, 3)``.
        weight: Weight on the model. May be negative.

    Returns:
        Pooled forecasts of shape ``(n, 3)``, rows summing to one.
    """
    log_market = np.log(np.clip(market, _PROBABILITY_FLOOR, None))
    log_model = np.log(np.clip(model, _PROBABILITY_FLOOR, None))
    pooled = np.exp((1.0 - weight) * log_market + weight * log_model)
    return pooled / pooled.sum(axis=1, keepdims=True)


def matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Extract an ``(n, 3)`` probability matrix for a column prefix."""
    return frame[[f"{prefix}{o}" for o in OUTCOMES]].to_numpy(dtype=float)


def profile(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Mean log loss on validation and test across a weight grid."""
    is_test = frame["is_test"].to_numpy()
    market, model = matrix(frame, "p_market_"), matrix(frame, "p_")
    outcomes = frame["outcome"].to_numpy(dtype=object)

    rows = []
    for weight in weights:
        pooled = pool(market, model, float(weight))
        rows.append(
            {
                "weight": float(weight),
                "validation": float(log_loss(pooled[~is_test], outcomes[~is_test]).mean()),
                "test": float(log_loss(pooled[is_test], outcomes[is_test]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Scan a coarse grid, then refine around the validation optimum."""
    cached = Path(REPORTS_DIR) / "forecasts.parquet"
    if not cached.exists():
        raise FileNotFoundError(f"{cached} missing. Run scripts/run_backtest.py first.")
    frame = pd.read_parquet(cached)

    coarse = profile(frame, np.round(np.arange(-1.0, 1.001, 0.05), 3))
    best = coarse.loc[coarse["validation"].idxmin()]
    centre = float(best["weight"])

    fine = profile(frame, np.round(np.arange(centre - 0.05, centre + 0.0501, 0.005), 4))
    best_fine = fine.loc[fine["validation"].idxmin()]

    output = Path(REPORTS_DIR)
    coarse.to_csv(output / "pool_profile_extended.csv", index=False)
    fine.to_csv(output / "pool_profile_extended_fine.csv", index=False)

    zero = coarse.loc[coarse["weight"].abs() < 1e-9].iloc[0]
    print("=== EXTENDED POOLING PROFILE ===")
    print(f"grid: [{coarse['weight'].min():.2f}, {coarse['weight'].max():.2f}]")
    print(f"validation argmin (coarse): w = {centre:+.3f}")
    print(f"validation argmin (fine):   w = {float(best_fine['weight']):+.4f}")
    print()
    print(f"  log loss at w = 0      : validation {zero['validation']:.8f}  test {zero['test']:.8f}")
    print(
        f"  log loss at validation argmin: validation {float(best_fine['validation']):.8f}"
        f"  test {float(best_fine['test']):.8f}"
    )
    print(f"  validation improvement over w=0: {zero['validation'] - float(best_fine['validation']):.3e}")
    print()
    negative = coarse[coarse["weight"] < 0]
    print(f"  min validation loss over w < 0: {negative['validation'].min():.8f}")
    print(f"  monotone increasing on [0, 1] : {bool(np.all(np.diff(coarse[coarse['weight'] >= 0]['validation'].to_numpy()) > 0))}")
    print(f"  monotone decreasing on [-1, 0]: {bool(np.all(np.diff(negative['validation'].to_numpy()) < 0))}")


if __name__ == "__main__":
    main()
