"""Rolling-origin backtesting.

The 2016 study selected models with five-fold cross-validation on time-ordered
data. That leaks: a fold containing March 2012 is used to predict January 2012,
so the model is scored partly on information it could not have had. The fix is
to walk forward — fit only on the past, predict the next window, advance, repeat
— which is how the model would actually be used.

Refitting before every single match would be exact but wasteful, so the origin
advances in windows of a configurable length. Because the Dixon-Coles time decay
already renders matches more than a couple of years old nearly weightless, the
training history can also be capped without materially changing the fit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from seriea.config import OUTCOMES

#: Days between successive refits of the model.
DEFAULT_REFIT_DAYS: int = 14

#: Days of history retained when fitting. At the default decay rate a match
#: three years old carries weight below 0.001, so this bound is immaterial to
#: the fit while making the walk-forward tractable.
DEFAULT_TRAINING_WINDOW_DAYS: int = 1_460


def rolling_origin_forecasts(
    matches: pd.DataFrame,
    make_forecaster: Callable[[], Any],
    start_date: pd.Timestamp,
    *,
    refit_days: int = DEFAULT_REFIT_DAYS,
    training_window_days: int | None = DEFAULT_TRAINING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Walk an expanding origin forward, forecasting each window out of sample.

    Args:
        matches: Canonical match frame, sorted by date.
        make_forecaster: Zero-argument factory returning a fresh, unfitted
            forecaster. A factory rather than an instance so each window is fit
            from scratch with no state carried across origins.
        start_date: First forecast date; everything earlier is history only.
        refit_days: Length in days of each forecast window.
        training_window_days: Cap on how far back training data extends. ``None``
            uses all available history.

    Returns:
        A frame carrying the original match columns for every forecast match,
        plus ``p_H``, ``p_D`` and ``p_A`` columns and the ``origin`` timestamp at
        which each forecast was made.

    Raises:
        ValueError: If no matches fall on or after ``start_date``.
    """
    ordered = matches.sort_values("date", kind="stable").reset_index(drop=True)
    horizon = ordered[ordered["date"] >= start_date]
    if horizon.empty:
        raise ValueError(f"No matches on or after {start_date}.")

    window = pd.Timedelta(days=refit_days)
    collected: list[pd.DataFrame] = []
    origin = pd.Timestamp(start_date)
    final_date = ordered["date"].max()

    while origin <= final_date:
        window_end = origin + window
        fixtures = ordered[(ordered["date"] >= origin) & (ordered["date"] < window_end)]
        if fixtures.empty:
            origin = window_end
            continue

        history = ordered[ordered["date"] < origin]
        if training_window_days is not None:
            earliest = origin - pd.Timedelta(days=training_window_days)
            history = history[history["date"] >= earliest]

        if history.empty:
            origin = window_end
            continue

        forecaster = make_forecaster()
        forecaster.fit(history, origin)
        probabilities = forecaster.predict_proba(fixtures)

        block = fixtures.copy()
        for position, outcome in enumerate(OUTCOMES):
            block[f"p_{outcome}"] = probabilities[:, position]
        block["origin"] = origin
        collected.append(block)

        origin = window_end

    if not collected:
        raise ValueError("Backtest produced no forecasts; check start_date and refit_days.")
    return pd.concat(collected, ignore_index=True)


def forecast_matrix(forecasts: pd.DataFrame) -> np.ndarray:
    """Extract the probability matrix from a backtest result.

    Args:
        forecasts: Frame carrying ``p_H``, ``p_D`` and ``p_A`` columns.

    Returns:
        Array of shape ``(n, 3)`` in H-D-A column order.
    """
    return forecasts[[f"p_{outcome}" for outcome in OUTCOMES]].to_numpy(dtype=float)


def align_on_common_matches(
    left: pd.DataFrame, right: pd.DataFrame, keys: tuple[str, ...] = ("date", "home", "away")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two forecast sets to the matches they share.

    Model comparison is only meaningful on a common sample; the market benchmark
    in particular is unavailable for some early fixtures.

    Args:
        left: First forecast frame.
        right: Second forecast frame.
        keys: Columns identifying a match.

    Returns:
        The two frames restricted to shared matches and sorted identically.

    Raises:
        ValueError: If either frame has duplicate keys. Without this check a
            duplicated key expands that side under ``.loc``, so the function
            returns frames of *different lengths* with no error — and any
            caller scoring them elementwise then either crashes on a shape
            mismatch or, worse, silently compares misaligned rows.
    """
    for name, frame in (("left", left), ("right", right)):
        duplicates = int(frame.duplicated(subset=list(keys)).sum())
        if duplicates:
            raise ValueError(
                f"{name} frame has {duplicates} duplicate rows for keys {list(keys)}; "
                "alignment would silently misalign rows."
            )

    left_keyed = left.set_index(list(keys)).sort_index()
    right_keyed = right.set_index(list(keys)).sort_index()
    shared = left_keyed.index.intersection(right_keyed.index)
    return (
        left_keyed.loc[shared].reset_index(),
        right_keyed.loc[shared].reset_index(),
    )
