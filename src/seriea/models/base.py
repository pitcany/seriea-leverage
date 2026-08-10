"""Common interface for match-outcome forecasters.

Every forecaster is fitted on matches strictly before a cut-off timestamp and
then asked for probabilities on fixtures at or after it. Enforcing the cut-off
in one place is what keeps the rolling-origin backtest free of look-ahead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    """A model that turns fixtures into H-D-A probabilities."""

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "Forecaster":
        """Fit on matches played strictly before ``as_of``.

        Args:
            history: Canonical match frame; may contain matches after ``as_of``,
                which the implementation must ignore.
            as_of: Cut-off timestamp separating history from the forecast set.

        Returns:
            The fitted forecaster.
        """
        ...

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Forecast H-D-A probabilities for the given fixtures.

        Args:
            fixtures: Frame with at least ``home`` and ``away`` columns.

        Returns:
            Array of shape ``(len(fixtures), 3)`` in H-D-A column order, whose
            rows sum to one.
        """
        ...


def past_matches(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Select the matches a forecaster is permitted to learn from.

    Args:
        history: Canonical match frame.
        as_of: Cut-off timestamp.

    Returns:
        The subset played strictly before ``as_of``.
    """
    return history[history["date"] < as_of]
