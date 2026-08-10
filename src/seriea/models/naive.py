"""Naive reference forecasters.

These exist to be beaten. The 2016 study benchmarked itself against uniform
random guessing at 33%, which no serious model should be credited for clearing:
home advantage alone puts the majority class above 44% in Serie A. A model that
cannot beat :class:`BaseRateForecaster` has learned nothing at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seriea.config import OUTCOMES
from seriea.models.base import past_matches


class BaseRateForecaster:
    """Predicts the training-set H-D-A frequencies for every fixture.

    The constant forecast that minimises log loss in the absence of any
    covariate information, and therefore the correct floor for a probabilistic
    model.
    """

    def __init__(self) -> None:
        self._rates: np.ndarray | None = None

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "BaseRateForecaster":
        """Estimate outcome frequencies from matches before the cut-off.

        Args:
            history: Canonical match frame.
            as_of: Cut-off timestamp.

        Returns:
            The fitted forecaster.

        Raises:
            ValueError: If no matches precede the cut-off.
        """
        past = past_matches(history, as_of)
        if past.empty:
            raise ValueError(f"No matches available before {as_of}.")

        counts = past["outcome"].value_counts()
        rates = np.array([float(counts.get(outcome, 0)) for outcome in OUTCOMES])
        self._rates = rates / rates.sum()
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Return the fitted base rates, repeated for every fixture.

        Args:
            fixtures: Frame of fixtures; only its length is used.

        Returns:
            Array of shape ``(len(fixtures), 3)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._rates is None:
            raise RuntimeError("BaseRateForecaster must be fitted before predicting.")
        return np.tile(self._rates, (len(fixtures), 1))


class UniformForecaster:
    """Predicts one third for each outcome, ignoring all information.

    Included solely to reproduce the 2016 study's stated benchmark so that the
    two can be compared on a common scoring rule.
    """

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "UniformForecaster":
        """Accept the interface and learn nothing.

        Args:
            history: Ignored.
            as_of: Ignored.

        Returns:
            The forecaster, unchanged.
        """
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Return a uniform forecast for every fixture.

        Args:
            fixtures: Frame of fixtures; only its length is used.

        Returns:
            Array of shape ``(len(fixtures), 3)`` filled with one third.
        """
        return np.full((len(fixtures), len(OUTCOMES)), 1.0 / len(OUTCOMES))
