"""Combining a structural model with the market price.

This is where the 2016 study's central question should have been posed. That
study fed bookmaker odds in as features and then compared its accuracy against
the market's, concluding its predictions were "comparable". The right question
is narrower and answerable: *given* the market's forecast, does the structural
model carry any information the market has not already priced?

A logarithmic opinion pool answers it directly. Combining

.. math:: p \\propto p_{\\text{market}}^{1-w} \\, p_{\\text{model}}^{w}

and fitting the single weight ``w`` out of sample gives a clean read. If the
fitted weight is indistinguishable from zero, the model adds nothing to the
market and any apparent edge was an artefact. A weight materially above zero is
evidence of genuine incremental information.

The logarithmic pool is preferred to a linear one because it is externally
Bayesian and keeps the combination in the exponential family: pooling log-odds
is the natural operation on probabilistic forecasts.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from seriea.evaluation.metrics import outcomes_to_indicator

#: Weight bounds. Zero is "the market alone", one is "the model alone".
_WEIGHT_BOUNDS: tuple[float, float] = (0.0, 1.0)

#: Floor applied before taking logarithms.
_PROBABILITY_FLOOR: float = 1e-12


def logarithmic_pool(
    market: np.ndarray, model: np.ndarray, weight: float
) -> np.ndarray:
    """Combine two forecast sets as a weighted geometric mean.

    Args:
        market: Market forecasts, shape ``(n, 3)``.
        model: Structural model forecasts, shape ``(n, 3)``.
        weight: Weight on the model, in ``[0, 1]``.

    Returns:
        Pooled forecasts of shape ``(n, 3)``, rows summing to one.

    Raises:
        ValueError: If the inputs are misaligned or the weight is out of range.
    """
    market_array = np.asarray(market, dtype=float)
    model_array = np.asarray(model, dtype=float)
    if market_array.shape != model_array.shape:
        raise ValueError(
            f"Forecasts must align: got {market_array.shape} and {model_array.shape}."
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must lie in [0, 1], got {weight}.")

    log_market = np.log(np.clip(market_array, _PROBABILITY_FLOOR, None))
    log_model = np.log(np.clip(model_array, _PROBABILITY_FLOOR, None))
    pooled = np.exp((1.0 - weight) * log_market + weight * log_model)
    return pooled / pooled.sum(axis=1, keepdims=True)


class MarketModelPool:
    """Fits the weight a structural model deserves alongside the market.

    Attributes are populated by :meth:`fit`; the fitted weight is the headline
    quantity, since it measures incremental information rather than raw skill.
    """

    def __init__(self) -> None:
        self._weight: float | None = None

    def fit(
        self, market: np.ndarray, model: np.ndarray, outcomes: np.ndarray
    ) -> "MarketModelPool":
        """Choose the pooling weight that minimises out-of-sample log loss.

        Args:
            market: Market forecasts on validation matches, shape ``(n, 3)``.
            model: Model forecasts on the same matches, shape ``(n, 3)``.
            outcomes: Realised outcome labels, shape ``(n,)``.

        Returns:
            The fitted pool.

        Raises:
            ValueError: If the inputs are empty or misaligned.
        """
        market_array = np.asarray(market, dtype=float)
        model_array = np.asarray(model, dtype=float)
        if market_array.shape[0] == 0:
            raise ValueError("Cannot fit a pool on zero matches.")

        indicator = outcomes_to_indicator(outcomes)
        if indicator.shape != market_array.shape:
            raise ValueError("Market forecasts and outcomes must align.")

        def negative_log_likelihood(weight: float) -> float:
            pooled = logarithmic_pool(market_array, model_array, weight)
            realised = (pooled * indicator).sum(axis=1)
            return float(-np.log(np.clip(realised, _PROBABILITY_FLOOR, None)).mean())

        result = minimize_scalar(
            negative_log_likelihood, bounds=_WEIGHT_BOUNDS, method="bounded"
        )
        self._weight = float(result.x)
        return self

    def transform(self, market: np.ndarray, model: np.ndarray) -> np.ndarray:
        """Pool forecasts using the fitted weight.

        Args:
            market: Market forecasts, shape ``(n, 3)``.
            model: Model forecasts, shape ``(n, 3)``.

        Returns:
            Pooled forecasts of shape ``(n, 3)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._weight is None:
            raise RuntimeError("MarketModelPool must be fitted before transforming.")
        return logarithmic_pool(market, model, self._weight)

    @property
    def weight(self) -> float:
        """Fitted weight on the structural model.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._weight is None:
            raise RuntimeError("MarketModelPool has not been fitted.")
        return self._weight
