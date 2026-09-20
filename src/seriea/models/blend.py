"""Validation-fitted logarithmic opinion pools.

Weights describe performance within this specified family; neither zero nor
positive weight is a general test of incremental information. Calibration
changes can produce gains even without a match-specific additional signal.
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


def logarithmic_pool_many(
    forecast_sets: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Combine any number of forecast sets as a weighted geometric mean.

    Generalises :func:`logarithmic_pool` so several candidate models can be
    weighed against the market simultaneously rather than pairwise. Pairwise
    pooling can hide a model whose information only shows up once another
    model's contribution is already accounted for.

    Args:
        forecast_sets: Forecast arrays, each of shape ``(n, 3)``.
        weights: Non-negative weights, one per forecast set. They are
            normalised to sum to one.

    Returns:
        Pooled forecasts of shape ``(n, 3)``, rows summing to one.

    Raises:
        ValueError: If the inputs are empty, misaligned, or the weights are
            negative or sum to zero.
    """
    if not forecast_sets:
        raise ValueError("Need at least one forecast set to pool.")

    arrays = [np.asarray(forecasts, dtype=float) for forecasts in forecast_sets]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Forecast sets must all share a shape; got {sorted(shapes)}.")

    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != (len(arrays),):
        raise ValueError(
            f"Expected {len(arrays)} weights, got shape {weight_array.shape}."
        )
    if (weight_array < 0).any():
        raise ValueError("Pool weights must be non-negative.")

    total = weight_array.sum()
    if total <= 0:
        raise ValueError("Pool weights must sum to a positive value.")
    normalised = weight_array / total

    log_mixture = sum(
        weight * np.log(np.clip(array, _PROBABILITY_FLOOR, None))
        for weight, array in zip(normalised, arrays)
    )
    pooled = np.exp(log_mixture)
    return pooled / pooled.sum(axis=1, keepdims=True)


def fit_pool_weights(
    forecast_sets: list[np.ndarray],
    outcomes: np.ndarray,
    *,
    grid_step: float = 0.05,
) -> np.ndarray:
    """Choose pool weights minimising out-of-sample log loss.

    Searches the simplex on a coarse grid. A grid rather than a gradient method
    because the number of models is small, the surface is shallow near the
    optimum, and an exhaustive search cannot land on a spurious local minimum.

    Args:
        forecast_sets: Forecast arrays, each of shape ``(n, 3)``.
        outcomes: Realised outcome labels, shape ``(n,)``.
        grid_step: Spacing of the weight grid; must divide 1.

    Returns:
        Normalised weights, one per forecast set.

    Raises:
        ValueError: If fewer than one forecast set is given or the grid step is
            not in (0, 1].
    """
    if not forecast_sets:
        raise ValueError("Need at least one forecast set to pool.")
    if not 0 < grid_step <= 1:
        raise ValueError(f"grid_step must lie in (0, 1], got {grid_step}.")

    indicator = outcomes_to_indicator(outcomes)
    steps = int(round(1.0 / grid_step))
    n_models = len(forecast_sets)

    def loss(weights: np.ndarray) -> float:
        pooled = logarithmic_pool_many(forecast_sets, weights)
        realised = (pooled * indicator).sum(axis=1)
        return float(-np.log(np.clip(realised, _PROBABILITY_FLOOR, None)).mean())

    best_weights = np.zeros(n_models)
    best_weights[0] = 1.0
    best_loss = loss(best_weights)

    def walk(prefix: list[int], remaining: int, depth: int) -> None:
        nonlocal best_weights, best_loss
        if depth == n_models - 1:
            candidate = np.array(prefix + [remaining], dtype=float) / steps
            candidate_loss = loss(candidate)
            if candidate_loss < best_loss:
                best_loss, best_weights = candidate_loss, candidate
            return
        for allocation in range(remaining + 1):
            walk(prefix + [allocation], remaining - allocation, depth + 1)

    walk([], steps, 0)
    return best_weights


class MarketModelPool:
    """Fits the weight a structural model deserves alongside the market.

    Attributes are populated by :meth:`fit`; the fitted weight describes
    the optimum in this constrained pooling family.
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
        candidates = [0.0, float(result.x), 1.0]
        self._weight = min(candidates, key=negative_log_likelihood)
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
