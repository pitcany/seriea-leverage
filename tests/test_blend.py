"""Tests for pooling a structural model with the market."""

from __future__ import annotations

import numpy as np
import pytest

from seriea.models.blend import MarketModelPool, logarithmic_pool

RNG = np.random.default_rng(23)
TRUTH = np.array([0.45, 0.27, 0.28])


def sample_outcomes(size: int, probabilities: np.ndarray = TRUTH) -> np.ndarray:
    return RNG.choice(np.array(["H", "D", "A"], dtype=object), size=size, p=probabilities)


def test_zero_weight_returns_the_market_unchanged() -> None:
    market = np.array([[0.5, 0.3, 0.2]])
    model = np.array([[0.2, 0.3, 0.5]])
    assert logarithmic_pool(market, model, 0.0) == pytest.approx(market)


def test_unit_weight_returns_the_model_unchanged() -> None:
    market = np.array([[0.5, 0.3, 0.2]])
    model = np.array([[0.2, 0.3, 0.5]])
    assert logarithmic_pool(market, model, 1.0) == pytest.approx(model)


def test_pooled_forecasts_are_distributions() -> None:
    market = RNG.dirichlet(np.ones(3), size=50)
    model = RNG.dirichlet(np.ones(3), size=50)
    pooled = logarithmic_pool(market, model, 0.4)
    assert np.allclose(pooled.sum(axis=1), 1.0)


def test_pool_rejects_a_weight_outside_the_unit_interval() -> None:
    market = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match=r"weight must lie in \[0, 1\]"):
        logarithmic_pool(market, market, 1.5)


def test_pool_rejects_misaligned_forecasts() -> None:
    with pytest.raises(ValueError, match="must align"):
        logarithmic_pool(np.array([[0.5, 0.3, 0.2]]), np.array([[0.5, 0.3, 0.2]] * 2), 0.5)


def test_an_uninformative_model_earns_near_zero_weight() -> None:
    """A model that is pure noise must not be credited with any weight.

    This is the check that makes the fitted weight interpretable as evidence of
    incremental information rather than a free parameter that always helps.
    """
    size = 8000
    outcomes = sample_outcomes(size)
    market = np.tile(TRUTH, (size, 1))
    noise = RNG.dirichlet(np.ones(3), size=size)

    pool = MarketModelPool().fit(market, noise, outcomes)
    assert pool.weight < 0.1


def test_an_informative_model_earns_substantial_weight() -> None:
    size = 8000
    outcomes = sample_outcomes(size)

    # The market is deliberately mis-stated; the "model" knows the truth.
    market = np.tile(np.array([0.33, 0.34, 0.33]), (size, 1))
    informed = np.tile(TRUTH, (size, 1))

    pool = MarketModelPool().fit(market, informed, outcomes)
    assert pool.weight > 0.7


def test_pool_requires_fitting_before_transforming() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        MarketModelPool().transform(np.array([[0.5, 0.3, 0.2]]), np.array([[0.5, 0.3, 0.2]]))


def test_pool_rejects_empty_input() -> None:
    empty = np.empty((0, 3))
    with pytest.raises(ValueError, match="zero matches"):
        MarketModelPool().fit(empty, empty, np.array([], dtype=object))
