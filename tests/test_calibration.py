"""Tests for calibration diagnostics and recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from seriea.evaluation.calibration import (
    calibration_slope_intercept,
    expected_calibration_error,
    multiclass_calibration_error,
    reliability_curve,
)
from seriea.evaluation.calibrators import TemperatureScaling
from seriea.evaluation.metrics import log_loss

RNG = np.random.default_rng(11)


def calibrated_sample(size: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Draw forecasts that are calibrated by construction."""
    probabilities = RNG.uniform(0.05, 0.95, size=size)
    events = RNG.binomial(1, probabilities).astype(float)
    return probabilities, events


def test_reliability_curve_reports_only_occupied_bins() -> None:
    curve = reliability_curve(np.array([0.05, 0.06]), np.array([0.0, 1.0]), bins=10)
    assert len(curve.bin_centre) == 1
    assert curve.count[0] == 2
    assert curve.observed[0] == pytest.approx(0.5)


def test_calibrated_forecasts_have_near_zero_calibration_error() -> None:
    probabilities, events = calibrated_sample()
    assert expected_calibration_error(probabilities, events) < 0.02


def test_calibrated_forecasts_have_unit_slope_and_zero_intercept() -> None:
    probabilities, events = calibrated_sample()
    fit = calibration_slope_intercept(probabilities, events)
    assert fit.slope == pytest.approx(1.0, abs=0.1)
    assert fit.intercept == pytest.approx(0.0, abs=0.1)


def test_overconfident_forecasts_have_slope_below_one() -> None:
    """Sharpening a calibrated forecast must show up as a slope under one.

    This is the diagnostic signature of over-confidence, and the reason slope is
    reported alongside any headline discrimination number.
    """
    probabilities, events = calibrated_sample()
    log_odds = np.log(probabilities / (1 - probabilities))
    overconfident = 1.0 / (1.0 + np.exp(-2.0 * log_odds))

    fit = calibration_slope_intercept(overconfident, events)
    assert fit.slope < 0.75


def test_calibration_slope_requires_variation_in_outcomes() -> None:
    with pytest.raises(ValueError, match="unidentified"):
        calibration_slope_intercept(np.array([0.4, 0.6]), np.array([1.0, 1.0]))


def test_multiclass_calibration_error_reports_each_outcome_and_a_macro() -> None:
    forecasts = np.tile(np.array([0.45, 0.27, 0.28]), (500, 1))
    outcomes = np.array(["H"] * 225 + ["D"] * 135 + ["A"] * 140, dtype=object)
    errors = multiclass_calibration_error(forecasts, outcomes)

    assert set(errors) == {"H", "D", "A", "macro"}
    assert errors["macro"] == pytest.approx(np.mean([errors["H"], errors["D"], errors["A"]]))


def test_temperature_scaling_leaves_calibrated_forecasts_alone() -> None:
    forecasts = np.tile(np.array([0.45, 0.27, 0.28]), (4000, 1))
    outcomes = RNG.choice(np.array(["H", "D", "A"], dtype=object), size=4000, p=[0.45, 0.27, 0.28])

    scaler = TemperatureScaling().fit(forecasts, outcomes)
    assert scaler.temperature == pytest.approx(1.0, abs=0.15)


def test_temperature_scaling_improves_an_overconfident_forecast() -> None:
    truth = np.array([0.45, 0.27, 0.28])
    outcomes = RNG.choice(np.array(["H", "D", "A"], dtype=object), size=6000, p=truth)

    sharpened = truth**2 / (truth**2).sum()
    forecasts = np.tile(sharpened, (6000, 1))

    scaler = TemperatureScaling().fit(forecasts, outcomes)
    recalibrated = scaler.transform(forecasts)

    assert log_loss(recalibrated, outcomes).mean() < log_loss(forecasts, outcomes).mean()
    assert scaler.temperature > 1.0


def test_temperature_scaling_output_is_a_distribution() -> None:
    forecasts = np.tile(np.array([0.5, 0.3, 0.2]), (100, 1))
    outcomes = RNG.choice(np.array(["H", "D", "A"], dtype=object), size=100)
    transformed = TemperatureScaling().fit(forecasts, outcomes).transform(forecasts)
    assert np.allclose(transformed.sum(axis=1), 1.0)


def test_temperature_scaling_requires_fitting_first() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        TemperatureScaling().transform(np.array([[0.5, 0.3, 0.2]]))
