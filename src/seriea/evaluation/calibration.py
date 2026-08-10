"""Calibration diagnostics for three-way forecasts.

Discrimination and calibration are different properties and a model can have one
without the other. A forecaster that ranks matches perfectly but states 80% when
it means 55% will look excellent on accuracy or AUC and still be useless for any
decision that consumes the number itself — a bid, a rotation call, a stake.

Three views are provided:

* the **reliability curve**, binning forecasts and comparing stated probability
  against observed frequency;
* the **expected calibration error**, a scalar summary of that gap; and
* the **calibration slope and intercept**, from a logistic regression of the
  outcome on the forecast log-odds. Perfect calibration gives slope 1 and
  intercept 0; slope below 1 is the signature of over-confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from seriea.config import OUTCOMES
from seriea.evaluation.metrics import outcomes_to_indicator

#: Default number of equal-width probability bins for reliability curves.
DEFAULT_BINS: int = 10

#: Clip applied before taking log-odds, keeping the transform finite.
_PROBABILITY_CLIP: float = 1e-6


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned comparison of stated probability against observed frequency.

    Attributes:
        bin_centre: Mean forecast probability within each occupied bin.
        observed: Observed event frequency within each occupied bin.
        count: Number of forecasts in each occupied bin.
    """

    bin_centre: np.ndarray
    observed: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class CalibrationFit:
    """Logistic recalibration coefficients.

    Attributes:
        slope: Coefficient on the forecast log-odds. One means calibrated;
            below one means over-confident.
        intercept: Additive shift in log-odds. Zero means unbiased.
    """

    slope: float
    intercept: float


def reliability_curve(
    probabilities: np.ndarray, events: np.ndarray, bins: int = DEFAULT_BINS
) -> ReliabilityCurve:
    """Bin one-vs-rest forecasts and measure the observed frequency in each bin.

    Args:
        probabilities: Forecast probabilities for a single outcome, shape ``(n,)``.
        events: Binary indicator of whether that outcome occurred, shape ``(n,)``.
        bins: Number of equal-width bins spanning [0, 1].

    Returns:
        The curve, restricted to bins containing at least one forecast.

    Raises:
        ValueError: If the inputs differ in length or ``bins`` is below one.
    """
    forecast = np.asarray(probabilities, dtype=float)
    observed = np.asarray(events, dtype=float)
    if forecast.shape != observed.shape:
        raise ValueError(f"Inputs must align: got {forecast.shape} and {observed.shape}.")
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}.")

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(forecast, edges[1:-1], right=False), 0, bins - 1)

    centres, frequencies, counts = [], [], []
    for index in range(bins):
        mask = assignment == index
        if not mask.any():
            continue
        centres.append(float(forecast[mask].mean()))
        frequencies.append(float(observed[mask].mean()))
        counts.append(int(mask.sum()))

    return ReliabilityCurve(np.array(centres), np.array(frequencies), np.array(counts))


def expected_calibration_error(
    probabilities: np.ndarray, events: np.ndarray, bins: int = DEFAULT_BINS
) -> float:
    """Summarise miscalibration as a count-weighted mean absolute gap.

    Args:
        probabilities: Forecast probabilities for a single outcome.
        events: Binary indicator of whether that outcome occurred.
        bins: Number of bins.

    Returns:
        The weighted mean of ``|stated - observed|`` across occupied bins. Zero
        indicates perfect calibration at this resolution.
    """
    curve = reliability_curve(probabilities, events, bins)
    if curve.count.sum() == 0:
        return 0.0
    gaps = np.abs(curve.bin_centre - curve.observed)
    return float(np.average(gaps, weights=curve.count))


def multiclass_calibration_error(
    forecasts: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS
) -> dict[str, float]:
    """Report per-outcome calibration error plus a macro average.

    Aggregate calibration can look excellent while individual outcomes are badly
    off in offsetting directions, so the breakdown matters.

    Args:
        forecasts: Array of shape ``(n, 3)`` in H-D-A column order.
        outcomes: Array of shape ``(n,)`` of outcome labels.
        bins: Number of bins.

    Returns:
        Mapping from each outcome label to its calibration error, plus a
        ``"macro"`` key holding the unweighted mean across outcomes.
    """
    indicator = outcomes_to_indicator(outcomes)
    errors = {
        label: expected_calibration_error(forecasts[:, position], indicator[:, position], bins)
        for position, label in enumerate(OUTCOMES)
    }
    errors["macro"] = float(np.mean(list(errors.values())))
    return errors


def calibration_slope_intercept(
    probabilities: np.ndarray, events: np.ndarray
) -> CalibrationFit:
    """Fit a logistic recalibration of the outcome on the forecast log-odds.

    Args:
        probabilities: Forecast probabilities for a single outcome.
        events: Binary indicator of whether that outcome occurred.

    Returns:
        The fitted slope and intercept.

    Raises:
        ValueError: If the inputs differ in length or the events are constant,
            which leaves the slope unidentified.
    """
    forecast = np.asarray(probabilities, dtype=float)
    observed = np.asarray(events, dtype=float)
    if forecast.shape != observed.shape:
        raise ValueError(f"Inputs must align: got {forecast.shape} and {observed.shape}.")
    if len(np.unique(observed)) < 2:
        raise ValueError("Events are constant; the calibration slope is unidentified.")

    log_odds = logit(np.clip(forecast, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP))

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        slope, intercept = parameters
        predicted = expit(intercept + slope * log_odds)
        predicted = np.clip(predicted, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
        return float(
            -np.sum(observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted))
        )

    result = minimize(negative_log_likelihood, np.array([1.0, 0.0]), method="BFGS")
    return CalibrationFit(slope=float(result.x[0]), intercept=float(result.x[1]))
