"""Post-hoc recalibration of three-way forecasts.

A model can rank matches well and still state the wrong numbers. Recalibration
repairs the numbers without disturbing the ranking, and it is fitted strictly
out of sample: fitting a calibrator on the same forecasts used to fit the model
manufactures a calibration that will not survive deployment.

Temperature scaling is the workhorse — a single parameter, so it cannot overfit
a validation set of realistic size, and it is monotone, so discrimination is
preserved exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from seriea.evaluation.metrics import outcomes_to_indicator

#: Search bounds for the temperature. Values above one soften an over-confident
#: forecast; values below one sharpen an under-confident one.
_TEMPERATURE_BOUNDS: tuple[float, float] = (0.2, 5.0)

#: Floor applied before taking logarithms of forecast probabilities.
_PROBABILITY_FLOOR: float = 1e-12


class TemperatureScaling:
    """Single-parameter recalibration in log-probability space.

    Rescales the forecast log-probabilities by ``1 / temperature`` and
    renormalises. A temperature above one pulls forecasts toward uniform,
    correcting over-confidence; below one pushes them toward the extremes.
    """

    def __init__(self) -> None:
        self._temperature: float | None = None

    def fit(self, forecasts: np.ndarray, outcomes: np.ndarray) -> "TemperatureScaling":
        """Choose the temperature minimising log loss on held-out forecasts.

        Args:
            forecasts: Out-of-sample forecasts, shape ``(n, 3)``.
            outcomes: Realised outcome labels, shape ``(n,)``.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: If the inputs are empty or misaligned.
        """
        probabilities = np.asarray(forecasts, dtype=float)
        if probabilities.ndim != 2:
            raise ValueError(f"Forecasts must be two-dimensional, got {probabilities.shape}.")
        if probabilities.shape[0] == 0:
            raise ValueError("Cannot fit a calibrator on zero forecasts.")

        indicator = outcomes_to_indicator(outcomes)
        if indicator.shape != probabilities.shape:
            raise ValueError(
                f"Forecasts and outcomes must align: got {probabilities.shape} "
                f"and {indicator.shape}."
            )

        log_probabilities = np.log(np.clip(probabilities, _PROBABILITY_FLOOR, None))

        def negative_log_likelihood(temperature: float) -> float:
            scaled = softmax(log_probabilities / temperature, axis=1)
            realised = (scaled * indicator).sum(axis=1)
            return float(-np.log(np.clip(realised, _PROBABILITY_FLOOR, None)).mean())

        result = minimize_scalar(
            negative_log_likelihood, bounds=_TEMPERATURE_BOUNDS, method="bounded"
        )
        self._temperature = float(result.x)
        return self

    def transform(self, forecasts: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature.

        Args:
            forecasts: Forecasts to recalibrate, shape ``(n, 3)``.

        Returns:
            Recalibrated forecasts of the same shape, rows summing to one.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._temperature is None:
            raise RuntimeError("TemperatureScaling must be fitted before transforming.")
        log_probabilities = np.log(np.clip(np.asarray(forecasts, dtype=float), _PROBABILITY_FLOOR, None))
        return softmax(log_probabilities / self._temperature, axis=1)

    @property
    def temperature(self) -> float:
        """The fitted temperature.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._temperature is None:
            raise RuntimeError("TemperatureScaling has not been fitted.")
        return self._temperature
