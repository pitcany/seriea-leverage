"""Uncertainty quantification for forecast comparisons.

The 2016 study ranked five classifiers spanning 47.2% to 53.0% accuracy on 218
test matches and declared a winner. The standard error on an accuracy near one
half at that sample size is about 3.4 percentage points, so the entire spread
was under two standard errors and the ranking carried no information.

Every comparison here therefore comes with an interval. Because competing models
forecast the *same* matches, their scores are strongly correlated, and the
paired bootstrap below exploits that: it resamples matches, not models, and so
gives a far tighter — and correct — interval on the difference than treating the
two score sets as independent would.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Bootstrap resamples. Ten thousand keeps Monte Carlo error on a 95% interval
#: bound well below the interval's own width.
DEFAULT_RESAMPLES: int = 10_000

#: Seed for reproducibility. Fixed by default so reported figures are stable.
DEFAULT_SEED: int = 20260810


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
        estimate: The observed statistic.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        confidence: Nominal coverage, e.g. 0.95.
    """

    estimate: float
    lower: float
    upper: float
    confidence: float

    def excludes_zero(self) -> bool:
        """Report whether the interval lies wholly above or below zero.

        Returns:
            True if zero falls outside the interval.
        """
        return self.lower > 0.0 or self.upper < 0.0

    def format(self, digits: int = 4) -> str:
        """Render as ``estimate [lower, upper]``.

        Args:
            digits: Decimal places to show.

        Returns:
            Formatted string.
        """
        return (
            f"{self.estimate:.{digits}f} "
            f"[{self.lower:.{digits}f}, {self.upper:.{digits}f}]"
        )


def bootstrap_mean(
    scores: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Bootstrap a confidence interval for a mean score.

    Args:
        scores: Per-match scores.
        confidence: Nominal coverage.
        resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        The observed mean with its percentile interval.

    Raises:
        ValueError: If ``scores`` is empty or ``confidence`` is not in (0, 1).
    """
    values = np.asarray(scores, dtype=float)
    _check_inputs(values, confidence)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return Interval(float(values.mean()), float(lower), float(upper), confidence)


def paired_bootstrap_difference(
    model_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Bootstrap the mean score difference between two forecasters.

    Resamples matches and recomputes both models' means on the same resample, so
    the correlation induced by scoring identical fixtures is preserved.

    Args:
        model_scores: Per-match scores for the model under test.
        reference_scores: Per-match scores for the benchmark, on the same
            matches in the same order.
        confidence: Nominal coverage.
        resamples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        The mean of ``model - reference`` with its interval. For a loss such as
        the RPS, a negative estimate means the model beats the benchmark.

    Raises:
        ValueError: If the two score arrays differ in shape.
    """
    model = np.asarray(model_scores, dtype=float)
    reference = np.asarray(reference_scores, dtype=float)
    if model.shape != reference.shape:
        raise ValueError(f"Score arrays must align: got {model.shape} and {reference.shape}.")

    differences = model - reference
    _check_inputs(differences, confidence)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return Interval(float(differences.mean()), float(lower), float(upper), confidence)


def _check_inputs(values: np.ndarray, confidence: float) -> None:
    """Validate bootstrap inputs.

    Args:
        values: Score or difference array.
        confidence: Nominal coverage.

    Raises:
        ValueError: If the array is empty, holds non-finite values, or the
            confidence level is outside (0, 1).
    """
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty score array.")
    if not np.isfinite(values).all():
        raise ValueError("Score array contains non-finite values.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}.")
