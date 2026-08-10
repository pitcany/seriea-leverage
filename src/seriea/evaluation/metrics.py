"""Scoring rules for three-way match forecasts.

The 2016 study scored itself on classification accuracy. Accuracy is a poor
instrument here for two reasons: it discards the confidence attached to each
prediction, so a 34% favourite and a 90% favourite score identically when both
come in; and it ignores the fact that H-D-A is *ordered*, so predicting a home
win when the away side wins is treated as no worse than predicting a draw.

The Ranked Probability Score fixes both. It is the standard measure in football
forecasting following Constantinou and Fenton, *Solving the problem of
inadequate scoring rules for assessing probabilistic football forecast models*,
Journal of Quantitative Analysis in Sports 8(1), 2012. Lower is better.
"""

from __future__ import annotations

import numpy as np

from seriea.config import OUTCOMES

#: Tolerance when checking that forecast rows sum to one.
_SUM_TOLERANCE: float = 1e-6

#: Clip applied before taking logarithms, so a confident miss yields a large but
#: finite penalty rather than infinity.
_LOG_CLIP: float = 1e-15


def outcomes_to_indicator(outcomes: np.ndarray) -> np.ndarray:
    """One-hot encode outcome labels in canonical H-D-A column order.

    Args:
        outcomes: Array of shape ``(n,)`` holding ``"H"``, ``"D"`` or ``"A"``.

    Returns:
        Array of shape ``(n, 3)`` with a single 1.0 per row.

    Raises:
        ValueError: If any label is not a recognised outcome.
    """
    labels = np.asarray(outcomes, dtype=object)
    unknown = set(labels.tolist()) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"Unrecognised outcome labels: {sorted(unknown)}")

    index = {label: position for position, label in enumerate(OUTCOMES)}
    indicator = np.zeros((labels.shape[0], len(OUTCOMES)), dtype=float)
    indicator[np.arange(labels.shape[0]), [index[label] for label in labels]] = 1.0
    return indicator


def _validate_forecasts(forecasts: np.ndarray, n_outcomes: int = 3) -> np.ndarray:
    """Check that forecasts form a valid probability simplex.

    Args:
        forecasts: Array of shape ``(n, n_outcomes)``.
        n_outcomes: Expected number of columns.

    Returns:
        The forecasts as a float array.

    Raises:
        ValueError: If the shape is wrong, any entry is negative or non-finite,
            or any row fails to sum to one.
    """
    array = np.asarray(forecasts, dtype=float)
    if array.ndim != 2 or array.shape[1] != n_outcomes:
        raise ValueError(f"Forecasts must have shape (n, {n_outcomes}), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("Forecasts contain non-finite values.")
    if (array < 0).any():
        raise ValueError("Forecasts contain negative probabilities.")

    deviation = np.abs(array.sum(axis=1) - 1.0)
    if (deviation > _SUM_TOLERANCE).any():
        worst = float(deviation.max())
        raise ValueError(f"Forecast rows must sum to 1; largest deviation was {worst:.3g}.")
    return array


def ranked_probability_score(forecasts: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Compute the per-match Ranked Probability Score.

    For ordered categories the RPS compares cumulative distributions:

    .. math::
        \\mathrm{RPS} = \\frac{1}{r-1} \\sum_{i=1}^{r-1}
        \\left( \\sum_{j=1}^{i} (p_j - a_j) \\right)^2

    with categories taken in the order H, D, A so that "adjacent" errors
    (predicting a home win when it was a draw) are penalised less than
    "distant" ones (predicting a home win when it was an away win).

    Args:
        forecasts: Array of shape ``(n, 3)`` in H-D-A column order.
        outcomes: Array of shape ``(n,)`` holding outcome labels.

    Returns:
        Array of shape ``(n,)`` of per-match scores in ``[0, 1]``. Lower is
        better; a perfect forecast scores 0.
    """
    probabilities = _validate_forecasts(forecasts)
    indicator = outcomes_to_indicator(outcomes)

    cumulative_error = np.cumsum(probabilities - indicator, axis=1)
    # The final cumulative difference is zero by construction, so only the first
    # r-1 terms carry information.
    return (cumulative_error[:, :-1] ** 2).sum(axis=1) / (len(OUTCOMES) - 1)


def log_loss(forecasts: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Compute the per-match negative log likelihood.

    Args:
        forecasts: Array of shape ``(n, 3)`` in H-D-A column order.
        outcomes: Array of shape ``(n,)`` holding outcome labels.

    Returns:
        Array of shape ``(n,)``. Lower is better.
    """
    probabilities = _validate_forecasts(forecasts)
    indicator = outcomes_to_indicator(outcomes)
    realised = (probabilities * indicator).sum(axis=1)
    return -np.log(np.clip(realised, _LOG_CLIP, 1.0))


def brier_score(forecasts: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Compute the per-match multi-category Brier score.

    Args:
        forecasts: Array of shape ``(n, 3)`` in H-D-A column order.
        outcomes: Array of shape ``(n,)`` holding outcome labels.

    Returns:
        Array of shape ``(n,)`` in ``[0, 2]``. Lower is better. Unlike the RPS
        this ignores the ordering of the categories.
    """
    probabilities = _validate_forecasts(forecasts)
    indicator = outcomes_to_indicator(outcomes)
    return ((probabilities - indicator) ** 2).sum(axis=1)


def accuracy(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute top-1 accuracy, reported only for comparability with prior work.

    Args:
        forecasts: Array of shape ``(n, 3)`` in H-D-A column order.
        outcomes: Array of shape ``(n,)`` holding outcome labels.

    Returns:
        Share of matches whose most likely forecast outcome occurred.
    """
    probabilities = _validate_forecasts(forecasts)
    predicted = np.asarray(OUTCOMES, dtype=object)[probabilities.argmax(axis=1)]
    return float((predicted == np.asarray(outcomes, dtype=object)).mean())


def skill_score(model_scores: np.ndarray, reference_scores: np.ndarray) -> float:
    """Express a model's mean score as an improvement over a reference.

    Args:
        model_scores: Per-match scores for the model under test.
        reference_scores: Per-match scores for the benchmark, on the same
            matches and the same scoring rule.

    Returns:
        ``1 - mean(model) / mean(reference)``. Positive means the model beats
        the reference; zero means it matches it; negative means it is worse.

    Raises:
        ValueError: If the arrays differ in length or the reference mean is
            zero.
    """
    model = np.asarray(model_scores, dtype=float)
    reference = np.asarray(reference_scores, dtype=float)
    if model.shape != reference.shape:
        raise ValueError(
            f"Score arrays must align: got {model.shape} and {reference.shape}."
        )

    reference_mean = float(reference.mean())
    if reference_mean == 0.0:
        raise ValueError("Reference mean score is zero; skill score is undefined.")
    return 1.0 - float(model.mean()) / reference_mean
