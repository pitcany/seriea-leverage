"""Conversion of bookmaker prices into probability forecasts.

Decimal odds are a reciprocal transform of probability, and the reciprocals of a
quoted book sum to more than one: the excess is the bookmaker's margin, or
"overround". Feeding raw decimal odds to a linear model — as the 2016 study did
— asks that model to invert a nonlinearity and absorb a margin simultaneously,
and it will not.

Two margin-removal schemes are provided:

* **Multiplicative** — divide through by the book sum. Simple, but it spreads
  the margin proportionally, which is known to leave a favourite-longshot bias.
* **Shin** — solves for the implied proportion of insider trading, removing more
  margin from longshots than from favourites. Shin (1992, 1993); see Štrumbelj,
  *On determining probability forecasts from betting odds*, International
  Journal of Forecasting 30(4), 2014, which finds it the best-performing
  practical method.
"""

from __future__ import annotations

import numpy as np

#: Iterations of vectorised bisection when solving Shin's insider-trading share.
#: Sixty halvings of the unit interval resolve z far below any material effect
#: on the resulting probabilities.
_BISECTION_ITERATIONS: int = 60

#: Upper bound for the insider-trading proportion. Strictly below one because
#: the Shin expression divides by ``1 - z``.
#:
#: A valid book always has its root in (0, 1), and the root grows with the
#: margin — roughly ``margin / (k - 1)`` for a ``k``-outcome book. An earlier
#: bound of 0.35 silently mis-solved any book above roughly a 33% margin: the
#: bisection converged to the clamp, and the final renormalisation restored a
#: row sum of 1.0, so the wrong answer was undetectable downstream. Serie A 1X2
#: books never come close (observed maximum root 0.061), but two-way and
#: high-margin retail markets do.
_MAX_Z: float = 0.95


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to raw (un-normalised) implied probabilities.

    Args:
        odds: Array of decimal odds, shape ``(n, k)``. Values must exceed 1.

    Returns:
        Array of the same shape holding ``1 / odds``. Rows sum to more than one
        by the bookmaker's margin.

    Raises:
        ValueError: If any finite odds value is not greater than 1.
    """
    array = np.asarray(odds, dtype=float)
    finite = np.isfinite(array)
    if np.any(array[finite] <= 1.0):
        raise ValueError("Decimal odds must exceed 1.0.")
    return 1.0 / array


def overround(odds: np.ndarray) -> np.ndarray:
    """Compute the bookmaker's margin for each book.

    Args:
        odds: Array of decimal odds, shape ``(n, k)``.

    Returns:
        Array of shape ``(n,)`` holding ``sum(1 / odds) - 1``; a 5% margin
        returns 0.05.
    """
    return implied_probabilities(odds).sum(axis=1) - 1.0


def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    """Remove the margin by proportional normalisation.

    Args:
        odds: Array of decimal odds, shape ``(n, k)``.

    Returns:
        Array of shape ``(n, k)`` whose rows sum to one.
    """
    raw = implied_probabilities(odds)
    return raw / raw.sum(axis=1, keepdims=True)


def _shin_probabilities(raw: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate Shin's probability expression for a given insider share.

    Args:
        raw: Un-normalised implied probabilities, shape ``(n, k)``.
        z: Insider-trading proportion per row, shape ``(n, 1)``.

    Returns:
        Array of shape ``(n, k)``; rows sum to one only at the solved ``z``.
    """
    book_sum = raw.sum(axis=1, keepdims=True)
    discriminant = z**2 + 4.0 * (1.0 - z) * (raw**2) / book_sum
    return (np.sqrt(discriminant) - z) / (2.0 * (1.0 - z))


def devig_shin(odds: np.ndarray) -> np.ndarray:
    """Remove the margin using Shin's insider-trading model.

    Solves, per book, for the insider proportion ``z`` that makes the implied
    probabilities sum to one, then returns those probabilities. Rows containing
    any non-finite price yield all-NaN output.

    Args:
        odds: Array of decimal odds, shape ``(n, k)``.

    Returns:
        Array of shape ``(n, k)`` whose rows sum to one, up to floating-point
        tolerance, wherever the input was complete.

    Raises:
        ValueError: If any book's margin is so large that its insider-trading
            root lies beyond :data:`_MAX_Z`. Failing loudly matters here: the
            final renormalisation would otherwise return a plausible-looking
            row that sums to one but is quietly wrong.

    Note:
        A row is treated as incomplete, and returns NaN, only when a price is
        NaN. Infinite odds imply a probability of exactly zero and are handled
        normally. A book whose reciprocals sum to *less* than one — possible
        when aggregating best prices across bookmakers — has no solution with
        a non-negative insider share; such a row resolves at ``z = 0``, which
        makes the result identical to :func:`devig_multiplicative`.
    """
    raw = implied_probabilities(odds)
    complete = ~np.isnan(raw).any(axis=1)

    result = np.full(raw.shape, np.nan, dtype=float)
    if not complete.any():
        return result

    usable = raw[complete]
    low = np.zeros((usable.shape[0], 1))
    high = np.full((usable.shape[0], 1), _MAX_Z)

    # The row sum decreases monotonically in z, so the root is bracketed only
    # if the sum at the upper bound has already fallen to or below one.
    at_upper_bound = _shin_probabilities(usable, high).sum(axis=1)
    unbracketed = at_upper_bound > 1.0
    if unbracketed.any():
        worst = float(raw[complete][unbracketed].sum(axis=1).max() - 1.0)
        raise ValueError(
            f"{int(unbracketed.sum())} book(s) have a margin too large for Shin's "
            f"model to solve within z <= {_MAX_Z} (largest margin {worst:.1%}). "
            "Use devig_multiplicative for these, or verify the prices."
        )

    # The row sum decreases monotonically in z, so plain bisection is safe and
    # vectorises cleanly across every book at once.
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        total = _shin_probabilities(usable, mid).sum(axis=1, keepdims=True)
        too_high = total > 1.0
        low = np.where(too_high, mid, low)
        high = np.where(too_high, high, mid)

    solved = _shin_probabilities(usable, 0.5 * (low + high))
    result[complete] = solved / solved.sum(axis=1, keepdims=True)
    return result
