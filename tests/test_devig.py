"""Tests for margin removal from bookmaker prices."""

from __future__ import annotations

import numpy as np
import pytest

from seriea.odds.devig import (
    devig_multiplicative,
    devig_shin,
    implied_probabilities,
    overround,
)

#: A fair three-way book: reciprocals sum to exactly one.
FAIR_BOOK = np.array([[3.0, 3.0, 3.0]])

#: A realistic book with a heavy favourite and a longshot, ~5.9% margin.
SKEWED_BOOK = np.array([[1.5, 4.0, 7.0]])


def test_implied_probabilities_are_reciprocals() -> None:
    assert implied_probabilities(np.array([[2.0, 4.0]])) == pytest.approx(
        np.array([[0.5, 0.25]])
    )


def test_implied_probabilities_reject_odds_at_or_below_evens() -> None:
    with pytest.raises(ValueError, match="must exceed 1.0"):
        implied_probabilities(np.array([[1.0, 3.0, 3.0]]))


def test_overround_is_zero_for_a_fair_book() -> None:
    assert overround(FAIR_BOOK)[0] == pytest.approx(0.0, abs=1e-12)


def test_overround_is_positive_for_a_real_book() -> None:
    assert overround(SKEWED_BOOK)[0] > 0.0


def test_multiplicative_devig_normalises_rows() -> None:
    assert devig_multiplicative(SKEWED_BOOK).sum(axis=1)[0] == pytest.approx(1.0)


def test_shin_devig_normalises_rows() -> None:
    assert devig_shin(SKEWED_BOOK).sum(axis=1)[0] == pytest.approx(1.0)


def test_shin_reduces_to_the_book_when_there_is_no_margin() -> None:
    """With no margin to remove, Shin must return the quoted probabilities."""
    assert devig_shin(FAIR_BOOK)[0] == pytest.approx(np.array([1 / 3, 1 / 3, 1 / 3]), abs=1e-6)


def test_shin_shortens_longshots_relative_to_proportional_scaling() -> None:
    """Shin should strip more margin from the longshot than from the favourite.

    This is the favourite-longshot correction: proportional normalisation leaves
    longshots systematically over-priced as probabilities.
    """
    multiplicative = devig_multiplicative(SKEWED_BOOK)[0]
    shin = devig_shin(SKEWED_BOOK)[0]

    assert shin[0] > multiplicative[0]  # favourite gains
    assert shin[2] < multiplicative[2]  # longshot loses


def test_shin_returns_nan_for_incomplete_books() -> None:
    partial = np.array([[1.5, np.nan, 7.0]])
    assert np.isnan(devig_shin(partial)).all()


def test_shin_handles_a_mixture_of_complete_and_incomplete_books() -> None:
    books = np.array([[1.5, 4.0, 7.0], [np.nan, 4.0, 7.0], [2.5, 3.4, 3.0]])
    result = devig_shin(books)

    assert result[0].sum() == pytest.approx(1.0)
    assert np.isnan(result[1]).all()
    assert result[2].sum() == pytest.approx(1.0)


def test_devig_preserves_favourite_ordering() -> None:
    shin = devig_shin(SKEWED_BOOK)[0]
    assert shin[0] > shin[1] > shin[2]
