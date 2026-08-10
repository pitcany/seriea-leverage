"""Tests for the three-way scoring rules."""

from __future__ import annotations

import numpy as np
import pytest

from seriea.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss,
    outcomes_to_indicator,
    ranked_probability_score,
    skill_score,
)

UNIFORM = np.array([[1 / 3, 1 / 3, 1 / 3]])


def test_indicator_uses_hda_column_order() -> None:
    indicator = outcomes_to_indicator(np.array(["H", "D", "A"], dtype=object))
    assert np.array_equal(indicator, np.eye(3))


def test_indicator_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unrecognised outcome"):
        outcomes_to_indicator(np.array(["H", "X"], dtype=object))


def test_perfect_forecast_scores_zero() -> None:
    certain = np.array([[1.0, 0.0, 0.0]])
    outcome = np.array(["H"], dtype=object)
    assert ranked_probability_score(certain, outcome)[0] == pytest.approx(0.0)
    assert brier_score(certain, outcome)[0] == pytest.approx(0.0)
    assert log_loss(certain, outcome)[0] == pytest.approx(0.0)


def test_rps_penalises_distant_errors_more_than_adjacent_ones() -> None:
    """A confident home call should cost more against an away win than a draw.

    This ordinality is the property that distinguishes the RPS from accuracy and
    from the Brier score, both of which treat the two misses identically.
    """
    certain_home = np.array([[1.0, 0.0, 0.0]])
    against_draw = ranked_probability_score(certain_home, np.array(["D"], dtype=object))[0]
    against_away = ranked_probability_score(certain_home, np.array(["A"], dtype=object))[0]

    assert against_draw == pytest.approx(0.5)
    assert against_away == pytest.approx(1.0)
    assert against_away > against_draw

    # The Brier score, lacking any notion of order, cannot tell them apart.
    brier_draw = brier_score(certain_home, np.array(["D"], dtype=object))[0]
    brier_away = brier_score(certain_home, np.array(["A"], dtype=object))[0]
    assert brier_draw == pytest.approx(brier_away)


def test_rps_matches_hand_computation() -> None:
    forecast = np.array([[0.5, 0.3, 0.2]])
    # p - a = (-0.5, 0.3, 0.2); cumulative (-0.5, -0.2); (0.25 + 0.04) / 2.
    expected = (0.25 + 0.04) / 2
    assert ranked_probability_score(forecast, np.array(["H"], dtype=object))[0] == pytest.approx(
        expected
    )


def test_uniform_forecast_rps_is_five_eighteenths() -> None:
    score = ranked_probability_score(UNIFORM, np.array(["H"], dtype=object))[0]
    assert score == pytest.approx(5 / 18)


def test_log_loss_of_uniform_forecast_is_log_three() -> None:
    score = log_loss(UNIFORM, np.array(["D"], dtype=object))[0]
    assert score == pytest.approx(np.log(3))


def test_log_loss_is_finite_for_a_confident_miss() -> None:
    certain = np.array([[1.0, 0.0, 0.0]])
    assert np.isfinite(log_loss(certain, np.array(["A"], dtype=object))[0])


def test_accuracy_takes_the_argmax() -> None:
    forecasts = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    outcomes = np.array(["H", "H"], dtype=object)
    assert accuracy(forecasts, outcomes) == pytest.approx(0.5)


def test_forecast_validation_rejects_rows_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ranked_probability_score(np.array([[0.5, 0.3, 0.1]]), np.array(["H"], dtype=object))


def test_forecast_validation_rejects_negative_probabilities() -> None:
    with pytest.raises(ValueError, match="negative"):
        ranked_probability_score(np.array([[1.2, -0.2, 0.0]]), np.array(["H"], dtype=object))


def test_skill_score_signs() -> None:
    reference = np.array([0.2, 0.2])
    assert skill_score(np.array([0.1, 0.1]), reference) == pytest.approx(0.5)
    assert skill_score(np.array([0.2, 0.2]), reference) == pytest.approx(0.0)
    assert skill_score(np.array([0.4, 0.4]), reference) == pytest.approx(-1.0)


def test_skill_score_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        skill_score(np.array([0.1]), np.array([0.2, 0.2]))
