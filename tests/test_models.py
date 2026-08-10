"""Tests for the forecasting models and the walk-forward harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seriea.evaluation.backtest import (
    align_on_common_matches,
    forecast_matrix,
    rolling_origin_forecasts,
)
from seriea.evaluation.inference import bootstrap_mean, paired_bootstrap_difference
from seriea.models.base import past_matches
from seriea.models.dixon_coles import DixonColesForecaster
from seriea.models.market import MarketForecaster, has_market_price, market_probabilities
from seriea.models.naive import BaseRateForecaster, UniformForecaster

RNG = np.random.default_rng(101)


def synthetic_matches(n_rounds: int = 60) -> pd.DataFrame:
    """Build a league where one club is clearly the strongest.

    Fiorentina score at a high rate and concede little; Verona are the reverse.
    A working model must recover that ordering.
    """
    teams = ["Fiorentina", "Juventus", "Napoli", "Verona"]
    strength = {"Fiorentina": 1.9, "Juventus": 1.4, "Napoli": 1.2, "Verona": 0.7}

    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                home_goals = int(RNG.poisson(strength[home] * 1.15))
                away_goals = int(RNG.poisson(strength[away] * 0.9))
                rows.append(
                    {
                        "date": date,
                        "home": home,
                        "away": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "outcome": (
                            "H" if home_goals > away_goals
                            else "A" if home_goals < away_goals else "D"
                        ),
                        "b365_h": 2.0, "b365_d": 3.4, "b365_a": 3.8,
                        "ps_h": np.nan, "ps_d": np.nan, "ps_a": np.nan,
                        "psc_h": np.nan, "psc_d": np.nan, "psc_a": np.nan,
                    }
                )
            date += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- baselines


def test_base_rate_forecaster_recovers_training_frequencies() -> None:
    matches = synthetic_matches(10)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = BaseRateForecaster().fit(matches, cut)

    probabilities = forecaster.predict_proba(matches.head(5))
    observed = matches["outcome"].value_counts(normalize=True)
    assert probabilities.shape == (5, 3)
    assert probabilities[0, 0] == pytest.approx(observed["H"], abs=1e-9)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_base_rate_forecaster_requires_history() -> None:
    matches = synthetic_matches(2)
    with pytest.raises(ValueError, match="No matches available"):
        BaseRateForecaster().fit(matches, matches["date"].min())


def test_base_rate_forecaster_must_be_fitted() -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        BaseRateForecaster().predict_proba(pd.DataFrame({"home": ["A"]}))


def test_uniform_forecaster_is_uniform() -> None:
    matches = synthetic_matches(2)
    probabilities = UniformForecaster().fit(matches, matches["date"].max()).predict_proba(matches)
    assert np.allclose(probabilities, 1 / 3)


def test_past_matches_excludes_the_cut_off_date() -> None:
    matches = synthetic_matches(3)
    cut = matches["date"].iloc[5]
    assert (past_matches(matches, cut)["date"] < cut).all()


# ---------------------------------------------------------------------- market


def test_market_probabilities_are_distributions() -> None:
    matches = synthetic_matches(2)
    probabilities = market_probabilities(matches)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert has_market_price(matches).all()


def test_market_probabilities_are_nan_without_a_book() -> None:
    matches = synthetic_matches(2)
    matches.loc[:, ["b365_h", "b365_d", "b365_a"]] = np.nan
    assert np.isnan(market_probabilities(matches)).all()
    assert not has_market_price(matches).any()


def test_market_forecaster_passes_prices_through() -> None:
    matches = synthetic_matches(2)
    forecaster = MarketForecaster().fit(matches, matches["date"].max())
    assert np.allclose(forecaster.predict_proba(matches), market_probabilities(matches))


# ----------------------------------------------------------------- Dixon-Coles


def test_dixon_coles_recovers_the_strength_ordering() -> None:
    matches = synthetic_matches()
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    ratings = forecaster.ratings()
    assert ratings.index[0] == "Fiorentina"
    assert ratings.index[-1] == "Verona"


def test_dixon_coles_probabilities_are_distributions() -> None:
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    probabilities = forecaster.predict_proba(matches.head(20))
    assert probabilities.shape == (20, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_dixon_coles_favours_the_stronger_side() -> None:
    matches = synthetic_matches()
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    fixture = pd.DataFrame([{"home": "Fiorentina", "away": "Verona"}])
    home, _, away = forecaster.predict_proba(fixture)[0]
    assert home > away


def test_dixon_coles_score_grids_are_normalised() -> None:
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    grids = forecaster.predict_score_grid(matches.head(4))
    assert np.allclose(grids.sum(axis=(1, 2)), 1.0)
    assert (grids >= 0).all()


def test_dixon_coles_pools_unseen_clubs_to_the_league_mean() -> None:
    """A newly promoted club has no history and must not raise."""
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    fixture = pd.DataFrame([{"home": "Fiorentina", "away": "Pisa"}])
    probabilities = forecaster.predict_proba(fixture)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_dixon_coles_rejects_negative_decay() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DixonColesForecaster(decay_rate=-0.01)


def test_dixon_coles_must_be_fitted_before_use() -> None:
    forecaster = DixonColesForecaster()
    with pytest.raises(RuntimeError, match="must be fitted"):
        forecaster.predict_proba(pd.DataFrame([{"home": "A", "away": "B"}]))
    with pytest.raises(RuntimeError, match="must be fitted"):
        forecaster.ratings()


def test_dixon_coles_needs_at_least_two_teams() -> None:
    matches = synthetic_matches(2)
    with pytest.raises(ValueError, match="at least two teams"):
        DixonColesForecaster().fit(matches, matches["date"].min())


# --------------------------------------------------------------------- harness


def test_rolling_origin_forecasts_never_train_on_the_future() -> None:
    """Every forecast must be made from an origin at or before its match date."""
    matches = synthetic_matches(30)
    start = matches["date"].min() + pd.Timedelta(days=60)
    forecasts = rolling_origin_forecasts(
        matches, lambda: BaseRateForecaster(), start, refit_days=21, training_window_days=None
    )

    assert len(forecasts) > 0
    assert (forecasts["origin"] <= forecasts["date"]).all()
    assert (forecasts["date"] >= start).all()
    assert np.allclose(forecast_matrix(forecasts).sum(axis=1), 1.0)


def test_rolling_origin_forecasts_rejects_an_empty_horizon() -> None:
    matches = synthetic_matches(3)
    with pytest.raises(ValueError, match="No matches on or after"):
        rolling_origin_forecasts(
            matches, BaseRateForecaster, matches["date"].max() + pd.Timedelta(days=10)
        )


def test_align_on_common_matches_intersects() -> None:
    left = pd.DataFrame(
        {"date": [1, 2, 3], "home": list("abc"), "away": list("xyz"), "value": [1, 2, 3]}
    )
    right = pd.DataFrame(
        {"date": [2, 3, 4], "home": list("bcd"), "away": list("yzw"), "value": [9, 8, 7]}
    )
    aligned_left, aligned_right = align_on_common_matches(left, right)
    assert len(aligned_left) == len(aligned_right) == 2


# ------------------------------------------------------------------- inference


def test_bootstrap_mean_brackets_the_sample_mean() -> None:
    scores = RNG.normal(0.2, 0.05, size=800)
    interval = bootstrap_mean(scores, resamples=2000)
    assert interval.lower < interval.estimate < interval.upper
    assert interval.estimate == pytest.approx(scores.mean())


def test_bootstrap_mean_is_reproducible() -> None:
    scores = RNG.normal(0.2, 0.05, size=200)
    first = bootstrap_mean(scores, resamples=500, seed=7)
    second = bootstrap_mean(scores, resamples=500, seed=7)
    assert first.lower == second.lower


def test_bootstrap_mean_rejects_empty_and_bad_confidence() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean(np.array([]))
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean(np.array([0.1, 0.2]), confidence=1.5)


def test_paired_bootstrap_detects_a_real_difference() -> None:
    reference = RNG.normal(0.22, 0.03, size=1500)
    model = reference - 0.02
    difference = paired_bootstrap_difference(model, reference, resamples=2000)
    assert difference.estimate == pytest.approx(-0.02, abs=1e-9)
    assert difference.excludes_zero()


def test_paired_bootstrap_finds_no_difference_when_there_is_none() -> None:
    reference = RNG.normal(0.22, 0.03, size=1500)
    difference = paired_bootstrap_difference(reference, reference, resamples=1000)
    assert not difference.excludes_zero()


def test_paired_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        paired_bootstrap_difference(np.array([0.1]), np.array([0.1, 0.2]))


def test_interval_formats_readably() -> None:
    interval = bootstrap_mean(np.array([0.1, 0.2, 0.3]), resamples=200)
    assert "[" in interval.format() and "]" in interval.format()
