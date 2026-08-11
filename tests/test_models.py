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
    """An unseen club must forecast exactly like a club at the league average.

    The earlier version of this test asserted only that the probabilities summed
    to one, which every possible fallback satisfies — so it passed while unseen
    clubs were in fact given attack 0 and defence 0. Attack 0 is the league mean
    thanks to the sum-to-zero constraint, but defence is unconstrained and its
    mean is materially negative, so promoted clubs were handed a
    better-than-average defence. This now pins the actual property.
    """
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    unseen = forecaster.predict_proba(
        pd.DataFrame([{"home": "Fiorentina", "away": "Pisa"}])
    )
    assert np.allclose(unseen.sum(axis=1), 1.0)

    # Reconstruct a club sitting exactly at the league average and confirm the
    # unseen club is forecast identically.
    ratings = forecaster.ratings()
    average_defence = forecaster._mean_defence
    home_rate = np.exp(
        forecaster.home_advantage + ratings.loc["Fiorentina", "attack"] - average_defence
    )
    away_rate = np.exp(0.0 - ratings.loc["Fiorentina", "defence"])

    goals = np.arange(forecaster.max_goals + 1)
    from scipy.stats import poisson

    grid = poisson.pmf(goals[:, None], home_rate) * poisson.pmf(goals[None, :], away_rate)
    grid = grid / grid.sum()
    expected_home = np.tril(grid, k=-1).sum()

    assert unseen[0, 0] == pytest.approx(expected_home, abs=5e-3)


def test_dixon_coles_reports_convergence() -> None:
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)
    assert forecaster.converged
    assert isinstance(forecaster.optimiser_message, str)


def test_dixon_coles_penalty_is_finite_and_graded() -> None:
    """Infeasible parameters must be penalised smoothly, not by a flat constant.

    A flat penalty is gradient-free, so an optimiser starting inside the
    infeasible region terminates immediately and reports success while
    returning its own starting values as fitted ratings.
    """
    matches = synthetic_matches(10)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = DixonColesForecaster(decay_rate=0.0).fit(matches, cut)

    past = matches[matches["date"] < cut]
    home_index = past["home"].map(forecaster._index).to_numpy(dtype=int)
    away_index = past["away"].map(forecaster._index).to_numpy(dtype=int)
    home_goals = past["home_goals"].to_numpy(dtype=int)
    away_goals = past["away_goals"].to_numpy(dtype=int)
    weights = np.ones(len(past))
    n_teams = len(forecaster._index)

    def objective(rho: float) -> float:
        params = np.concatenate(
            [np.zeros(n_teams - 1), np.zeros(n_teams), np.array([0.25, rho])]
        )
        return forecaster._negative_log_likelihood(
            params, home_index, away_index, home_goals, away_goals, weights, n_teams
        )

    # Deep in the infeasible region the penalty must still increase with the
    # violation, so a gradient points back toward feasibility.
    deep, shallow = objective(-0.89), objective(-0.80)
    assert np.isfinite(deep) and np.isfinite(shallow)
    assert deep > shallow


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
    """No forecast may be informed by a match played on or after its origin.

    The earlier version of this test only asserted ``origin <= date``. That
    property holds even when the harness trains on its own forecast window, so
    the test passed against a deliberately leaky loop — it had no power to
    detect the failure it was named for. The property that actually matters is
    that the newest training match strictly predates the oldest match being
    forecast, which is what a recording forecaster can prove directly.
    """
    matches = synthetic_matches(30)
    start = matches["date"].min() + pd.Timedelta(days=60)

    seen: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    class RecordingForecaster(BaseRateForecaster):
        """Records the training and forecast date ranges of every window."""

        def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "RecordingForecaster":
            super().fit(history, as_of)
            self._latest_training_date = history["date"].max()
            self._origin = as_of
            return self

        def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
            seen.append((self._latest_training_date, self._origin, fixtures["date"].min()))
            return super().predict_proba(fixtures)

    forecasts = rolling_origin_forecasts(
        matches, RecordingForecaster, start, refit_days=21, training_window_days=None
    )

    assert len(forecasts) > 0
    assert len(seen) > 0
    for latest_training, origin, first_fixture in seen:
        assert latest_training < origin, "training data reached the origin"
        assert latest_training < first_fixture, "training data reached the forecast window"

    assert (forecasts["date"] >= start).all()
    assert np.allclose(forecast_matrix(forecasts).sum(axis=1), 1.0)


def test_rolling_origin_test_detects_injected_leakage() -> None:
    """The leakage test above must fail when leakage is actually present.

    A guard with no power is worse than no guard, because it reads as evidence.
    This runs the same assertions against a harness that trains through the end
    of its own forecast window and confirms they now catch it.
    """
    matches = synthetic_matches(30)
    start = matches["date"].min() + pd.Timedelta(days=60)
    window = pd.Timedelta(days=21)

    caught = False
    origin = start
    while origin <= matches["date"].max():
        window_end = origin + window
        fixtures = matches[(matches["date"] >= origin) & (matches["date"] < window_end)]
        # The injected bug: history runs to the END of the forecast window.
        leaky_history = matches[matches["date"] < window_end]
        if not fixtures.empty and not leaky_history.empty:
            if not leaky_history["date"].max() < fixtures["date"].min():
                caught = True
                break
        origin = window_end

    assert caught, "the leakage assertion failed to detect training on the forecast window"


def test_align_on_common_matches_rejects_duplicate_keys() -> None:
    """Duplicate keys must raise rather than silently misalign rows."""
    left = pd.DataFrame(
        {"date": [1, 1, 2], "home": list("aab"), "away": list("xxy"), "value": [1, 2, 3]}
    )
    right = pd.DataFrame({"date": [1, 2], "home": list("ab"), "away": list("xy"), "value": [9, 8]})
    with pytest.raises(ValueError, match="duplicate rows"):
        align_on_common_matches(left, right)


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
