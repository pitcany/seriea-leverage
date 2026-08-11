"""Tests for the shot-based forecaster and the n-way pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seriea.models.blend import fit_pool_weights, logarithmic_pool_many
from seriea.models.shots import ShotsForecaster

RNG = np.random.default_rng(31)
TRUTH = np.array([0.45, 0.27, 0.28])


def synthetic_matches(n_rounds: int = 40) -> pd.DataFrame:
    """A league where one club both shoots more and converts at league rate."""
    teams = ["Fiorentina", "Juventus", "Napoli", "Verona"]
    shot_rate = {"Fiorentina": 6.5, "Juventus": 5.0, "Napoli": 4.5, "Verona": 2.5}
    conversion = 0.32

    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                home_sot = int(RNG.poisson(shot_rate[home] * 1.1))
                away_sot = int(RNG.poisson(shot_rate[away] * 0.9))
                home_goals = int(RNG.binomial(home_sot, conversion))
                away_goals = int(RNG.binomial(away_sot, conversion))
                rows.append(
                    {
                        "date": date,
                        "home": home,
                        "away": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "home_sot": home_sot,
                        "away_sot": away_sot,
                        "outcome": (
                            "H" if home_goals > away_goals
                            else "A" if home_goals < away_goals else "D"
                        ),
                    }
                )
            date += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- forecaster


def test_shots_forecaster_recovers_the_shooting_order() -> None:
    matches = synthetic_matches()
    cut = matches["date"].max() + pd.Timedelta(days=1)
    ratings = ShotsForecaster(decay_rate=0.0).fit(matches, cut).shot_ratings()

    assert ratings.index[0] == "Fiorentina"
    assert ratings.index[-1] == "Verona"


def test_shots_forecaster_estimates_a_plausible_conversion() -> None:
    """The fitted conversion should land near the 0.32 used to generate goals."""
    matches = synthetic_matches()
    cut = matches["date"].max() + pd.Timedelta(days=1)
    factors = ShotsForecaster(decay_rate=0.0).fit(matches, cut).conversion_factors()

    assert factors["home"] == pytest.approx(0.32, abs=0.04)
    assert factors["away"] == pytest.approx(0.32, abs=0.04)


def test_shots_forecaster_probabilities_are_distributions() -> None:
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = ShotsForecaster(decay_rate=0.0).fit(matches, cut)

    probabilities = forecaster.predict_proba(matches.head(20))
    assert probabilities.shape == (20, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


def test_shots_forecaster_favours_the_stronger_side() -> None:
    matches = synthetic_matches()
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = ShotsForecaster(decay_rate=0.0).fit(matches, cut)

    fixture = pd.DataFrame([{"home": "Fiorentina", "away": "Verona"}])
    home, _, away = forecaster.predict_proba(fixture)[0]
    assert home > away


def test_shots_forecaster_score_grids_are_normalised() -> None:
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    grids = ShotsForecaster(decay_rate=0.0).fit(matches, cut).predict_score_grid(
        matches.head(4)
    )
    assert np.allclose(grids.sum(axis=(1, 2)), 1.0)


def test_shots_forecaster_pools_unseen_clubs() -> None:
    matches = synthetic_matches(20)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    forecaster = ShotsForecaster(decay_rate=0.0).fit(matches, cut)

    fixture = pd.DataFrame([{"home": "Fiorentina", "away": "Pisa"}])
    assert np.allclose(forecaster.predict_proba(fixture).sum(axis=1), 1.0)


def test_shots_forecaster_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ShotsForecaster(decay_rate=-0.1)
    with pytest.raises(ValueError, match="at least 1"):
        ShotsForecaster(max_goals=0)


def test_shots_forecaster_requires_shot_data() -> None:
    matches = synthetic_matches(5).assign(home_sot=np.nan, away_sot=np.nan)
    cut = matches["date"].max() + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="shots-on-target"):
        ShotsForecaster().fit(matches, cut)


def test_shots_forecaster_must_be_fitted() -> None:
    forecaster = ShotsForecaster()
    with pytest.raises(RuntimeError, match="must be fitted"):
        forecaster.predict_proba(pd.DataFrame([{"home": "A", "away": "B"}]))
    with pytest.raises(RuntimeError, match="must be fitted"):
        forecaster.conversion_factors()
    with pytest.raises(RuntimeError, match="must be fitted"):
        forecaster.shot_ratings()


# -------------------------------------------------------------------- n-way pool


def test_many_pool_with_one_member_returns_it_unchanged() -> None:
    forecasts = RNG.dirichlet(np.ones(3), size=10)
    assert logarithmic_pool_many([forecasts], np.array([1.0])) == pytest.approx(forecasts)


def test_many_pool_matches_the_pairwise_pool() -> None:
    """The n-way generalisation must agree with the two-model implementation."""
    from seriea.models.blend import logarithmic_pool

    first = RNG.dirichlet(np.ones(3), size=50)
    second = RNG.dirichlet(np.ones(3), size=50)
    pairwise = logarithmic_pool(first, second, 0.25)
    many = logarithmic_pool_many([first, second], np.array([0.75, 0.25]))
    assert many == pytest.approx(pairwise)


def test_many_pool_normalises_weights() -> None:
    first = RNG.dirichlet(np.ones(3), size=20)
    second = RNG.dirichlet(np.ones(3), size=20)
    scaled = logarithmic_pool_many([first, second], np.array([6.0, 2.0]))
    unit = logarithmic_pool_many([first, second], np.array([0.75, 0.25]))
    assert scaled == pytest.approx(unit)


def test_many_pool_rejects_bad_input() -> None:
    forecasts = RNG.dirichlet(np.ones(3), size=5)
    with pytest.raises(ValueError, match="at least one"):
        logarithmic_pool_many([], np.array([]))
    with pytest.raises(ValueError, match="share a shape"):
        logarithmic_pool_many([forecasts, forecasts[:3]], np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="Expected 2 weights"):
        logarithmic_pool_many([forecasts, forecasts], np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        logarithmic_pool_many([forecasts, forecasts], np.array([-1.0, 2.0]))
    with pytest.raises(ValueError, match="positive value"):
        logarithmic_pool_many([forecasts, forecasts], np.array([0.0, 0.0]))


def test_fit_pool_weights_ignores_a_noise_model() -> None:
    """A pure-noise member must earn near-zero weight alongside a truthful one."""
    size = 4000
    outcomes = RNG.choice(np.array(["H", "D", "A"], dtype=object), size=size, p=TRUTH)
    truthful = np.tile(TRUTH, (size, 1))
    noise = RNG.dirichlet(np.ones(3), size=size)

    weights = fit_pool_weights([truthful, noise], outcomes, grid_step=0.05)
    assert weights[0] > 0.9
    assert weights[1] < 0.1


def test_fit_pool_weights_splits_between_two_useful_models() -> None:
    size = 4000
    outcomes = RNG.choice(np.array(["H", "D", "A"], dtype=object), size=size, p=TRUTH)
    # Two differently-biased-but-informative forecasts; both should contribute.
    first = np.tile(np.array([0.55, 0.25, 0.20]), (size, 1))
    second = np.tile(np.array([0.35, 0.29, 0.36]), (size, 1))

    weights = fit_pool_weights([first, second], outcomes, grid_step=0.05)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0.05).all()


def test_fit_pool_weights_rejects_bad_input() -> None:
    forecasts = RNG.dirichlet(np.ones(3), size=5)
    outcomes = RNG.choice(np.array(["H", "D", "A"], dtype=object), size=5)
    with pytest.raises(ValueError, match="at least one"):
        fit_pool_weights([], outcomes)
    with pytest.raises(ValueError, match="grid_step"):
        fit_pool_weights([forecasts], outcomes, grid_step=0.0)
