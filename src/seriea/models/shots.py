"""Forecasting from shot volume rather than goals.

Goals are a sparse, noisy realisation of how much a side actually threatened.
A team that creates ten clear chances and converts one is, in expectation,
stronger than the scoreline records — and a rating fitted on goals inherits all
of that finishing noise. Expected goals is the standard fix; it is also
proprietary, and no free, permitted source covers Serie A back to 2007.

Shots on target are the pre-xG proxy for the same quantity, and
football-data.co.uk carries them for every match in the corpus. The model here
tests the underlying hypothesis without the licensed data:

1. fit the Dixon-Coles machinery to **shots on target** instead of goals, giving
   each club an attack and defence rating in shot-rate space;
2. convert the predicted shot rates to goal rates with a single league-wide
   conversion factor, estimated on the same training window and deliberately
   *not* per-team — team-specific finishing is the noise being removed;
3. build the scoreline distribution from those goal rates.

The composition is deliberate: :class:`~seriea.models.dixon_coles.DixonColesForecaster`
is reused unmodified by relabelling shots as goals, so both models share one
fitting path and any comparison between them isolates the target variable.

The low-score dependence correction is *not* applied here. Its parameter would
have been fitted against shot counts, where the 0-0/1-1 inflation it corrects
for does not arise; carrying it across to goal rates would import a
meaningless adjustment. This model is therefore independent-Poisson on
converted rates, which is a real difference from the goals model and is
reported as such rather than papered over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from seriea.config import OUTCOMES
from seriea.models.base import past_matches
from seriea.models.dixon_coles import DixonColesForecaster

#: Largest goal tally represented in the score grid.
_MAX_GOALS: int = 12

#: Guards against a degenerate conversion factor when a training window holds
#: too few shots to estimate one.
_MIN_SHOTS_FOR_CONVERSION: int = 100

#: Fallback conversion factor (goals per shot on target) if the window is too
#: thin. Serie A sits near this across the corpus.
_FALLBACK_CONVERSION: float = 0.32


class ShotsForecaster:
    """Dixon-Coles fitted to shots on target, converted to goal rates.

    Args:
        decay_rate: Exponential decay per day, applied both to the shot-rate
            fit and to the conversion-factor estimate.
        max_goals: Largest goal tally in the prediction grid.

    Raises:
        ValueError: If ``decay_rate`` is negative or ``max_goals`` is below one.
    """

    def __init__(self, decay_rate: float = 0.002, max_goals: int = _MAX_GOALS) -> None:
        if decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}.")
        if max_goals < 1:
            raise ValueError(f"max_goals must be at least 1, got {max_goals}.")

        self.decay_rate = decay_rate
        self.max_goals = max_goals
        self._shot_model: DixonColesForecaster | None = None
        self._ratings: pd.DataFrame | None = None
        self._home_advantage: float = 0.0
        self._home_conversion: float = _FALLBACK_CONVERSION
        self._away_conversion: float = _FALLBACK_CONVERSION

    # ------------------------------------------------------------------ fitting

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "ShotsForecaster":
        """Fit shot-rate ratings and the conversion factor on prior matches.

        Args:
            history: Canonical match frame carrying ``home_sot``/``away_sot``.
            as_of: Cut-off timestamp; only earlier matches are used.

        Returns:
            The fitted forecaster.

        Raises:
            ValueError: If no match before the cut-off records shots on target.
        """
        past = past_matches(history, as_of)
        usable = past.dropna(subset=["home_sot", "away_sot"])
        if usable.empty:
            raise ValueError(f"No matches with shots-on-target data before {as_of}.")

        # Relabel shots as goals so the Dixon-Coles fitter can be reused as-is.
        relabelled = usable.assign(
            home_goals=usable["home_sot"].astype(int),
            away_goals=usable["away_sot"].astype(int),
        )
        self._shot_model = DixonColesForecaster(
            decay_rate=self.decay_rate, max_goals=self.max_goals
        ).fit(relabelled, as_of)
        self._ratings = self._shot_model.ratings()
        self._home_advantage = self._shot_model.home_advantage

        self._home_conversion, self._away_conversion = self._estimate_conversion(usable, as_of)
        return self

    def _estimate_conversion(
        self, usable: pd.DataFrame, as_of: pd.Timestamp
    ) -> tuple[float, float]:
        """Estimate league-wide goals per shot on target, home and away.

        Home and away are estimated separately because the two differ
        systematically — chance quality is not symmetric across venue.

        Args:
            usable: Matches with complete shot data, before the cut-off.
            as_of: Cut-off timestamp, the reference for time decay.

        Returns:
            Tuple of home and away conversion factors.
        """
        age_days = (as_of - usable["date"]).dt.total_seconds().to_numpy() / 86_400.0
        weights = np.exp(-self.decay_rate * age_days)

        def conversion(goals: str, shots: str) -> float:
            shot_total = float((weights * usable[shots].to_numpy(dtype=float)).sum())
            if shot_total < _MIN_SHOTS_FOR_CONVERSION:
                return _FALLBACK_CONVERSION
            goal_total = float((weights * usable[goals].to_numpy(dtype=float)).sum())
            return goal_total / shot_total

        return (
            conversion("home_goals", "home_sot"),
            conversion("away_goals", "away_sot"),
        )

    # --------------------------------------------------------------- prediction

    def _goal_rates(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict shot rates and convert them to goal rates.

        Clubs absent from the training window fall back to league-average
        ratings, matching the goals model's handling of newly promoted sides.

        Args:
            fixtures: Frame with ``home`` and ``away`` columns.

        Returns:
            Tuple of home and away expected goal rates.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._ratings is None:
            raise RuntimeError("ShotsForecaster must be fitted before predicting.")

        attack = self._ratings["attack"]
        defence = self._ratings["defence"]

        def rating(names: pd.Series, table: pd.Series) -> np.ndarray:
            return names.map(table).fillna(0.0).to_numpy(dtype=float)

        home_shots = np.exp(
            self._home_advantage
            + rating(fixtures["home"], attack)
            - rating(fixtures["away"], defence)
        )
        away_shots = np.exp(
            rating(fixtures["away"], attack) - rating(fixtures["home"], defence)
        )
        return home_shots * self._home_conversion, away_shots * self._away_conversion

    def predict_score_grid(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Forecast the joint scoreline distribution.

        Args:
            fixtures: Frame with ``home`` and ``away`` columns.

        Returns:
            Array of shape ``(n, max_goals + 1, max_goals + 1)`` where entry
            ``[i, x, y]`` is the probability match ``i`` finishes ``x-y``. Each
            grid sums to one.
        """
        home_rate, away_rate = self._goal_rates(fixtures)
        goals = np.arange(self.max_goals + 1)

        home_pmf = poisson.pmf(goals[None, :], home_rate[:, None])
        away_pmf = poisson.pmf(goals[None, :], away_rate[:, None])
        joint = home_pmf[:, :, None] * away_pmf[:, None, :]
        return joint / joint.sum(axis=(1, 2), keepdims=True)

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Forecast H-D-A probabilities by summing the score grid.

        Args:
            fixtures: Frame with ``home`` and ``away`` columns.

        Returns:
            Array of shape ``(len(fixtures), 3)`` in H-D-A column order.
        """
        grids = self.predict_score_grid(fixtures)
        # Entry [x, y] is P(home x, away y): home wins strictly below the
        # diagonal, away wins strictly above it.
        away_mask = np.triu(np.ones((self.max_goals + 1, self.max_goals + 1)), k=1)
        home_win = (grids * away_mask.T).sum(axis=(1, 2))
        draw = np.trace(grids, axis1=1, axis2=2)
        away_win = (grids * away_mask).sum(axis=(1, 2))
        return np.column_stack([home_win, draw, away_win])

    # ------------------------------------------------------------------ readout

    def conversion_factors(self) -> dict[str, float]:
        """Return the fitted goals-per-shot-on-target factors.

        Returns:
            Mapping with ``home`` and ``away`` keys.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._shot_model is None:
            raise RuntimeError("ShotsForecaster must be fitted before reading conversion.")
        return {"home": self._home_conversion, "away": self._away_conversion}

    def shot_ratings(self) -> pd.DataFrame:
        """Return club ratings in shot-rate space.

        Returns:
            Frame indexed by club with ``attack`` and ``defence`` columns.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._ratings is None:
            raise RuntimeError("ShotsForecaster must be fitted before reading ratings.")
        return self._ratings


def outcome_columns() -> tuple[str, ...]:
    """Return the canonical H-D-A column order.

    Returns:
        The outcome labels in the order :meth:`ShotsForecaster.predict_proba`
        emits them.
    """
    return OUTCOMES
