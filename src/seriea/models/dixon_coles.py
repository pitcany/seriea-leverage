"""Dixon-Coles bivariate-Poisson forecaster.

Each club carries an attack and a defence rating; the home side's goal rate is
``exp(gamma + attack_home - defence_away)`` and the away side's is
``exp(attack_away - defence_home)``. Two refinements from Dixon and Coles,
*Modelling association football scores and inefficiencies in the football
betting market*, Applied Statistics 46(2), 1997, make this work in practice:

* a dependence correction ``tau`` on the four low-scoring lines (0-0, 0-1, 1-0,
  1-1), where independent Poissons demonstrably misfit; and
* exponential time decay, so a match from three seasons ago informs the ratings
  less than one from last month.

The decay rate is a tuned hyperparameter rather than a fixed constant. This is
the specific failure of the 2016 study's ``e^{-i}`` weighting, which hard-coded
a decay so aggressive that a nominal five-match form window carried an effective
sample size of about two matches, and which normalised by the window length
instead of by the sum of weights.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from seriea.models.base import past_matches


class ConvergenceWarning(UserWarning):
    """Raised when the optimiser stops without reporting success.

    Its own class so callers can promote it to an error, count it across a
    backtest, or silence it deliberately — rather than discovering after the
    fact that some fits never converged.
    """

#: Largest goal tally represented in the score grid. Serie A sides essentially
#: never exceed this, and the residual mass is renormalised away.
_MAX_GOALS: int = 12

#: Bounds on the low-score dependence parameter.
#:
#: The genuine admissibility condition is data-dependent — tau stays positive
#: while ``rho >= -min(1/lambda, 1/mu)`` and ``rho <= min(1/(lambda*mu), 1)`` —
#: which for Serie A scoring rates is roughly [-0.24, +0.21] and moves with the
#: fitted rates. An earlier hard-coded box of [-0.15, 0.15] sat *inside* that
#: region and truncated the maximum-likelihood estimate: fitted values reach
#: -0.140 against the old lower bound, and on some windows the estimate pinned
#: to it outright. The box is now wide enough not to bind, with infeasible
#: values handled by the smooth penalty in the objective instead.
_RHO_BOUNDS: tuple[float, float] = (-0.9, 0.9)

#: Bounds on attack, defence and home-advantage parameters, in log-rate units.
#: Note these constrain the n-1 *free* attack parameters; the nth is determined
#: by the sum-to-zero constraint and is therefore not itself bounded.
_RATING_BOUNDS: tuple[float, float] = (-2.5, 2.5)

#: Floor applied to tau so the log-likelihood stays finite during the search.
_TAU_FLOOR: float = 1e-10

#: Multiplier on the squared tau violation. Large enough to make infeasible
#: parameters unattractive, finite enough to leave a usable gradient pointing
#: back toward the feasible region.
_INFEASIBILITY_WEIGHT: float = 1e6

#: Optimiser iteration cap.
_MAX_ITERATIONS: int = 500

#: Objective-evaluation cap. Set explicitly because scipy's L-BFGS-B applies a
#: separate ``maxfun`` default of 15000 that, with no analytic gradient and
#: ~80 parameters costing ~80 evaluations per iteration, binds long before
#: ``maxiter`` ever does.
_MAX_FUNCTION_EVALUATIONS: int = 200_000


def _tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rate: np.ndarray,
    away_rate: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Apply the Dixon-Coles low-score dependence correction.

    Args:
        home_goals: Observed home goals.
        away_goals: Observed away goals.
        home_rate: Fitted home scoring rate.
        away_rate: Fitted away scoring rate.
        rho: Dependence parameter.

    Returns:
        Multiplicative correction, one per match; 1.0 outside the four
        low-scoring lines.
    """
    correction = np.ones_like(home_rate, dtype=float)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    nil_one = (home_goals == 0) & (away_goals == 1)
    one_nil = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)

    correction[nil_nil] = 1.0 - home_rate[nil_nil] * away_rate[nil_nil] * rho
    correction[nil_one] = 1.0 + home_rate[nil_one] * rho
    correction[one_nil] = 1.0 + away_rate[one_nil] * rho
    correction[one_one] = 1.0 - rho
    return correction


class DixonColesForecaster:
    """Time-weighted bivariate-Poisson model of Serie A scorelines.

    Args:
        decay_rate: Exponential decay per day. A match ``d`` days before the
            cut-off receives weight ``exp(-decay_rate * d)``. Zero disables
            decay. The Dixon-Coles paper suggests roughly 0.0065 per day, a
            half-life near 107 days.
        max_goals: Largest goal tally in the prediction grid.
    """

    def __init__(self, decay_rate: float = 0.0065, max_goals: int = _MAX_GOALS) -> None:
        if decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}.")
        if max_goals < 1:
            raise ValueError(f"max_goals must be at least 1, got {max_goals}.")

        self.decay_rate = decay_rate
        self.max_goals = max_goals
        self._teams: tuple[str, ...] = ()
        self._index: dict[str, int] = {}
        self._attack: np.ndarray | None = None
        self._defence: np.ndarray | None = None
        self._home_advantage: float = 0.0
        self._rho: float = 0.0
        self._mean_defence: float = 0.0
        self._converged: bool = False
        self._optimiser_message: str = "not fitted"

    # ------------------------------------------------------------------ fitting

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "DixonColesForecaster":
        """Estimate ratings from matches before the cut-off.

        Args:
            history: Canonical match frame.
            as_of: Cut-off timestamp; only earlier matches are used.

        Returns:
            The fitted forecaster.

        Raises:
            ValueError: If fewer than two teams appear before the cut-off.
        """
        past = past_matches(history, as_of)
        self._teams = tuple(sorted(set(past["home"]) | set(past["away"])))
        if len(self._teams) < 2:
            raise ValueError(f"Need at least two teams before {as_of}; found {len(self._teams)}.")
        self._index = {team: position for position, team in enumerate(self._teams)}

        home_index = past["home"].map(self._index).to_numpy(dtype=int)
        away_index = past["away"].map(self._index).to_numpy(dtype=int)
        home_goals = past["home_goals"].to_numpy(dtype=int)
        away_goals = past["away_goals"].to_numpy(dtype=int)

        age_days = (as_of - past["date"]).dt.total_seconds().to_numpy() / 86_400.0
        weights = np.exp(-self.decay_rate * age_days)

        n_teams = len(self._teams)
        initial = np.concatenate(
            [np.zeros(n_teams - 1), np.zeros(n_teams), np.array([0.25, 0.0])]
        )
        bounds = (
            [_RATING_BOUNDS] * (n_teams - 1)
            + [_RATING_BOUNDS] * n_teams
            + [_RATING_BOUNDS, _RHO_BOUNDS]
        )

        result = minimize(
            self._negative_log_likelihood,
            initial,
            args=(home_index, away_index, home_goals, away_goals, weights, n_teams),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": _MAX_ITERATIONS, "maxfun": _MAX_FUNCTION_EVALUATIONS},
        )

        self._converged = bool(result.success)
        self._optimiser_message = str(result.message)
        if not self._converged:
            warnings.warn(
                f"Dixon-Coles fit at {as_of} did not converge: {self._optimiser_message}. "
                "Ratings and forecasts from this fit may be unreliable.",
                ConvergenceWarning,
                stacklevel=2,
            )

        attack, defence, self._home_advantage, self._rho = self._unpack(result.x, n_teams)
        self._attack = attack
        self._defence = defence

        # An unseen club must fall back to the league average, not to zero.
        # Attack is already centred by the sum-to-zero constraint, so zero is
        # its mean; defence carries no such constraint and its mean is
        # materially non-zero (around -0.13 on this corpus). Defaulting defence
        # to zero handed newly promoted clubs a better-than-average defence —
        # backwards, since promoted sides are typically the weakest in the
        # division. The mean is weighted by each club's exposure so that clubs
        # with almost no weight, whose ratings are barely identified, do not
        # drag it around.
        exposure = np.zeros(n_teams)
        np.add.at(exposure, home_index, weights)
        np.add.at(exposure, away_index, weights)
        self._mean_defence = (
            float(np.average(defence, weights=exposure)) if exposure.sum() > 0 else 0.0
        )
        return self

    @staticmethod
    def _unpack(
        parameters: np.ndarray, n_teams: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Expand the free parameter vector into named quantities.

        Attack ratings are constrained to sum to zero, which removes the
        translation invariance between attack and defence (adding a constant to
        every attack and every defence leaves all goal rates unchanged).

        Args:
            parameters: Free parameter vector.
            n_teams: Number of clubs.

        Returns:
            Tuple of attack ratings, defence ratings, home advantage and rho.
        """
        free_attack = parameters[: n_teams - 1]
        attack = np.concatenate([free_attack, [-free_attack.sum()]])
        defence = parameters[n_teams - 1 : 2 * n_teams - 1]
        return attack, defence, float(parameters[-2]), float(parameters[-1])

    def _negative_log_likelihood(
        self,
        parameters: np.ndarray,
        home_index: np.ndarray,
        away_index: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
    ) -> float:
        """Evaluate the time-weighted negative log likelihood.

        Args:
            parameters: Free parameter vector.
            home_index: Integer index of the home club per match.
            away_index: Integer index of the away club per match.
            home_goals: Observed home goals.
            away_goals: Observed away goals.
            weights: Time-decay weight per match.
            n_teams: Number of clubs.

        Returns:
            The weighted negative log likelihood, plus a smooth penalty
            proportional to the squared magnitude of any tau violation.

        Note:
            An earlier version returned a flat ``1e10`` whenever the correction
            went non-positive. That made the objective discontinuous and, worse,
            *flat* across the whole infeasible region: a starting point inside
            it produced a zero gradient, so the optimiser terminated
            immediately and reported success while returning the initial values
            as fitted ratings. A penalty that grows with the violation keeps the
            objective continuous and leaves a gradient pointing back toward
            feasibility.
        """
        attack, defence, home_advantage, rho = self._unpack(parameters, n_teams)

        home_rate = np.exp(home_advantage + attack[home_index] - defence[away_index])
        away_rate = np.exp(attack[away_index] - defence[home_index])

        correction = _tau(home_goals, away_goals, home_rate, away_rate, rho)

        log_likelihood = (
            np.log(np.clip(correction, _TAU_FLOOR, None))
            + poisson.logpmf(home_goals, home_rate)
            + poisson.logpmf(away_goals, away_rate)
        )
        negative_log_likelihood = -float(np.sum(weights * log_likelihood))

        violation = np.clip(-correction, 0.0, None)
        if violation.any():
            negative_log_likelihood += _INFEASIBILITY_WEIGHT * float(np.sum(violation**2))
        return negative_log_likelihood

    # --------------------------------------------------------------- prediction

    def _rates(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute goal rates for fixtures, pooling unseen clubs to the mean.

        A newly promoted club has no Serie A history, so its ratings default to
        the league average rather than raising an error.

        Args:
            fixtures: Frame with ``home`` and ``away`` columns.

        Returns:
            Tuple of home and away scoring rates.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._attack is None or self._defence is None:
            raise RuntimeError("DixonColesForecaster must be fitted before predicting.")

        def rating(names: pd.Series, table: np.ndarray, unseen: float) -> np.ndarray:
            positions = names.map(self._index)
            values = np.where(
                positions.isna(), unseen, table[positions.fillna(0).to_numpy(dtype=int)]
            )
            return values.astype(float)

        home_attack = rating(fixtures["home"], self._attack, 0.0)
        home_defence = rating(fixtures["home"], self._defence, self._mean_defence)
        away_attack = rating(fixtures["away"], self._attack, 0.0)
        away_defence = rating(fixtures["away"], self._defence, self._mean_defence)

        home_rate = np.exp(self._home_advantage + home_attack - away_defence)
        away_rate = np.exp(away_attack - home_defence)
        return home_rate, away_rate

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Forecast H-D-A probabilities by summing the score grid.

        Args:
            fixtures: Frame with ``home`` and ``away`` columns.

        Returns:
            Array of shape ``(len(fixtures), 3)`` in H-D-A column order.
        """
        grids = self.predict_score_grid(fixtures)
        upper = np.triu(np.ones((self.max_goals + 1, self.max_goals + 1)), k=1)
        home_win = (grids * upper.T).sum(axis=(1, 2))
        draw = np.trace(grids, axis1=1, axis2=2)
        away_win = (grids * upper).sum(axis=(1, 2))
        return np.column_stack([home_win, draw, away_win])

    def predict_score_grid(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Forecast the full joint scoreline distribution.

        Args:
            fixtures: Frame with ``home`` and ``away`` columns.

        Returns:
            Array of shape ``(n, max_goals + 1, max_goals + 1)`` where entry
            ``[i, x, y]`` is the probability of match ``i`` finishing ``x-y``.
            Each grid sums to one.
        """
        home_rate, away_rate = self._rates(fixtures)
        goals = np.arange(self.max_goals + 1)

        home_pmf = poisson.pmf(goals[None, :], home_rate[:, None])
        away_pmf = poisson.pmf(goals[None, :], away_rate[:, None])
        joint = home_pmf[:, :, None] * away_pmf[:, None, :]

        corrections = {
            (0, 0): 1.0 - home_rate * away_rate * self._rho,
            (0, 1): 1.0 + home_rate * self._rho,
            (1, 0): 1.0 + away_rate * self._rho,
            (1, 1): np.full_like(home_rate, 1.0 - self._rho),
        }
        for (x, y), values in corrections.items():
            joint[:, x, y] *= np.clip(values, _TAU_FLOOR, None)

        return joint / joint.sum(axis=(1, 2), keepdims=True)

    # ------------------------------------------------------------------ ratings

    def ratings(self) -> pd.DataFrame:
        """Return the fitted club ratings.

        Args:
            None.

        Returns:
            Frame indexed by club with ``attack`` and ``defence`` columns,
            sorted by attack descending.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._attack is None or self._defence is None:
            raise RuntimeError("DixonColesForecaster must be fitted before reading ratings.")
        return pd.DataFrame(
            {"attack": self._attack, "defence": self._defence}, index=list(self._teams)
        ).sort_values("attack", ascending=False)

    @property
    def home_advantage(self) -> float:
        """Fitted home advantage in log-goal-rate units."""
        return self._home_advantage

    @property
    def rho(self) -> float:
        """Fitted low-score dependence parameter."""
        return self._rho

    @property
    def converged(self) -> bool:
        """Whether the optimiser reported success on the last fit."""
        return self._converged

    @property
    def optimiser_message(self) -> str:
        """The optimiser's termination message from the last fit."""
        return self._optimiser_message

    def rating_exposure(self, history: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Report how much decayed evidence sits behind each club's rating.

        Time decay means a club relegated several seasons ago contributes
        almost no weight, so its rating is determined by almost nothing —
        :meth:`ratings` will still print a number, and that number is close to
        arbitrary. This exposes the weight so such ratings can be filtered or
        flagged rather than read at face value.

        Args:
            history: The same match frame passed to :meth:`fit`.
            as_of: The same cut-off passed to :meth:`fit`.

        Returns:
            Series indexed by club, holding total decayed match weight,
            sorted descending.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._index:
            raise RuntimeError("DixonColesForecaster must be fitted before reading exposure.")

        past = past_matches(history, as_of)
        age_days = (as_of - past["date"]).dt.total_seconds().to_numpy() / 86_400.0
        weights = np.exp(-self.decay_rate * age_days)

        exposure = pd.Series(0.0, index=list(self._teams))
        for column in ("home", "away"):
            contribution = pd.Series(weights, index=past.index).groupby(past[column]).sum()
            exposure = exposure.add(contribution, fill_value=0.0)
        return exposure.sort_values(ascending=False)
