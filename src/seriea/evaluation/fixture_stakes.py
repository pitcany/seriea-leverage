"""Draw-aware conditional fixture consequences, with simulation-only uncertainty.

These are descriptive transformations of an existing joint season simulation.
The hypothetical probability transfer preserves conditional scorelines, all
other fixtures, and playoff treatment; it identifies no causal tactical effect.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from statistics import NormalDist
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FiellerInterval:
    """Confidence set for a ratio; disconnected sets retain both intervals.

    Infinite endpoints are represented by float infinities internally. An
    undefined set has no intervals and must not be interpreted as an empty set.
    """

    status: str
    intervals: tuple[tuple[float, float], ...]


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between zero and one.")
    return result


def _count(value: int, name: str) -> int:
    result = _finite(value, name)
    if isinstance(value, bool) or result <= 0 or not result.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    return int(result)


def _critical_value(confidence: float) -> float:
    confidence = _finite(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one.")
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    if not math.isfinite(critical):
        raise ValueError("confidence is too close to one for numerical evaluation.")
    return critical


def fieller_interval(
    numerator: float,
    denominator: float,
    numerator_variance: float,
    denominator_variance: float,
    covariance: float,
    *,
    confidence: float = 0.95,
) -> FiellerInterval:
    """Invert the normal contrast test for a correlated numerator/denominator.

    Solve (numerator - r * denominator)^2 <= z^2 Var(numerator - r *
    denominator) over all real r. Do not truncate to positive ratios: negative
    values or unbounded sets reveal limitations of the estimated threshold.
    """
    num = _finite(numerator, "numerator")
    den = _finite(denominator, "denominator")
    vn = _finite(numerator_variance, "numerator_variance")
    vd = _finite(denominator_variance, "denominator_variance")
    cov = _finite(covariance, "covariance")
    z2 = _critical_value(confidence) ** 2
    if vn < 0 or vd < 0:
        raise ValueError("Variances must be nonnegative.")
    covariance_bound = math.sqrt(vn * vd)
    if abs(cov) > covariance_bound and not math.isclose(
        abs(cov), covariance_bound, rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValueError("Covariance matrix must be positive semidefinite.")
    if den == 0 and vd == 0:
        return FiellerInterval("undefined", ())

    a = den * den - z2 * vd
    b = -2 * num * den + 2 * z2 * cov
    c = num * num - z2 * vn
    # Relative tolerances only: a tiny but well-determined probability contrast
    # must not be mistaken for a zero denominator merely because of its units.
    tolerance = 32 * math.ulp(1.0)
    if abs(a) <= tolerance * max(den * den, z2 * vd):
        a = 0.0
    if abs(b) <= tolerance * max(abs(2 * num * den), abs(2 * z2 * cov)):
        b = 0.0
    if a == 0:
        if b == 0:
            return (FiellerInterval("all_real", ((-math.inf, math.inf),))
                    if c <= 0 else FiellerInterval("empty", ()))
        endpoint = -c / b
        if b > 0:
            return FiellerInterval("lower_unbounded", ((-math.inf, endpoint),))
        return FiellerInterval("upper_unbounded", ((endpoint, math.inf),))

    discriminant = b * b - 4 * a * c
    if abs(discriminant) <= tolerance * max(b * b, abs(4 * a * c)):
        discriminant = 0.0
    if discriminant < 0:
        return (FiellerInterval("all_real", ((-math.inf, math.inf),))
                if a < 0 else FiellerInterval("empty", ()))
    if discriminant == 0:
        root = -b / (2 * a)
        return (FiellerInterval("bounded", ((root, root),))
                if a > 0 else FiellerInterval("all_real", ((-math.inf, math.inf),)))

    # Stable quadratic solution avoids subtracting almost equal numbers.
    q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
    low, high = sorted((q / a, c / q))
    if a > 0:
        return FiellerInterval("bounded", ((low, high),))
    return FiellerInterval("two_unbounded_intervals", ((-math.inf, low), (high, math.inf)))


def _normal_summary(name: str, estimate: float, variance: float, z: float) -> dict[str, float]:
    se = math.sqrt(max(0.0, variance))
    return {
        name: estimate,
        f"{name}_mc_se": se,
        f"{name}_mc_ci_low": estimate - z * se,
        f"{name}_mc_ci_high": estimate + z * se,
    }


def fixture_stakes(
    q_win: float,
    q_draw: float,
    q_loss: float,
    n_win: int,
    n_draw: int,
    n_loss: int,
    *,
    draw_to_win: float = 0.05,
    draw_to_loss: float = 0.05,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize one fixture, conditional on observed simulation category counts.

    For U=q_win-q_draw and D=q_draw-q_loss, Cov(U,D)=-Var(q_draw).
    The transfer (a, -(a+b), b) has survival change aU-bD and expected-points
    change 2a-b. Feasibility uses empirical n_draw/N, not an external estimate.
    Infeasible transfers retain component estimates but suppress scenario effects.
    """
    qw, qd, ql = (_probability(q, name) for q, name in
                  ((q_win, "q_win"), (q_draw, "q_draw"), (q_loss, "q_loss")))
    nw, nd, nl = (_count(n, name) for n, name in
                  ((n_win, "n_win"), (n_draw, "n_draw"), (n_loss, "n_loss")))
    a = _probability(draw_to_win, "draw_to_win")
    b = _probability(draw_to_loss, "draw_to_loss")
    z = _critical_value(confidence)
    vw, vd, vl = qw * (1 - qw) / nw, qd * (1 - qd) / nd, ql * (1 - ql) / nl
    u, d = qw - qd, qd - ql
    vu, vdown, cov = vw + vd, vd + vl, -vd
    result: dict[str, Any] = {}
    result.update(_normal_summary("win_over_draw", u, vu, z))
    result.update(_normal_summary("draw_over_loss", d, vdown, z))
    result["component_mc_covariance"] = cov
    result.update(_normal_summary("draw_loss_minus_win_draw", d - u, vu + vdown - 2 * cov, z))
    ratio_set = fieller_interval(d, u, vdown, vu, cov, confidence=confidence)
    result["draw_loss_to_win_draw_ratio"] = d / u if u != 0 else None
    result["ratio_point_defined"] = u != 0
    result["ratio_is_positive_upside_threshold"] = u > 0
    result["ratio_mc_ci_status"] = ratio_set.status
    single_interval = ratio_set.intervals[0] if len(ratio_set.intervals) == 1 else (math.nan, math.nan)
    result["ratio_mc_ci_low"] = single_interval[0] if math.isfinite(single_interval[0]) else None
    result["ratio_mc_ci_high"] = single_interval[1] if math.isfinite(single_interval[1]) else None
    result["ratio_mc_ci_intervals"] = json.dumps([
        [endpoint if math.isfinite(endpoint) else None for endpoint in interval]
        for interval in ratio_set.intervals
    ], separators=(",", ":"), allow_nan=False)

    total = nw + nd + nl
    pw, pd_, pl = nw / total, nd / total, nl / total
    feasible = a + b <= pd_ or math.isclose(a + b, pd_, rel_tol=0.0, abs_tol=1e-14)
    result.update({
        "empirical_p_win": pw, "empirical_p_draw": pd_, "empirical_p_loss": pl,
        "scenario_draw_to_win": a, "scenario_draw_to_loss": b,
        "scenario_feasible": feasible,
        "scenario_status": "feasible" if feasible else "infeasible_draw_mass",
        "scenario_p_win": pw + a if feasible else None,
        "scenario_p_draw": max(0.0, pd_ - a - b) if feasible else None,
        "scenario_p_loss": pl + b if feasible else None,
        "scenario_delta_expected_points": 2 * a - b if feasible else None,
    })
    if feasible:
        scenario_variance = a * a * vu + b * b * vdown - 2 * a * b * cov
        result.update(_normal_summary("scenario_delta_survival", a * u - b * d, scenario_variance, z))
    else:
        result.update({key: None for key in (
            "scenario_delta_survival", "scenario_delta_survival_mc_se",
            "scenario_delta_survival_mc_ci_low", "scenario_delta_survival_mc_ci_high",
        )})
    return result


def analyze_fixture_stakes(
    frame: pd.DataFrame, *, draw_to_win: float = 0.05,
    draw_to_loss: float = 0.05, confidence: float = 0.95,
) -> pd.DataFrame:
    """Preserve input order and columns, adding consequences and descriptive ranks."""
    required = ["p_objective_if_win", "p_objective_if_draw", "p_objective_if_loss",
                "n_win", "n_draw", "n_loss"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("At least one fixture is required.")
    records = [fixture_stakes(*[row[column] for column in required],
                             draw_to_win=draw_to_win, draw_to_loss=draw_to_loss,
                             confidence=confidence) for _, row in frame.iterrows()]
    derived = pd.DataFrame(records, index=frame.index)
    collisions = sorted(set(frame.columns) & set(derived.columns))
    if collisions:
        raise ValueError(f"Input already contains derived fields: {', '.join(collisions)}.")
    result = pd.concat([frame.copy(), derived], axis=1)
    total = result["win_over_draw"] + result["draw_over_loss"]
    result["total_importance_rank"] = total.rank(method="min", ascending=False).astype(int)
    for name in ("win_over_draw", "draw_over_loss"):
        result[f"{name}_rank"] = result[name].rank(method="min", ascending=False).astype(int)
    return result
