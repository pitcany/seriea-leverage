"""Scientific checks for component covariance, ratio sets and valid scenarios."""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from seriea.evaluation.fixture_stakes import analyze_fixture_stakes, fieller_interval, fixture_stakes


def test_shared_draw_covariance_and_scenario_match_direct_linear_contrast() -> None:
    q = np.array([0.9, 0.8, 0.5])
    counts = np.array([1000, 2000, 3000])
    variances = q * (1 - q) / counts
    row = fixture_stakes(*q, *counts)
    assert row["win_over_draw"] == pytest.approx(0.1)
    assert row["draw_over_loss"] == pytest.approx(0.3)
    assert row["component_mc_covariance"] == pytest.approx(-variances[1])
    assert row["draw_loss_minus_win_draw_mc_se"] ** 2 == pytest.approx(
        np.dot([1, 4, 1], variances)
    )
    # Derive from the W/D/L transfer itself rather than reproducing U/D algebra.
    transfer = np.array([0.05, -0.10, 0.05])
    assert row["scenario_delta_survival"] == pytest.approx(transfer @ q)
    assert row["scenario_delta_survival_mc_se"] ** 2 == pytest.approx(transfer**2 @ variances)
    assert row["scenario_delta_expected_points"] == pytest.approx(transfer @ [3, 1, 0])
    assert row["scenario_delta_survival_mc_ci_high"] - row["scenario_delta_survival"] == pytest.approx(
        NormalDist().inv_cdf(0.975) * row["scenario_delta_survival_mc_se"]
    )
    # Omitting the negative covariance understates this scenario's uncertainty.
    uncorrelated = 0.05**2 * (row["win_over_draw_mc_se"]**2 + row["draw_over_loss_mc_se"]**2)
    assert row["scenario_delta_survival_mc_se"]**2 > uncorrelated


def test_fieller_endpoints_invert_the_contrast_test_and_account_for_covariance() -> None:
    result = fieller_interval(0.30, 0.10, 0.0002, 0.0001, -0.00005)
    assert result.status == "bounded"
    low, high = result.intervals[0]
    assert low < 3 < high
    z2 = NormalDist().inv_cdf(0.975)**2
    for endpoint in (low, high):
        assert (0.30 - endpoint * 0.10)**2 == pytest.approx(
            z2 * (0.0002 + endpoint**2 * 0.0001 + 2 * endpoint * 0.00005)
        )
    independent = fieller_interval(0.30, 0.10, 0.0002, 0.0001, 0)
    assert low < independent.intervals[0][0]
    assert high > independent.intervals[0][1]


def test_fieller_retains_unbounded_and_undefined_cases() -> None:
    assert fieller_interval(0, 0, 0, 0, 0).status == "undefined"
    assert fieller_interval(1, 0, 0.01, 0, 0).status == "undefined"
    all_real = fieller_interval(0, 0, 1, 1, 0)
    assert all_real.status == "all_real"
    assert all_real.intervals == ((-math.inf, math.inf),)
    disconnected = fieller_interval(1, 0, 0.01, 0.01, 0)
    assert disconnected.status == "two_unbounded_intervals"
    assert disconnected.intervals[0][0] == -math.inf
    assert disconnected.intervals[1][1] == math.inf
    assert disconnected.intervals[0][1] < 0 < disconnected.intervals[1][0]
    assert fieller_interval(2, 1, 0, 0, 0).intervals == ((2, 2),)
    z = NormalDist().inv_cdf(0.975)
    assert fieller_interval(1, z, 1, 1, 0).status == "upper_unbounded"
    assert fieller_interval(-1, z, 1, 1, 0).status == "lower_unbounded"


def test_degenerate_equal_conditionals_do_not_produce_a_ratio_or_fake_threshold() -> None:
    row = fixture_stakes(1, 1, 1, 100, 100, 100)
    assert row["draw_loss_to_win_draw_ratio"] is None
    assert row["ratio_point_defined"] is False
    assert row["ratio_is_positive_upside_threshold"] is False
    assert row["ratio_mc_ci_status"] == "undefined"
    assert json.loads(row["ratio_mc_ci_intervals"]) == []
    assert row["scenario_delta_survival"] == 0
    negative_upside = fixture_stakes(0.7, 0.8, 0.6, 100, 100, 100)
    assert negative_upside["ratio_point_defined"] is True
    assert negative_upside["ratio_is_positive_upside_threshold"] is False


def test_tiny_but_determined_ratio_is_not_mistaken_for_zero() -> None:
    tiny = fieller_interval(2e-12, 1e-12, 0, 0, 0)
    assert tiny.status == "bounded"
    assert tiny.intervals[0] == pytest.approx((2, 2))


def test_scenario_uses_empirical_draw_mass_and_preserves_probability_simplex() -> None:
    boundary = fixture_stakes(0.9, 0.8, 0.6, 45, 10, 45)
    assert boundary["scenario_feasible"] is True
    assert boundary["scenario_p_draw"] == pytest.approx(0)
    assert sum(boundary[f"scenario_p_{r}"] for r in ("win", "draw", "loss")) == pytest.approx(1)
    infeasible = fixture_stakes(0.9, 0.8, 0.6, 45, 5, 50)
    assert infeasible["scenario_feasible"] is False
    assert infeasible["scenario_status"] == "infeasible_draw_mass"
    assert infeasible["scenario_delta_survival"] is None
    assert infeasible["scenario_delta_expected_points"] is None
    assert infeasible["win_over_draw"] == pytest.approx(0.1)


@pytest.mark.parametrize("field,value", [
    ("q_win", -0.1), ("q_draw", 1.1), ("q_loss", math.nan),
    ("q_win", math.inf), ("n_win", 0), ("n_draw", -1),
    ("n_loss", 1.2), ("n_win", True), ("n_win", math.nan),
    ("draw_to_win", -0.05), ("draw_to_loss", 1.1),
    ("confidence", 0), ("confidence", 1),
])
def test_invalid_inputs_are_rejected(field: str, value: float) -> None:
    arguments = dict(q_win=0.9, q_draw=0.8, q_loss=0.6, n_win=100, n_draw=100, n_loss=100)
    arguments[field] = value
    with pytest.raises(ValueError):
        fixture_stakes(**arguments)


@pytest.mark.parametrize("vn,vd,cov", [(-1, 1, 0), (1, -1, 0), (1, 1, 2), (0, 1, 0.001)])
def test_fieller_rejects_invalid_covariance(vn: float, vd: float, cov: float) -> None:
    with pytest.raises(ValueError):
        fieller_interval(1, 1, vn, vd, cov)


def test_frame_preserves_order_and_reports_descriptive_rank_changes() -> None:
    frame = pd.DataFrame({
        "opponent": ["A", "B"], "p_objective_if_win": [0.95, 0.9],
        "p_objective_if_draw": [0.9, 0.8], "p_objective_if_loss": [0.5, 0.6],
        "n_win": [1000, 1000], "n_draw": [1000, 1000], "n_loss": [1000, 1000],
    }, index=[4, 9])
    result = analyze_fixture_stakes(frame)
    assert result.index.tolist() == [4, 9]
    assert result["opponent"].tolist() == ["A", "B"]
    assert result["total_importance_rank"].tolist() == [1, 2]
    assert result["win_over_draw_rank"].tolist() == [2, 1]
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    with pytest.raises(ValueError, match="Missing required columns"):
        analyze_fixture_stakes(frame.drop(columns="n_win"))
    with pytest.raises(ValueError, match="At least one fixture"):
        analyze_fixture_stakes(frame.iloc[:0])
    with pytest.raises(ValueError, match="already contains derived"):
        analyze_fixture_stakes(result)
