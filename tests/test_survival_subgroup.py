"""Protect prior-only subgroup selection and exact paired forecast alignment."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "survival_subgroup_script", Path(__file__).resolve().parents[1] / "scripts/analyze_survival_subgroup.py"
)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
TEAMS = tuple(f"Team{i:02d}" for i in range(20))


def empty_matches():
    return pd.DataFrame({"season": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                         "home": pd.Series(dtype=str), "away": pd.Series(dtype=str),
                         "home_goals": pd.Series(dtype=int), "away_goals": pd.Series(dtype=int)})


def forecasts():
    records = []
    for season, as_of in (("1920", "2020-01-01"), ("2021", "2021-01-01")):
        for team in TEAMS:
            for model, error in (("Dixon-Coles", 0.1), ("League Poisson", 0.2)):
                records.append(dict(season=season, horizon="01-01", as_of=as_of, team=team,
                                    model=model, survived=1, actual_points=40,
                                    brier=error, points_absolute_error=error * 10,
                                    points_crps=error * 5, covered_90=1))
    return pd.DataFrame(records)


def test_selection_ignores_future_and_origin_date_results():
    before = audit.membership_at_origin(empty_matches(), TEAMS, "1920", "2020-01-01")
    future = pd.DataFrame([
        dict(season="1920", date=pd.Timestamp(date), home="Team00", away="Team19", home_goals=99, away_goals=0)
        for date in ("2020-01-01", "2020-05-01")
    ])
    after = audit.membership_at_origin(future, TEAMS, "1920", "2020-01-01")
    pd.testing.assert_frame_equal(before, after)
    future.loc[0, "date"] = pd.Timestamp("2019-12-31")
    prior = audit.membership_at_origin(future, TEAMS, "1920", "2020-01-01").set_index("team")
    assert prior.loc["Team00", "current_points"] == 3
    assert not prior.loc["Team00", "bottom8"]


def test_exactly_eight_and_stable_ties_use_names_not_input_order():
    left = audit.membership_at_origin(empty_matches(), TEAMS, "1920", "2020-01-01")
    right = audit.membership_at_origin(empty_matches(), tuple(reversed(TEAMS)), "1920", "2020-01-01")
    pd.testing.assert_frame_equal(left, right)
    assert left.bottom8.sum() == 8
    assert left[left.bottom8].team.tolist() == list(TEAMS[:8])
    assert left.eligibility_rank.tolist() == list(range(1, 21))


def test_points_ties_use_goal_difference_then_goals_for():
    results = pd.DataFrame([
        dict(season="1920", date=pd.Timestamp("2019-12-31"),
             home=home, away=away, home_goals=hg, away_goals=ag)
        for home, away, hg, ag in (
            ("Team00", "Team01", 1, 1), ("Team02", "Team03", 2, 2),
            ("Team04", "Team05", 1, 0), ("Team06", "Team07", 2, 0),
        )
    ])
    table = audit.membership_at_origin(results, TEAMS, "1920", "2020-01-01").set_index("team")
    assert table.loc["Team07", "eligibility_rank"] < table.loc["Team05", "eligibility_rank"]
    assert table.loc["Team00", "eligibility_rank"] < table.loc["Team02", "eligibility_rank"]
    assert table.loc["Team04", "eligibility_rank"] < table.loc["Team06", "eligibility_rank"]


@pytest.mark.parametrize("date,points", [
    ("2023-01-20", 0), ("2023-01-21", -15), ("2023-04-20", -15),
    ("2023-04-21", 0), ("2023-05-22", 0), ("2023-05-23", -10), ("2023-07-01", 0),
])
def test_known_sanction_timing_matches_existing_validation(date, points):
    teams = ("Juventus",) + TEAMS[1:]
    table = audit.membership_at_origin(empty_matches(), teams, "2223", date).set_index("team")
    assert table.loc["Juventus", "current_points"] == points
    assert table.loc["Juventus", "known_points_deduction"] == -points


def test_forecast_pairing_rejects_duplicates_missing_keys_and_mismatched_outcomes():
    frame = forecasts()
    audit.validate_forecasts(frame)
    with pytest.raises(ValueError, match="unique"):
        audit.validate_forecasts(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="matched keys"):
        audit.validate_forecasts(frame.iloc[1:])
    altered = frame.copy()
    altered.loc[0, "actual_points"] = 41
    with pytest.raises(ValueError, match="share observed outcomes"):
        audit.validate_forecasts(altered)


def test_aggregate_audit_preserves_all_clubs_and_exact_score_pairs():
    frame = forecasts().sample(frac=1, random_state=2)
    original = frame.copy(deep=True)
    membership, scores, comparisons = audit.analyze_survival_subgroup(empty_matches(), frame, resamples=100)
    assert len(membership) == 40
    assert membership.groupby(["season", "as_of"]).bottom8.sum().eq(8).all()
    assert scores[scores.subgroup == "bottom8"].n.eq(16).all()
    assert scores[scores.subgroup == "other12"].n.eq(24).all()
    assert comparisons.n_seasons.eq(2).all()
    assert np.allclose(comparisons[comparisons.metric == "brier"]["estimate"], -0.1)
    # Recover every original all-club average from the exhaustive partition.
    for model in audit.MODELS:
        group = scores[scores.model == model]
        assert np.average(group.brier, weights=group.n) == pytest.approx(frame[frame.model == model].brier.mean())
    pd.testing.assert_frame_equal(frame, original)


def test_membership_does_not_depend_on_forecast_scores_or_realised_outcomes():
    frame = forecasts()
    before, _, _ = audit.analyze_survival_subgroup(empty_matches(), frame, resamples=100)
    frame["brier"] = np.linspace(0, 1, len(frame))
    frame["actual_points"] = 100
    frame["survived"] = 0
    after, _, _ = audit.analyze_survival_subgroup(empty_matches(), frame, resamples=100)
    pd.testing.assert_frame_equal(before, after)
