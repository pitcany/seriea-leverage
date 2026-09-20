"""Exploratory bottom-eight-at-origin audit of existing survival forecasts.

Membership uses only prior results and sanctions already represented in the
validation script. No models are fitted and no new seasons are simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from seriea.config import PROJECT_ROOT, RAW_DIR, REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.inference import DEFAULT_SEED, cluster_bootstrap_mean
from seriea.simulation.season import current_standings

_validation_spec = importlib.util.spec_from_file_location(
    "survival_subgroup_validation", Path(__file__).with_name("validate_seasons.py")
)
_validation = importlib.util.module_from_spec(_validation_spec)
_validation_spec.loader.exec_module(_validation)
deduction_at = _validation.deduction_at

MODELS = ("Dixon-Coles", "League Poisson")
KEYS = ["season", "horizon", "as_of", "team"]
METRICS = ["brier", "points_absolute_error", "points_crps", "covered_90"]


def validate_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Require exact, unique model pairs at every club-season-origin key."""
    required = KEYS + ["model", "survived", "actual_points"] + METRICS
    missing = sorted(set(required) - set(forecasts.columns))
    if missing:
        raise ValueError(f"Missing forecast columns: {', '.join(missing)}.")
    frame = forecasts.copy()
    frame["season"] = frame["season"].astype(str)
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="raise").dt.strftime("%Y-%m-%d")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Forecast keys and score fields must be nonempty and nonmissing.")
    if frame.duplicated(KEYS + ["model"]).any():
        raise ValueError("Forecast keys must be unique within each model.")
    if set(frame["model"]) != set(MODELS):
        raise ValueError("Forecasts must contain exactly Dixon-Coles and League Poisson.")
    if not np.isfinite(frame[METRICS + ["survived", "actual_points"]].to_numpy(dtype=float)).all():
        raise ValueError("Forecast scores and outcomes must be finite.")
    left = frame[frame.model == MODELS[0]].set_index(KEYS).sort_index()
    right = frame[frame.model == MODELS[1]].set_index(KEYS).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Model forecasts must have exactly matched keys.")
    if not np.array_equal(left[["survived", "actual_points"]], right[["survived", "actual_points"]]):
        raise ValueError("Matched model forecasts must share observed outcomes.")
    for (season, horizon), group in left.reset_index().groupby(["season", "horizon"]):
        if group["as_of"].nunique() != 1 or group["team"].nunique() != 20 or len(group) != 20:
            raise ValueError(f"Expected 20 clubs and one origin for {season} {horizon}.")
        if group["as_of"].iloc[0][5:] != horizon:
            raise ValueError("Horizon must match the forecast-origin month and day.")
    return frame


def membership_at_origin(
    matches: pd.DataFrame, teams: tuple[str, ...], season: str, as_of: str,
) -> pd.DataFrame:
    """Select exactly eight lowest clubs; eligibility rank 1 means lowest.

    Sort ascending by points, goal difference, goals for and team name. This is
    a deterministic eligibility convention, not an official final-table ranking.
    """
    if len(teams) != 20 or len(set(teams)) != 20:
        raise ValueError("Membership requires 20 distinct clubs.")
    origin = pd.Timestamp(as_of)
    played = matches[(matches["season"].astype(str) == str(season)) & (matches["date"] < origin)]
    if not (set(played["home"]) | set(played["away"])).issubset(set(teams)):
        raise ValueError("Prior results contain clubs absent from the forecast roster.")
    table = current_standings(played, teams).rename_axis("team").reset_index()
    table["current_points_before_deduction"] = table["points"]
    table["known_points_deduction"] = 0
    if str(season) == "2223" and "Juventus" in teams:
        table.loc[table.team == "Juventus", "known_points_deduction"] = deduction_at(origin)
    table["points"] -= table["known_points_deduction"]
    table = table.sort_values(["points", "goal_difference", "goals_for", "team"], kind="stable")
    table["eligibility_rank"] = np.arange(1, 21)
    table["bottom8"] = table["eligibility_rank"] <= 8
    table["subgroup"] = np.where(table["bottom8"], "bottom8", "other12")
    table = table.rename(columns={"points": "current_points", "goal_difference": "current_goal_difference",
                                  "goals_for": "current_goals_for"})
    table["season"], table["horizon"], table["as_of"] = str(season), origin.strftime("%m-%d"), origin.strftime("%Y-%m-%d")
    return table[KEYS + ["current_points_before_deduction", "known_points_deduction", "current_points",
                         "current_goal_difference", "current_goals_for", "eligibility_rank", "bottom8", "subgroup"]].reset_index(drop=True)


def analyze_survival_subgroup(
    matches: pd.DataFrame, forecasts: pd.DataFrame, *, resamples: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all-club membership, subgroup scores and paired season intervals."""
    frame = validate_forecasts(forecasts)
    membership = pd.concat([
        membership_at_origin(matches, tuple(sorted(group.team.unique())), season, as_of)
        for (season, _, as_of), group in frame.groupby(["season", "horizon", "as_of"], sort=True)
    ], ignore_index=True)
    joined = frame.merge(membership[KEYS + ["bottom8", "subgroup"]], on=KEYS,
                         how="left", validate="many_to_one", indicator=True)
    if len(joined) != len(frame) or not joined["_merge"].eq("both").all():
        raise ValueError("Membership must join every forecast exactly once.")
    scores = joined.groupby(["horizon", "subgroup", "model"], sort=True).agg(
        n=("brier", "size"), brier=("brier", "mean"),
        points_mae=("points_absolute_error", "mean"), points_crps=("points_crps", "mean"),
        coverage_90=("covered_90", "mean"),
    ).reset_index()
    comparisons = []
    for (horizon, subgroup), group in joined.groupby(["horizon", "subgroup"], sort=True):
        left = group[group.model == MODELS[0]].set_index(KEYS).sort_index()
        right = group[group.model == MODELS[1]].set_index(KEYS).sort_index()
        if not left.index.equals(right.index):
            raise ValueError("Subgroup model forecasts must remain exactly paired.")
        for metric in ("brier", "points_absolute_error"):
            interval = cluster_bootstrap_mean(
                (left[metric] - right[metric]).to_numpy(), left.index.get_level_values("season"),
                resamples=resamples, seed=DEFAULT_SEED,
            )
            comparisons.append({"horizon": horizon, "subgroup": subgroup, "metric": metric,
                                "n": len(left), "n_seasons": left.index.get_level_values("season").nunique(),
                                **vars(interval)})
    return membership, scores, pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, default=REPORTS_DIR / "season_validation_forecasts.csv")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    forecasts = pd.read_csv(args.forecasts, dtype={"season": str}, float_precision="round_trip")
    seasons = tuple(sorted(forecasts["season"].unique()))
    matches = load_all(seasons=seasons)
    membership, scores, comparisons = analyze_survival_subgroup(matches, forecasts)
    if len(seasons) != 7 or set(forecasts.horizon) != {"01-01", "03-01", "05-01"}:
        raise ValueError("This reported audit expects the existing seven-season, three-origin study.")
    expected_n = scores.subgroup.map({"bottom8": 56, "other12": 84})
    if not scores.n.eq(expected_n).all() or len(membership) != 420:
        raise ValueError("Expected 56 bottom-eight and 84 other-club rows per horizon/model.")
    settings = {
        "schema_version": 1,
        "forecast_file": str(args.forecasts.resolve().relative_to(PROJECT_ROOT)) if args.forecasts.resolve().is_relative_to(PROJECT_ROOT) else str(args.forecasts.resolve()),
        "forecast_sha256": hashlib.sha256(args.forecasts.read_bytes()).hexdigest(),
        "raw_match_sha256": {f"data/raw/I1_{season}.csv": hashlib.sha256((RAW_DIR / f"I1_{season}.csv").read_bytes()).hexdigest() for season in seasons},
        "status": "Retrospective exploratory subgroup audit chosen after inspecting historical outcomes; not a new holdout.",
        "seasons": seasons, "horizons": sorted(forecasts.horizon.unique()),
        "selection": "Exactly eight lowest clubs at each origin, using only same-season matches with date strictly before origin and known deductions inherited from validate_seasons.py.",
        "ordering": "Ascending current points, goal difference, goals for, then team name; eligibility_rank=1 is lowest. This is an eligibility convention, not official final-table tiebreaking.",
        "sanctions": "Juventus 2022/23 deductions: 15 points from 2023-01-21 to before 2023-04-21; 10 from 2023-05-23 to before 2023-07-01. Date-only inputs make changes effective at next midnight, matching the existing validation.",
        "roster": "Club identities are obtained from the exact saved forecast keys; predicted probabilities and realised future results do not select membership.",
        "counts": {"all_club_membership_rows": len(membership), "bottom8_per_horizon_model": 56, "other12_per_horizon_model": 84},
        "comparisons": "Dixon-Coles minus League Poisson, paired on season/horizon/as_of/team; negative Brier or points-error differences favor Dixon-Coles.",
        "bootstrap": {"function": "seriea.evaluation.inference.cluster_bootstrap_mean", "resamples": 10000, "seed": DEFAULT_SEED, "confidence": 0.95, "clusters": "Whole seasons", "n_clusters": 7},
        "limitations": ["Seven seasons limit interval coverage and calibration assessment.",
                        "Membership uses current points, not matches played or an estimated relegation-risk threshold.",
                        "The split and analyses are exploratory; intervals have no multiplicity adjustment.",
                        "No model refitting, new simulations, intervention effects or prospective validation are added.",
                        "Original all-club scores remain unchanged; subgroup estimates must be interpreted alongside both groups.",
                        "Intervals condition on fitted forecasts and do not propagate model-estimation or specification uncertainty."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("membership", membership), ("scores", scores), ("comparisons", comparisons)):
        frame.to_csv(args.output_dir / f"survival_subgroup_{name}.csv", index=False, float_format="%.17g")
    (args.output_dir / "survival_subgroup_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(scores.to_string(index=False))
    print(comparisons[comparisons.metric == "brier"].to_string(index=False))


if __name__ == "__main__":
    main()
