"""Supporting analyses and vector figures for the paper.

The headline tables answer "which model scores best". A referee will reasonably
ask three further questions, and this script answers them:

1. **Is the market's advantage stable, or driven by one anomalous season?**
   Season-by-season scores on the test period.
2. **Is the zero pooling weight a boundary artefact of the optimiser?**
   The full log-loss profile across the weight simplex, on validation and test.
3. **Is the model's calibration claim visible, not just a slope coefficient?**
   Reliability curves for both forecasters.

Figures are written as PDF so they embed as vectors in the LaTeX build.

Run: ``python scripts/paper_supplements.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seriea.config import FIGURES_DIR, OUTCOMES, REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.backtest import rolling_origin_forecasts
from seriea.evaluation.calibration import reliability_curve
from seriea.evaluation.metrics import log_loss, outcomes_to_indicator, ranked_probability_score
from seriea.models.blend import logarithmic_pool
from seriea.models.shots import ShotsForecaster

BACKTEST_START = pd.Timestamp("2013-08-01")
TEST_START = pd.Timestamp("2019-08-01")

COLOURS = {"market": "#1b6ca8", "goals": "#c1121f", "shots": "#e08b2a", "grid": "#cccccc"}
FIGURE_KWARGS = {"bbox_inches": "tight"}


def season_label(code: str) -> str:
    """Render a season code as a readable label, e.g. ``1920`` -> ``19/20``."""
    return f"{code[:2]}/{code[2:]}"


def load_forecasts() -> pd.DataFrame:
    """Load cached goals/market forecasts and attach shot-based forecasts."""
    cached = Path(REPORTS_DIR) / "forecasts.parquet"
    if not cached.exists():
        raise FileNotFoundError(
            f"{cached} missing. Run scripts/run_backtest.py first."
        )
    frame = pd.read_parquet(cached)

    decay = float(json.loads((Path(REPORTS_DIR) / "decay_tuning.json").read_text())["best"]["decay_rate"])
    print("recomputing shot-based forecasts (a few minutes)...", flush=True)
    shots = rolling_origin_forecasts(
        load_all(), lambda: ShotsForecaster(decay_rate=decay), BACKTEST_START
    )

    keys = ["date", "home", "away"]
    merged = frame.merge(
        shots[keys + [f"p_{o}" for o in OUTCOMES]].rename(
            columns={f"p_{o}": f"p_shots_{o}" for o in OUTCOMES}
        ),
        on=keys,
        how="left",
    )
    if merged[[f"p_shots_{o}" for o in OUTCOMES]].isna().any().any():
        raise ValueError("Shot forecasts failed to align with the cached backtest.")
    return merged


def matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Extract an (n, 3) probability matrix for a given column prefix."""
    return frame[[f"{prefix}{o}" for o in OUTCOMES]].to_numpy(dtype=float)


def season_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Score each test season separately."""
    test = frame[frame["is_test"]]
    outcomes = test["outcome"].to_numpy(dtype=object)

    scores = {
        "market": ranked_probability_score(matrix(test, "p_market_"), outcomes),
        "goals": ranked_probability_score(matrix(test, "p_"), outcomes),
        "shots": ranked_probability_score(matrix(test, "p_shots_"), outcomes),
    }

    rows = []
    for season, block in test.groupby("season"):
        mask = (test["season"] == season).to_numpy()
        row = {"season": season_label(season), "n": int(mask.sum())}
        row.update({name: float(values[mask].mean()) for name, values in scores.items()})
        row["gap"] = row["goals"] - row["market"]
        rows.append(row)
    return pd.DataFrame(rows)


def pool_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Trace log loss across the pooling weight, on validation and on test."""
    is_test = frame["is_test"].to_numpy()
    market, goals = matrix(frame, "p_market_"), matrix(frame, "p_")
    outcomes = frame["outcome"].to_numpy(dtype=object)

    rows = []
    for weight in np.round(np.arange(0.0, 1.01, 0.05), 2):
        pooled = logarithmic_pool(market, goals, float(weight))
        rows.append(
            {
                "weight": float(weight),
                "validation": float(log_loss(pooled[~is_test], outcomes[~is_test]).mean()),
                "test": float(log_loss(pooled[is_test], outcomes[is_test]).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_pool_profile(profile: pd.DataFrame) -> None:
    """Plot the log-loss profile, showing the optimum sits at the boundary."""
    figure, axes = plt.subplots(figsize=(5.4, 3.4))
    for column, style in (("validation", "-"), ("test", "--")):
        axes.plot(
            profile["weight"], profile[column], style,
            color=COLOURS["goals"], linewidth=1.8, label=column,
        )
    axes.set_xlabel("weight on the structural model")
    axes.set_ylabel("mean log loss")
    axes.legend(frameon=False, fontsize=9)
    axes.grid(alpha=0.3)
    figure.savefig(FIGURES_DIR / "fig_pool_profile.pdf", **FIGURE_KWARGS)
    plt.close(figure)


def plot_calibration(frame: pd.DataFrame) -> None:
    """Plot home-win reliability curves for the market and the goals model."""
    test = frame[frame["is_test"]]
    outcomes = test["outcome"].to_numpy(dtype=object)
    occurred = outcomes_to_indicator(outcomes)[:, 0]

    figure, axes = plt.subplots(figsize=(4.6, 4.4))
    axes.plot([0, 1], [0, 1], ":", color="black", linewidth=1, label="perfect")
    for prefix, name, colour in (
        ("p_market_", "market", COLOURS["market"]),
        ("p_", "goals model", COLOURS["goals"]),
    ):
        curve = reliability_curve(matrix(test, prefix)[:, 0], occurred, bins=10)
        axes.plot(curve.bin_centre, curve.observed, "o-", color=colour, label=name, linewidth=1.6)

    axes.set_xlabel("forecast probability of a home win")
    axes.set_ylabel("observed frequency")
    axes.set_aspect("equal")
    axes.legend(frameon=False, fontsize=9, loc="upper left")
    axes.grid(alpha=0.3)
    figure.savefig(FIGURES_DIR / "fig_calibration.pdf", **FIGURE_KWARGS)
    plt.close(figure)


def plot_season_scores(table: pd.DataFrame) -> None:
    """Plot per-season RPS for all three forecasters."""
    figure, axes = plt.subplots(figsize=(6.4, 3.4))
    positions = np.arange(len(table))
    for column, name, colour in (
        ("market", "market", COLOURS["market"]),
        ("goals", "goals model", COLOURS["goals"]),
        ("shots", "shots model", COLOURS["shots"]),
    ):
        axes.plot(positions, table[column], "o-", color=colour, label=name, linewidth=1.6, markersize=4)
    axes.set_xticks(positions)
    axes.set_xticklabels(table["season"], fontsize=8)
    axes.set_ylabel("mean RPS")
    axes.legend(frameon=False, fontsize=9, ncol=3)
    axes.grid(alpha=0.3)
    figure.savefig(FIGURES_DIR / "fig_season_rps.pdf", **FIGURE_KWARGS)
    plt.close(figure)


def plot_home_advantage() -> None:
    """Plot the season-by-season erosion of home advantage."""
    matches = load_all()
    by_season = matches.groupby("season_start_year").agg(
        home_win_rate=("outcome", lambda column: (column == "H").mean()),
    )

    figure, axes = plt.subplots(figsize=(6.0, 3.2))
    axes.plot(by_season.index, by_season["home_win_rate"], "o-", color=COLOURS["goals"], linewidth=1.8)
    trend = np.poly1d(np.polyfit(by_season.index, by_season["home_win_rate"], 1))
    axes.plot(by_season.index, trend(by_season.index), "--", color="#777777", linewidth=1.2)
    axes.set_xlabel("season (starting year)")
    axes.set_ylabel("home win rate")
    axes.grid(alpha=0.3)
    figure.savefig(FIGURES_DIR / "fig_home_advantage.pdf", **FIGURE_KWARGS)
    plt.close(figure)


def plot_decay() -> None:
    """Plot the validation RPS profile across the decay grid."""
    path = Path(REPORTS_DIR) / "decay_tuning.json"
    grid = pd.DataFrame(json.loads(path.read_text())["grid"])

    figure, axes = plt.subplots(figsize=(5.4, 3.2))
    axes.plot(grid["decay_rate"], grid["rps"], "o-", color=COLOURS["goals"], linewidth=1.8)
    best = grid.loc[grid["rps"].idxmin()]
    axes.axvline(best["decay_rate"], color="#2a9d8f", linestyle="--", linewidth=1.2)
    axes.set_xlabel(r"decay rate $\xi$ (per day)")
    axes.set_ylabel("validation RPS")
    axes.grid(alpha=0.3)
    figure.savefig(FIGURES_DIR / "fig_decay.pdf", **FIGURE_KWARGS)
    plt.close(figure)


def main() -> None:
    """Generate every supplement and report the headline numbers."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_forecasts()

    seasons = season_table(frame)
    profile = pool_profile(frame)

    output = Path(REPORTS_DIR)
    seasons.to_csv(output / "paper_season_rps.csv", index=False)
    profile.to_csv(output / "paper_pool_profile.csv", index=False)

    plot_pool_profile(profile)
    plot_calibration(frame)
    plot_season_scores(seasons)
    plot_home_advantage()
    plot_decay()

    print("\n=== SEASON-BY-SEASON TEST RPS ===")
    print(seasons.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    beaten = int((seasons["gap"] > 0).sum())
    print(f"\nseasons where the market beats the goals model: {beaten}/{len(seasons)}")

    print("\n=== POOL PROFILE (first rows) ===")
    print(profile.head(5).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(
        "validation argmin weight:",
        float(profile.loc[profile["validation"].idxmin(), "weight"]),
        "| test argmin weight:",
        float(profile.loc[profile["test"].idxmin(), "weight"]),
    )
    print(f"\nWrote tables to {output} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
