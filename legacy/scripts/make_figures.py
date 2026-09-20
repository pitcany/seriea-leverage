"""Generate the report figures.

Produces four panels into ``reports/figures``:

1. the decay-rate validation curve;
2. reliability diagrams for the market, the model and the base rate;
3. mean RPS with bootstrap intervals; and
4. the long decline of home advantage in Serie A.

Run: ``python scripts/make_figures.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seriea.config import FIGURES_DIR, REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.calibration import reliability_curve
from seriea.evaluation.metrics import outcomes_to_indicator

#: Consistent series colours across every panel.
COLOURS: dict[str, str] = {
    "market": "#1b6ca8",
    "model": "#c1121f",
    "base": "#6c757d",
    "accent": "#2a9d8f",
}

FIGURE_DPI: int = 150


def _save(figure: plt.Figure, name: str) -> None:
    """Write a figure to the figures directory and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def plot_decay_curve() -> None:
    """Plot validation RPS against the decay rate."""
    path = REPORTS_DIR / "decay_tuning.json"
    if not path.exists():
        print("! decay_tuning.json missing; skipping decay curve")
        return

    grid = pd.DataFrame(json.loads(path.read_text())["grid"])
    figure, axes = plt.subplots(figsize=(7, 4.2))
    axes.plot(grid["decay_rate"], grid["rps"], "o-", color=COLOURS["model"], linewidth=2)

    best = grid.loc[grid["rps"].idxmin()]
    axes.axvline(best["decay_rate"], color=COLOURS["accent"], linestyle="--", linewidth=1.2)
    axes.annotate(
        f"selected {best['decay_rate']:g}\n(half-life {best['half_life_days']:.0f} d)",
        xy=(best["decay_rate"], best["rps"]),
        xytext=(12, 22),
        textcoords="offset points",
        fontsize=9,
        color=COLOURS["accent"],
    )
    axes.set_xlabel("time-decay rate per day")
    axes.set_ylabel("validation RPS (lower is better)")
    axes.set_title("Decay rate chosen on 2013-19, not assumed", fontsize=11)
    axes.grid(alpha=0.25)
    _save(figure, "decay_curve.png")


def plot_reliability() -> None:
    """Plot home-win reliability diagrams for each forecaster."""
    path = REPORTS_DIR / "forecasts.parquet"
    if not path.exists():
        print("! forecasts.parquet missing; skipping reliability diagram")
        return

    forecasts = pd.read_parquet(path)
    test = forecasts[forecasts["is_test"]]
    outcomes = test["outcome"].to_numpy(dtype=object)
    home_occurred = outcomes_to_indicator(outcomes)[:, 0]

    series = {
        "Market (Shin)": (test["p_market_H"].to_numpy(), COLOURS["market"]),
        "Dixon-Coles": (test["p_H"].to_numpy(), COLOURS["model"]),
    }

    figure, axes = plt.subplots(figsize=(5.6, 5.4))
    axes.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle=":", label="perfect")

    for label, (probabilities, colour) in series.items():
        curve = reliability_curve(probabilities, home_occurred, bins=10)
        axes.plot(curve.bin_centre, curve.observed, "o-", color=colour, label=label, linewidth=1.8)

    axes.set_xlabel("forecast probability of a home win")
    axes.set_ylabel("observed frequency")
    axes.set_title("Both forecasts are well calibrated\n(the market is simply sharper)", fontsize=11)
    axes.legend(frameon=False, fontsize=9)
    axes.grid(alpha=0.25)
    axes.set_aspect("equal")
    _save(figure, "reliability.png")


def plot_scores() -> None:
    """Plot mean RPS with bootstrap intervals for every forecaster."""
    path = REPORTS_DIR / "backtest_scores.csv"
    if not path.exists():
        print("! backtest_scores.csv missing; skipping score panel")
        return

    scores = pd.read_csv(path).sort_values("rps", ascending=False)
    figure, axes = plt.subplots(figsize=(8, 4))

    positions = np.arange(len(scores))
    errors = np.vstack(
        [scores["rps"] - scores["rps_lower"], scores["rps_upper"] - scores["rps"]]
    )
    colours = [
        COLOURS["market"] if "Market" in name else
        COLOURS["model"] if "Dixon" in name else COLOURS["base"]
        for name in scores["model"]
    ]
    axes.barh(positions, scores["rps"], xerr=errors, color=colours, alpha=0.85, height=0.6)
    axes.set_yticks(positions)
    axes.set_yticklabels(scores["model"], fontsize=9)
    axes.set_xlabel("mean RPS on 2019-26 test period (lower is better)")
    axes.set_xlim(0.17, 0.24)
    axes.set_title("The market wins; the model beats only the naive baselines", fontsize=11)
    axes.grid(alpha=0.25, axis="x")
    _save(figure, "scores.png")


def plot_home_advantage() -> None:
    """Plot the season-by-season decline in home advantage."""
    matches = load_all()
    by_season = matches.groupby("season_start_year").agg(
        home_goals=("home_goals", "mean"),
        away_goals=("away_goals", "mean"),
        home_win_rate=("outcome", lambda column: (column == "H").mean()),
    )

    figure, axes = plt.subplots(figsize=(7.4, 4.2))
    axes.plot(
        by_season.index, by_season["home_win_rate"], "o-",
        color=COLOURS["model"], linewidth=2, label="home-win rate",
    )
    trend = np.poly1d(np.polyfit(by_season.index, by_season["home_win_rate"], 1))
    axes.plot(by_season.index, trend(by_season.index), "--", color=COLOURS["base"], label="linear trend")

    axes.set_xlabel("season (starting year)")
    axes.set_ylabel("share of matches won by the home side")
    axes.set_title("Home advantage in Serie A has eroded since 2007", fontsize=11)
    axes.legend(frameon=False, fontsize=9)
    axes.grid(alpha=0.25)
    _save(figure, "home_advantage.png")


def main() -> None:
    """Generate every figure."""
    plot_decay_curve()
    plot_reliability()
    plot_scores()
    plot_home_advantage()


if __name__ == "__main__":
    main()
