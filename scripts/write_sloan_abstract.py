"""Generate a Sloan-format abstract from the repaired reports; enforce <500 words."""
from pathlib import Path
import json,re
import pandas as pd
from seriea.config import PROJECT_ROOT,REPORTS_DIR

def main():
    out=Path(REPORTS_DIR);s=pd.read_csv(out/"revised_scores.csv").set_index("model")
    comparisons=pd.read_csv(out/"revised_comparisons.csv")
    def row(model,reference):
        return comparisons[(comparisons.model==model)&(comparisons.reference==reference)&
            (comparisons.metric=="rps")&(comparisons.bootstrap=="season")].iloc[0]
    gap=row("Goals calibrated","Market")
    season=pd.read_csv(out/"season_validation_scores.csv")
    a=season[(season.horizon=="03-01")&(season.model=="Dixon-Coles")].iloc[0]
    b=season[(season.horizon=="03-01")&(season.model=="League Poisson")].iloc[0]
    stakes=pd.read_csv(out/"fiorentina_stakes.csv").set_index("opponent")
    lecce=stakes.loc["Lecce"]
    assert lecce.ratio_mc_ci_status=="bounded"
    season_ci=pd.read_csv(out/"season_validation_comparisons.csv")
    pts=season_ci[(season_ci.horizon=="03-01")&(season_ci.metric=="points_absolute_error")].iloc[0]
    surv=season_ci[(season_ci.horizon=="03-01")&(season_ci.metric=="brier")].iloc[0]
    subgroup=pd.read_csv(out/"survival_subgroup_scores.csv")
    bottom=subgroup[(subgroup.horizon=="03-01")&(subgroup.subgroup=="bottom8")].set_index("model")
    subgroup_ci=pd.read_csv(out/"survival_subgroup_comparisons.csv")
    bottom_survival=subgroup_ci[(subgroup_ci.horizon=="03-01")&(subgroup_ci.subgroup=="bottom8")&(subgroup_ci.metric=="brier")].iloc[0]
    text=f"""# Beyond Points Accuracy: Survival Forecasts and Fixture Stakes in Serie A

**Introduction**

Does improved final-points forecasting establish better survival probabilities? We evaluate both targets in Serie A, then apply established conditional match importance to a separate scenario question: when does avoiding defeat have more survival value than turning a draw into a win?

**Methods**

We use 7,220 Serie A matches from 2007/08–2025/26. Goals-based Dixon–Coles and independent-Poisson shots models use strictly prior results and a four-year training window. Parameters are selected on 2013/14–2018/19; 2,660 subsequent fixtures evaluate match forecasts against documented closing prices. A retrospective season study compares raw goals-model simulations with a league-average Poisson baseline for all clubs at three dates across seven seasons, implementing head-to-head and playoff rules. Paired uncertainty resamples seasons. An exploratory breakdown selects the bottom eight using standings available at each forecast origin. For Fiorentina at March 1, 2026, 20,000 simulations estimate survival conditional on wins, draws and losses. Hypothetical probability transfers preserve conditional scorelines and all other fixtures; their intervals measure simulation error only.

**Results**

March points mean absolute error is {a.points_mae:.2f}, versus {b.points_mae:.2f} for the league-average baseline: difference {pts.estimate:+.2f}, 95% season interval [{pts.lower:+.2f}, {pts.upper:+.2f}]. Survival Brier scores are {a.brier:.4f} and {b.brier:.4f}: difference {surv.estimate:+.4f} [{surv.lower:+.4f}, {surv.upper:+.4f}]. Points gains have intervals excluding zero at all three dates; survival gains remain unresolved. Among the origin-defined bottom eight, March points MAE is {bottom.loc['Dixon-Coles','points_mae']:.2f} versus {bottom.loc['League Poisson','points_mae']:.2f}; survival Brier difference is {bottom_survival.estimate:+.4f} [{bottom_survival.lower:+.4f}, {bottom_survival.upper:+.4f}]. Match RPS is {s.loc['Goals calibrated','rps']:.5f} for calibrated goals versus {s.loc['Market','rps']:.5f} for the closing market. In the separate Lecce illustration, the draw-over-loss survival difference is {100*lecce.draw_over_loss:.2f} percentage points, versus {100*lecce.win_over_draw:.2f} for win-over-draw. Moving five percentage points from draws to each of wins and losses adds {lecce.scenario_delta_expected_points:.2f} expected league points but changes Fiorentina's survival by {100*lecce.scenario_delta_survival:+.3f} percentage points (approximate 95% simulation interval [{100*lecce.scenario_delta_survival_mc_ci_low:+.3f}, {100*lecce.scenario_delta_survival_mc_ci_high:+.3f}]).

**Conclusion**

Points accuracy and survival probabilities require separate evaluation. Distinguishing avoiding defeat from turning draws into wins also makes hypothetical probability trade-offs explicit, without explaining the historical forecast-score differences or identifying an effective tactic. All analyses are retrospective; seven season clusters and one club's small illustrative effects limit inference. Prospective forecast and action validation remain necessary.
"""
    words=len(re.findall(r"\S+",text))
    assert words<500,words
    (PROJECT_ROOT/"paper/SLOAN_ABSTRACT.md").write_text(text)
    (out/"abstract_metadata.json").write_text(json.dumps({"words_including_title_and_headings":words,
        "limit":"fewer than 500","status":"draft; not submitted"},indent=2))
    print(f"Wrote Sloan abstract: {words} words including title and headings.")
if __name__=="__main__":main()
