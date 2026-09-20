"""Generate manuscript numbers, tables and figures directly from revised reports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seriea.config import PROJECT_ROOT,REPORTS_DIR

def main():
    paper=PROJECT_ROOT/"paper";out=Path(REPORTS_DIR);figures=out/"figures";figures.mkdir(exist_ok=True)
    scores=pd.read_csv(out/"revised_scores.csv").set_index("model")
    comparisons=pd.read_csv(out/"revised_comparisons.csv")
    seasons=pd.read_csv(out/"revised_season_rps.csv",dtype={"season":str})
    season_scores=pd.read_csv(out/"season_validation_scores.csv")
    season_ci=pd.read_csv(out/"season_validation_comparisons.csv")
    settings=json.loads((out/"revised_analysis_settings.json").read_text())
    tuning=json.loads((out/"decay_tuning.json").read_text())
    fio=json.loads((out/"fiorentina_projection.json").read_text())
    leverage=pd.read_csv(out/"fiorentina_leverage.csv")
    stakes=pd.read_csv(out/"fiorentina_stakes.csv")
    subgroup_scores=pd.read_csv(out/"survival_subgroup_scores.csv")
    subgroup_comparisons=pd.read_csv(out/"survival_subgroup_comparisons.csv")
    def comparison(model,reference="Market",metric="rps"):
        return comparisons[(comparisons.model==model)&(comparisons.reference==reference)&
            (comparisons.metric==metric)&(comparisons.bootstrap=="season")].iloc[0]
    def ci(row,digits=5):
        return f"[{row.lower:+.{digits}f}, {row.upper:+.{digits}f}]"
    gap=comparison("Goals calibrated")
    shots=comparison("Goals + shots","Goals raw")
    nested=comparison("Market + structural logits","Market recalibrated")
    nested_log=comparison("Market + structural logits","Market recalibrated","log_loss")
    march=season_scores[(season_scores.horizon=="03-01")&(season_scores.model=="Dixon-Coles")].iloc[0]
    baseline=season_scores[(season_scores.horizon=="03-01")&(season_scores.model=="League Poisson")].iloc[0]
    march_ci=season_ci[(season_ci.horizon=="03-01")&(season_ci.metric=="brier")].iloc[0]
    march_points=season_ci[(season_ci.horizon=="03-01")&(season_ci.metric=="points_absolute_error")].iloc[0]
    pivot=seasons.pivot(index="season",columns="model",values="rps")
    values={
        "GoalsRPS":f"{scores.loc['Goals calibrated','rps']:.5f}",
        "MarketRPS":f"{scores.loc['Market','rps']:.5f}",
        "GapRPS":f"{gap.estimate:+.5f}","GapCI":ci(gap),
        "SelectedDecay":str(tuning["best"]["decay_rate"]),
        "HalfLife":f"{np.log(2)/tuning['best']['decay_rate']:.1f}",
        "MarketWinningSeasons":str(int((pivot.Market<pivot["Goals calibrated"]).sum())),
        "GoalsShotsWeights":"/".join(f"{v:.2f}" for v in settings["goals_shots_weights"]),
        "ShotsGain":f"{shots.estimate:+.5f}","ShotsGainCI":ci(shots),
        "NestedGain":f"{nested.estimate:+.5f}","NestedGainCI":ci(nested),
        "NestedLogGain":f"{nested_log.estimate:+.5f}","NestedLogCI":ci(nested_log),
        "NestedInterpretation":("The RPS interval includes zero, so these data do not establish a stable improvement from the added structural forecasts."
            if nested.lower<=0<=nested.upper else
            "The RPS interval excludes zero in this retrospective comparison; replication on a new period is needed before treating the difference as stable."),
        "SeasonMarchBrier":f"{march.brier:.4f}","SeasonMarchBaseline":f"{baseline.brier:.4f}",
        "SeasonMarchDifference":f"{march_ci.estimate:+.4f}",
        "SeasonMarchDifferenceCI":ci(march_ci,4),
        "SeasonMarchCoverage":f"{100*march.coverage_90:.1f}\\%",
        "SeasonMarchMAE":f"{march.points_mae:.2f}",
        "SeasonMarchBaselineMAE":f"{baseline.points_mae:.2f}",
        "SeasonMarchMAEDifference":f"{march_points.estimate:+.2f}",
        "SeasonMarchMAECI":ci(march_points,2),
        "SeasonInterpretation":("Coverage is below the nominal 90\\%, consistent with uncertainty missing from a frozen-rate simulator."
            if march.coverage_90<.9 else
            "Coverage at this horizon is at least 90\\%; this finite sample does not establish calibration across horizons or subgroups."),
        "FioSurvival":f"{100*fio['survival']:.2f}\\%",
        "FioSimplified":f"{100*fio['simplified_survival']:.2f}\\%",
        "FioPoints":f"{fio['expected_points']:.2f}",
        "PlayoffFrequency":f"{100*fio['diagnostics']['survival_playoffs']/fio['n_simulations']:.2f}\\%",
        "TopOpponent":str(leverage.iloc[0].opponent),
        "TopImportance":f"{100*leverage.iloc[0].leverage:.2f}",
        "InterImportance":f"{100*leverage[leverage.opponent=='Inter'].iloc[0].leverage:.2f}",
    }
    def mc_interval(row,prefix,scale=100,digits=3,signed=False):
        low=getattr(row,prefix+"_mc_ci_low")*scale
        high=getattr(row,prefix+"_mc_ci_high")*scale
        spec=("+" if signed else "")+f".{digits}f"
        return f"[{low:{spec}}, {high:{spec}}]"
    for opponent in ("Lecce","Lazio"):
        row=stakes[stakes.opponent==opponent].iloc[0]
        assert row.ratio_mc_ci_status=="bounded", "Narrative requires a bounded break-even interval."
        values.update({
            opponent+"WinBenefit":f"{100*row.win_over_draw:.2f}",
            opponent+"DrawBenefit":f"{100*row.draw_over_loss:.2f}",
            opponent+"BreakEven":f"{row.draw_loss_to_win_draw_ratio:.2f}",
            opponent+"BreakEvenCI":f"[{row.ratio_mc_ci_low:.2f}, {row.ratio_mc_ci_high:.2f}]",
            opponent+"ScenarioChange":f"{100*row.scenario_delta_survival:+.3f}",
            opponent+"ScenarioCI":mc_interval(row,"scenario_delta_survival",signed=True),
        })
    values["ScenarioPointsGain"]=f"{stakes.iloc[0].scenario_delta_expected_points:.2f}"
    values["SeasonMarchMAEReduction"]=f"{100*(1-march.points_mae/baseline.points_mae):.0f}\\%"
    for group,prefix in (("bottom8","BottomMarch"),("other12","OtherMarch")):
        selected=subgroup_scores[(subgroup_scores.horizon=="03-01")&(subgroup_scores.subgroup==group)].set_index("model")
        brier_digits=4 if group=="bottom8" else 6
        for model,label in (("Dixon-Coles",""),("League Poisson","Baseline")):
            values[prefix+label+"MAE"]=f"{selected.loc[model,'points_mae']:.2f}"
            values[prefix+label+"Brier"]=f"{selected.loc[model,'brier']:.{brier_digits}f}"
        for metric,label,digits in (("points_absolute_error","MAE",2),("brier","Brier",brier_digits)):
            result=subgroup_comparisons[(subgroup_comparisons.horizon=="03-01")&
                (subgroup_comparisons.subgroup==group)&(subgroup_comparisons.metric==metric)].iloc[0]
            values[prefix+label+"Difference"]=f"{result.estimate:+.{digits}f}"
            values[prefix+label+"CI"]=ci(result,digits)
    (paper/"generated-results.tex").write_text("\n".join(
        "\\newcommand{\\"+key+"}{"+value+"}" for key,value in values.items())+"\n")
    def table(path,headers,rows,layout):
        text="\\small\n\\begin{tabular}{"+layout+"}\n\\toprule\n"+" & ".join(headers)+r" \\"+"\n\\midrule\n"
        text+="\n".join(" & ".join(map(str,row))+r" \\" for row in rows)
        text+="\n\\bottomrule\n\\end{tabular}\n"
        (paper/path).write_text(text)
    rows=[]
    for model,row in scores.sort_values("rps").iterrows():
        rows.append([model,f"{row.rps:.5f}",f"{row.log_loss:.4f}",f"{row.brier:.4f}",f"{100*row.accuracy:.1f}\\%"])
    table("generated-match-table.tex",["Forecast","RPS","Log loss","Brier","Accuracy"],rows,"lrrrr")
    rows=[]
    for row in season_scores.itertuples():
        label={"01-01":"Jan 1","03-01":"Mar 1","05-01":"May 1"}[row.horizon]
        rows.append([label,row.model,f"{row.brier:.4f}",f"{row.points_mae:.2f}",
            f"{row.points_crps:.2f}",f"{100*row.coverage_90:.1f}\\%"])
    table("generated-season-table.tex",["Date","Model","Brier","Pts MAE","Pts CRPS","Coverage"],rows,"llrrrr")
    rows=[]
    for horizon in ("01-01","03-01","05-01"):
        for group,label in (("bottom8","Bottom eight"),("other12","Other twelve")):
            results=subgroup_comparisons[(subgroup_comparisons.horizon==horizon)&(subgroup_comparisons.subgroup==group)].set_index("metric")
            points=results.loc["points_absolute_error"];survival=results.loc["brier"]
            rows.append([{"01-01":"Jan 1","03-01":"Mar 1","05-01":"May 1"}[horizon],label,int(points.n),
                f"{points.estimate:+.2f} "+ci(points,2),f"{survival.estimate:+.6f} "+ci(survival,6)])
    table("generated-subgroup-table.tex",["Date","Origin group",r"$n$","Points MAE difference","Survival Brier difference"],rows,"llrrr")
    rows=[]
    selected=stakes.head(3)
    selected=pd.concat([selected,stakes[stakes.opponent.isin(["Lazio","Inter"])]]).drop_duplicates(["date","opponent"])
    for row in selected.itertuples():
        assert row.scenario_feasible
        ratio=(f"{row.draw_loss_to_win_draw_ratio:.2f} [{row.ratio_mc_ci_low:.2f}, {row.ratio_mc_ci_high:.2f}]"
               if row.ratio_mc_ci_status=="bounded" else "Unbounded/undefined")
        rows.append([row.opponent,row.venue,
            f"{100*row.win_over_draw:.2f} "+mc_interval(row,"win_over_draw",digits=2),
            f"{100*row.draw_over_loss:.2f} "+mc_interval(row,"draw_over_loss",digits=2),ratio])
    table("generated-leverage-table.tex",["Opponent","Venue",r"Win over draw $U$",r"Draw over loss $A$",r"Break-even $A/U$"],rows,"llrrr")
    plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
        "pdf.fonttype":42,"axes.labelcolor":"#253342","text.color":"#253342"})
    colors={"Goals calibrated":"#245a81","Shots":"#b3632f","Goals + shots":"#548554",
            "Market recalibrated":"#6d5082","Market + structural logits":"#363636"}
    fig,axes=plt.subplots(2,1,figsize=(7.2,5.5),layout="constrained")
    for model in ("Goals calibrated","Shots","Goals + shots"):
        axes[0].plot(range(7),pivot[model]-pivot.Market,"o-",label=model,color=colors[model],lw=1.4,ms=4)
    axes[0].axhline(0,color="gray",lw=.8)
    axes[0].set_xticks(range(7),[f"{s[:2]}/{s[2:]}" for s in pivot.index],rotation=40)
    axes[0].set_ylabel("RPS difference from market");axes[0].legend(fontsize=8)
    labels=list(colors)
    for j,model in enumerate(labels):
        row=comparison(model)
        axes[1].errorbar(row.estimate,j,xerr=[[row.estimate-row.lower],[row.upper-row.estimate]],
                        fmt="o",capsize=3,color=colors[model])
    axes[1].set_yticks(range(len(labels)),["Goals calibrated","Shots","Goals + shots","Market recalibrated","Added structural logits"])
    axes[1].axvline(0,color="gray",lw=.8);axes[1].set_xlabel("Test RPS difference from raw market")
    axes[1].invert_yaxis()
    fig.savefig(figures/"revised_match_evaluation.pdf");plt.close(fig)
    reliability=pd.read_csv(out/"revised_reliability.csv")
    fig,axes=plt.subplots(4,3,figsize=(7.2,6.0),gridspec_kw={"height_ratios":[3,1,3,1]},layout="constrained")
    for i,model in enumerate(("Goals raw","Market")):
        for j,outcome in enumerate(("H","D","A")):
            ax=axes[2*i,j];counts=axes[2*i+1,j];data=reliability[(reliability.model==model)&(reliability.outcome==outcome)]
            ax.plot([0,1],[0,1],color="#abb2b9",lw=1)
            ax.errorbar(data.mean_forecast,data.observed,yerr=[np.maximum(0,data.observed-data.lower),np.maximum(0,data.upper-data.observed)],
                        fmt="none",color="#8aa4b8",capsize=2,lw=.8)
            ax.scatter(data.mean_forecast,data.observed,s=10+2*np.sqrt(data.n),color="#245a81")
            centers=.05+.1*data.bin
            counts.bar(centers,data.n,width=.075,color="#aebfcb")
            for row in data.itertuples():
                counts.text(.05+.1*row.bin,data.n.max()*(1.06+.30*(row.bin%2)),str(row.n),ha="center",va="bottom",fontsize=6.8)
            counts.set(xlim=(-.03,1.05),ylim=(0,data.n.max()*1.68),yticks=[])
            counts.set_xticks([0,.25,.5,.75,1])
            ax.set(xlim=(-.03,1.05),ylim=(-.03,1.05),title=f"{model}: {outcome}")
            ax.tick_params(labelbottom=False)
            if i==1:counts.set_xlabel("Predicted probability / bin")
            if j==0:
                ax.set_ylabel("Observed frequency")
                counts.set_ylabel("Count",fontsize=8)
    fig.savefig(figures/"revised_reliability.pdf");plt.close(fig)
    (out/"manuscript_values.json").write_text(json.dumps(values,indent=2))
    print(json.dumps(values,indent=2))

if __name__=="__main__":
    main()
