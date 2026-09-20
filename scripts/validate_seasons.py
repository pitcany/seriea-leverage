"""Retrospective all-club season forecasts at Jan 1, Mar 1 and May 1, 2019/20-2025/26.

This study was specified after reviewing the historical data. It is exploratory,
not a preregistered or untouched holdout. Future fixture identities and realised
schedule dates are available retrospectively; no future scores or prices are used.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import poisson
from seriea.config import REPORTS_DIR
from seriea.data.load import load_all
from seriea.data.teams import teams_in_season
from seriea.models.dixon_coles import DixonColesForecaster
from seriea.simulation.season import current_standings,simulate_season
from seriea.simulation.rules import playoff_score_grids
from seriea.evaluation.inference import cluster_bootstrap_mean

SEED=20260920
HORIZONS=("01-01","03-01","05-01")

def deduction_at(as_of):
    """Known Juventus sanctions, effective from the next midnight (date-only inputs)."""
    if pd.Timestamp("2023-01-21")<=as_of<pd.Timestamp("2023-04-21"):
        return 15
    if pd.Timestamp("2023-05-23")<=as_of<pd.Timestamp("2023-07-01"):
        return 10
    return 0

def league_grid(history, as_of, decay, neutral=False):
    weights=np.exp(-decay*(as_of-history.date).dt.total_seconds().to_numpy()/86400)
    h=np.average(history.home_goals.to_numpy(dtype=float),weights=weights)
    a=np.average(history.away_goals.to_numpy(dtype=float),weights=weights)
    if neutral:
        h=a=np.sqrt(h*a)
    goals=np.arange(13)
    grid=poisson.pmf(goals[:,None],h)*poisson.pmf(goals[None,:],a)
    return grid/grid.sum()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations",type=int,default=10000)
    args=parser.parse_args()
    output=Path(REPORTS_DIR);output.mkdir(exist_ok=True)
    decay=json.loads((output/"decay_tuning.json").read_text())["best"]["decay_rate"]
    matches=load_all(); records=[];diagnostics=[]
    for year in range(2019,2026):
        season=f"{year%100:02d}{(year+1)%100:02d}"
        data=matches[matches.season==season]
        teams=teams_in_season(matches,season)
        actual_table=current_standings(data,teams)
        if year==2022:
            actual_table.loc["Juventus","points"]-=10
        # Completed season: head-to-head ordering with no stochastic playoff.
        empty=data.iloc[:0]
        actual_sim=simulate_season(actual_table,empty,np.zeros((0,13,13)),n_simulations=1,
            played=data,season_start=2021,seed=SEED)
        actual_survival=(actual_sim.positions[0]<=17)
        if year==2022:
            # Actual neutral playoff: Spezia 1-3 Verona, 11 June 2023.
            actual_survival[teams.index("Spezia")]=False
            actual_survival[teams.index("Verona")]=True
        for horizon in HORIZONS:
            as_of=pd.Timestamp(f"{year+1}-{horizon}")
            played=data[data.date<as_of]
            remaining=data[data.date>=as_of].reset_index(drop=True)
            history=matches[(matches.date<as_of)&(matches.date>=as_of-pd.Timedelta(days=1460))]
            assert history.date.max()<as_of
            standings=current_standings(played,teams)
            if year==2022:
                standings.loc["Juventus","points"]-=deduction_at(as_of)
            model=DixonColesForecaster(decay_rate=decay).fit(history,as_of)
            if not model.converged:
                raise RuntimeError(f"Unconverged season fit {as_of}")
            for name in ("Dixon-Coles","League Poisson"):
                if name=="Dixon-Coles":
                    grids=model.predict_score_grid(remaining)
                    playoffs=playoff_score_grids(model,teams)
                    neutral=playoff_score_grids(model,teams,neutral=True)
                else:
                    grid=league_grid(history,as_of,decay)
                    ng=league_grid(history,as_of,decay,neutral=True)
                    grids=np.broadcast_to(grid,(len(remaining),13,13))
                    playoffs=np.broadcast_to(grid,(20,20,13,13))
                    neutral=np.broadcast_to(ng,(20,20,13,13))
                sim=simulate_season(standings,remaining,grids,n_simulations=args.simulations,
                    seed=SEED,played=played,season_start=year,playoff_grids=playoffs,
                    neutral_playoff_grids=neutral)
                diagnostics.append({"season":season,"as_of":str(as_of.date()),"model":name,**sim.diagnostics})
                for j,team in enumerate(teams):
                    samples=sim.points[:,j];actual=float(actual_table.loc[team,"points"])
                    q=float((sim.positions[:,j]<=17).mean())
                    lo,hi=np.quantile(samples,[.05,.95])
                    n=len(samples);ordered=np.sort(samples)
                    crps=np.abs(samples-actual).mean()-np.sum((2*np.arange(1,n+1)-n-1)*ordered)/n**2
                    records.append({"season":season,"horizon":horizon,"as_of":str(as_of.date()),
                        "team":team,"model":name,"survival_probability":q,
                        "survived":int(actual_survival[j]),"brier":(q-int(actual_survival[j]))**2,
                        "expected_points":float(samples.mean()),"actual_points":actual,
                        "points_absolute_error":abs(samples.mean()-actual),"points_crps":crps,
                        "points_lower_90":lo,"points_upper_90":hi,"covered_90":int(lo<=actual<=hi),
                        "remaining_fixtures":int(((remaining.home==team)|(remaining.away==team)).sum())})
            print(f"Completed {season} {horizon}",flush=True)
    frame=pd.DataFrame(records);frame.to_csv(output/"season_validation_forecasts.csv",index=False)
    summary=frame.groupby(["horizon","model"]).agg(n=("brier","size"),brier=("brier","mean"),
        points_mae=("points_absolute_error","mean"),points_crps=("points_crps","mean"),
        coverage_90=("covered_90","mean")).reset_index()
    summary.to_csv(output/"season_validation_scores.csv",index=False)
    comparisons=[]
    for horizon in HORIZONS:
        data=frame[frame.horizon==horizon]
        keys=["season","team"]
        left=data[data.model=="Dixon-Coles"].set_index(keys).sort_index()
        right=data[data.model=="League Poisson"].set_index(keys).sort_index()
        assert left.index.equals(right.index)
        for metric in ("brier","points_crps","points_absolute_error"):
            interval=cluster_bootstrap_mean((left[metric]-right[metric]).to_numpy(),left.index.get_level_values("season"))
            comparisons.append({"horizon":horizon,"metric":metric,**vars(interval)})
    pd.DataFrame(comparisons).to_csv(output/"season_validation_comparisons.csv",index=False)
    (output/"season_validation_settings.json").write_text(json.dumps({"seed":SEED,
        "simulations":args.simulations,"decay_rate":decay,"horizons":HORIZONS,
        "status":"retrospective exploratory revision; not a new holdout",
        "diagnostics":diagnostics},indent=2))
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
