"""Cross-check saved scientific outputs and manuscript numbers against forecasts."""
from pathlib import Path
import hashlib,json
from statistics import NormalDist
import numpy as np
import pandas as pd
from seriea.config import REPORTS_DIR
from seriea.data.load import load_all
from seriea.evaluation.metrics import ranked_probability_score,log_loss,brier_score

def main():
    out=Path(REPORTS_DIR)
    data=load_all()
    assert len(data)==7220
    assert not data.duplicated(["season","home","away"]).any()
    assert data[["home_sot","away_sot"]].isna().any(axis=1).sum()==1
    diagnostics={}
    for name in ("forecasts","shots_forecasts"):
        frame=pd.read_parquet(out/f"{name}.parquet")
        assert len(frame)==4940
        assert frame.origin.nunique()==262
        assert frame.fit_converged.all()
        assert np.isfinite(frame.minimum_tau).all() and (frame.minimum_tau>0).all()
        assert (frame.origin<=frame.date).all()
        p=frame[["p_H","p_D","p_A"]].to_numpy()
        assert (p>=0).all() and np.allclose(p.sum(axis=1),1)
        diagnostics[name]={"origins":int(frame.origin.nunique()),"all_converged":True,
                          "minimum_forecast_tau":float(frame.minimum_tau.min())}
    frame=pd.read_parquet(out/"revised_forecasts.parquet")
    test=frame[frame.is_test]
    assert len(test)==2660
    assert (frame.market_timing=="closing").all()
    assert (frame[frame.season=="2526"].market_source=="b365c").all()
    scores=pd.read_csv(out/"revised_scores.csv")
    for row in scores.itertuples():
        p=test[[f"{row.model}::{o}" for o in ("H","D","A")]].to_numpy()
        for column,fn in (("rps",ranked_probability_score),("log_loss",log_loss),("brier",brier_score)):
            assert np.isclose(getattr(row,column),fn(p,test.outcome.to_numpy()).mean(),atol=1e-12,rtol=0)
    seasons=pd.read_csv(out/"season_validation_forecasts.csv")
    assert len(seasons)==840
    assert not seasons.duplicated(["season","horizon","team","model"]).any()
    assert seasons.survival_probability.between(0,1).all()
    assert (seasons.groupby(["season","horizon","model"]).survived.sum()==17).all()
    assert (seasons.groupby(["season","horizon","model"]).size()==20).all()
    leverage=pd.read_csv(out/"fiorentina_leverage.csv")
    assert ((leverage.n_win+leverage.n_draw+leverage.n_loss)==20000).all()
    assert np.allclose(leverage.leverage,leverage.p_objective_if_win-leverage.p_objective_if_loss)
    assert len(leverage)==12
    stakes=pd.read_csv(out/"fiorentina_stakes.csv")
    assert len(stakes)==12
    keys=["date","opponent","venue"]
    raw=leverage.set_index(keys).sort_index()
    derived=stakes.set_index(keys).sort_index()
    assert raw.index.equals(derived.index)
    q=raw[["p_objective_if_win","p_objective_if_draw","p_objective_if_loss"]].to_numpy()
    counts=raw[["n_win","n_draw","n_loss"]].to_numpy()
    v=q*(1-q)/counts
    u=q[:,0]-q[:,1];a=q[:,1]-q[:,2]
    assert np.allclose(derived.win_over_draw,u,atol=1e-12,rtol=0)
    assert np.allclose(derived.draw_over_loss,a,atol=1e-12,rtol=0)
    assert np.allclose(derived.component_mc_covariance,-v[:,1],atol=1e-12,rtol=0)
    assert np.allclose(derived.win_over_draw_mc_se**2,v[:,0]+v[:,1],atol=1e-12,rtol=0)
    assert np.allclose(derived.draw_over_loss_mc_se**2,v[:,1]+v[:,2],atol=1e-12,rtol=0)
    assert derived.scenario_feasible.all()
    assert (counts[:,1]/counts.sum(axis=1)>=.1).all()
    assert np.allclose(derived.scenario_delta_expected_points,.05,atol=1e-12,rtol=0)
    assert np.allclose(derived.scenario_delta_survival,.05*(u-a),atol=1e-12,rtol=0)
    assert np.allclose(derived.scenario_delta_survival_mc_se**2,
                       .05**2*(v[:,0]+4*v[:,1]+v[:,2]),atol=1e-12,rtol=0)
    assert (derived.ratio_mc_ci_status=="bounded").all()
    z2=NormalDist().inv_cdf(.975)**2
    for j,row in enumerate(derived.itertuples()):
        vu=v[j,0]+v[j,1];va=v[j,1]+v[j,2];cov=-v[j,1]
        roots=np.sort(np.roots([u[j]**2-z2*vu,-2*(a[j]*u[j]-z2*cov),a[j]**2-z2*va]))
        assert np.allclose([row.ratio_mc_ci_low,row.ratio_mc_ci_high],roots,atol=1e-12,rtol=0)
    settings=json.loads((out/"fiorentina_stakes_settings.json").read_text())
    assert settings["input_sha256"]==hashlib.sha256((out/"fiorentina_leverage.csv").read_bytes()).hexdigest()
    membership=pd.read_csv(out/"survival_subgroup_membership.csv",dtype={"season":str})
    subgroup_scores=pd.read_csv(out/"survival_subgroup_scores.csv")
    subgroup_comparisons=pd.read_csv(out/"survival_subgroup_comparisons.csv")
    keys=["season","horizon","as_of","team"]
    assert len(membership)==420 and not membership.duplicated(keys).any()
    assert (membership.groupby(["season","horizon"]).bottom8.sum()==8).all()
    assert (membership.groupby(["season","horizon"]).size()==20).all()
    seasons["season"]=seasons.season.astype(str)
    joined=seasons.merge(membership[keys+["subgroup"]],on=keys,validate="many_to_one")
    assert len(joined)==len(seasons)
    for row in subgroup_scores.itertuples():
        selected=joined[(joined.horizon==row.horizon)&(joined.subgroup==row.subgroup)&(joined.model==row.model)]
        assert len(selected)==row.n==({"bottom8":56,"other12":84}[row.subgroup])
        for name,original in (("brier","brier"),("points_mae","points_absolute_error"),("points_crps","points_crps"),("coverage_90","covered_90")):
            assert np.isclose(getattr(row,name),selected[original].mean(),atol=1e-12,rtol=0)
    for row in subgroup_comparisons.itertuples():
        selected=joined[(joined.horizon==row.horizon)&(joined.subgroup==row.subgroup)]
        paired=selected.pivot(index=keys,columns="model",values=row.metric)
        assert len(paired)==row.n and selected.season.nunique()==row.n_seasons==7
        assert np.isclose(row.estimate,(paired["Dixon-Coles"]-paired["League Poisson"]).mean(),atol=1e-12,rtol=0)
    subgroup_settings=json.loads((out/"survival_subgroup_settings.json").read_text())
    assert subgroup_settings["forecast_sha256"]==hashlib.sha256((out/"season_validation_forecasts.csv").read_bytes()).hexdigest()
    abstract=json.loads((out/"abstract_metadata.json").read_text())
    assert abstract["words_including_title_and_headings"]<500
    result={"status":"passed","corpus_matches":len(data),"test_matches":len(test),
        "season_forecasts":len(seasons),"fiorentina_fixtures":len(leverage),
        "abstract_words":abstract["words_including_title_and_headings"],"fits":diagnostics,
        "fixture_stakes_checks":"all 12 components, shared-draw covariance, Fieller endpoints, scenario effects and simulation variances independently recomputed; input hash verified",
        "survival_subgroup_checks":"420 exhaustive membership rows, 840 joined forecasts, all subgroup scores and paired differences checked; source forecast hash verified",
        "score_checks":"all revised score-table entries recomputed from saved test probabilities"}
    (out/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
