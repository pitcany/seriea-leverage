"""Additional retrospective analyses, using saved validation and test forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from seriea.config import REPORTS_DIR
from seriea.evaluation.metrics import ranked_probability_score,log_loss,brier_score,accuracy,outcomes_to_indicator
from seriea.evaluation.inference import bootstrap_mean,cluster_bootstrap_mean
from seriea.evaluation.calibrators import TemperatureScaling
from seriea.evaluation.calibration import calibration_slope_intercept,multiclass_calibration_error
from seriea.evaluation.encompassing import ResidualLogitCalibrator
from seriea.models.blend import fit_pool_weights,logarithmic_pool_many

def main():
    out=Path(REPORTS_DIR)
    goals_frame=pd.read_parquet(out/"forecasts.parquet")
    shots_frame=pd.read_parquet(out/"shots_forecasts.parquet")
    keys=["date","home","away"]
    frame=goals_frame.merge(shots_frame[keys+["p_H","p_D","p_A"]],on=keys,
        suffixes=("","_shots"),validate="one_to_one")
    assert len(frame)==len(goals_frame)
    cols=["H","D","A"]
    market=frame[[f"p_market_{c}" for c in cols]].to_numpy()
    goals=frame[[f"p_{c}" for c in cols]].to_numpy()
    shots=frame[[f"p_{c}_shots" for c in cols]].to_numpy()
    y=frame.outcome.to_numpy();test=frame.is_test.to_numpy(dtype=bool);validation=~test
    calibrated=TemperatureScaling().fit(goals[validation],y[validation]).transform(goals)
    weights=fit_pool_weights([goals[validation],shots[validation]],y[validation])
    market_fit=ResidualLogitCalibrator().fit(market[validation],y[validation])
    augmented_fit=ResidualLogitCalibrator().fit(market[validation],y[validation],
        [goals[validation],shots[validation]])
    candidates={"Goals raw":goals,"Goals calibrated":calibrated,"Shots":shots,
        "Goals + shots":logarithmic_pool_many([goals,shots],weights),"Market":market,
        "Market recalibrated":market_fit.transform(market),
        "Market + structural logits":augmented_fit.transform(market,[goals,shots])}
    settings={"market_only_coefficients":market_fit.coefficients.tolist(),
        "augmented_coefficients":augmented_fit.coefficients.tolist(),
        "coefficient_order":["intercept_H","intercept_D","market","goals","shots"],
        "goals_shots_weights":weights.tolist(),"test_status":"historical, inspected before revision",
        "n_validation":int(validation.sum()),"n_test":int(test.sum())}
    (out/"revised_analysis_settings.json").write_text(json.dumps(settings,indent=2))
    metric_functions={"rps":ranked_probability_score,"log_loss":log_loss,"brier":brier_score}
    scores=[];comparisons=[];calibration=[];reliability=[];per_season=[]
    indicator=outcomes_to_indicator(y[test]);seasons=frame.loc[test,"season"].to_numpy()
    for name,probabilities in candidates.items():
        p=probabilities[test]
        record={"model":name,"n":len(p),"accuracy":accuracy(p,y[test]),
            "macro_auc":roc_auc_score(indicator,p,average="macro"),
            "entropy":float(-np.sum(p*np.log(p),axis=1).mean())}
        for metric,fn in metric_functions.items():
            values=fn(p,y[test]);record[metric]=float(values.mean())
            references=["Market"]
            if name=="Goals + shots": references.append("Goals raw")
            if name=="Market + structural logits": references.append("Market recalibrated")
            for reference in references:
                diff=values-fn(candidates[reference][test],y[test])
                for method,interval in (("match",bootstrap_mean(diff)),
                    ("season",cluster_bootstrap_mean(diff,seasons))):
                    comparisons.append({"model":name,"reference":reference,"metric":metric,
                        "bootstrap":method,**vars(interval)})
        scores.append(record)
        for season in np.unique(seasons):
            selected=seasons==season
            per_season.append({"season":season,"model":name,"n":int(selected.sum()),
                "rps":float(ranked_probability_score(p[selected],y[test][selected]).mean())})
        ece=multiclass_calibration_error(p,y[test])
        for j,outcome in enumerate(cols):
            fit=calibration_slope_intercept(p[:,j],indicator[:,j])
            x=np.log(np.clip(p[:,j],1e-9,1-1e-9)/(1-np.clip(p[:,j],1e-9,1-1e-9)))
            design=np.column_stack([np.ones(len(p)),x])
            fitted=expit(fit.intercept+fit.slope*x)
            information=design.T@((fitted*(1-fitted))[:,None]*design)
            se=np.sqrt(np.diag(np.linalg.pinv(information)))
            calibration.append({"model":name,"outcome":outcome,"ece":ece[outcome],
                "intercept":fit.intercept,"intercept_lower":fit.intercept-1.96*se[0],
                "intercept_upper":fit.intercept+1.96*se[0],"slope":fit.slope,
                "slope_lower":fit.slope-1.96*se[1],"slope_upper":fit.slope+1.96*se[1],
                "interval_type":"iid Fisher descriptive; not cluster adjusted"})
            bin_id=np.minimum((p[:,j]*10).astype(int),9)
            for b in range(10):
                mask=bin_id==b;n=int(mask.sum())
                if n==0:continue
                observed=float(indicator[mask,j].mean());z=1.96
                center=(observed+z*z/(2*n))/(1+z*z/n)
                half=z*np.sqrt(observed*(1-observed)/n+z*z/(4*n*n))/(1+z*z/n)
                reliability.append({"model":name,"outcome":outcome,"bin":b,"n":n,
                    "mean_forecast":float(p[mask,j].mean()),"observed":observed,
                    "lower":center-half,"upper":center+half})
        for j,o in enumerate(cols):
            frame[f"{name}::{o}"]=probabilities[:,j]
    pd.DataFrame(scores).to_csv(out/"revised_scores.csv",index=False)
    pd.DataFrame(comparisons).to_csv(out/"revised_comparisons.csv",index=False)
    pd.DataFrame(calibration).to_csv(out/"revised_calibration.csv",index=False)
    pd.DataFrame(reliability).to_csv(out/"revised_reliability.csv",index=False)
    pd.DataFrame(per_season).to_csv(out/"revised_season_rps.csv",index=False)
    frame.to_parquet(out/"revised_forecasts.parquet",index=False)
    frame.groupby(["season","market_source","market_timing"]).size().rename("n").to_csv(out/"market_provenance.csv")
    print(pd.DataFrame(scores).to_string(index=False))
    print(json.dumps(settings,indent=2))

if __name__=="__main__":
    main()
