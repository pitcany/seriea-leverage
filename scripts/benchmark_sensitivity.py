"""Closing-source sensitivities on the same repaired goals forecasts."""
from pathlib import Path
import numpy as np
import pandas as pd
from seriea.config import REPORTS_DIR
from seriea.odds.devig import devig_shin
from seriea.evaluation.calibrators import TemperatureScaling
from seriea.evaluation.metrics import ranked_probability_score
from seriea.evaluation.inference import cluster_bootstrap_mean

def main():
    out=Path(REPORTS_DIR);frame=pd.read_parquet(out/"forecasts.parquet")
    y=frame.outcome.to_numpy();test=frame.is_test.to_numpy(dtype=bool)
    goals=frame[["p_H","p_D","p_A"]].to_numpy()
    goals=TemperatureScaling().fit(goals[~test],y[~test]).transform(goals)
    market=frame[["p_market_H","p_market_D","p_market_A"]].to_numpy()
    alt=market.copy();late=frame.season=="2526"
    alt[late]=devig_shin(frame.loc[late,["avgc_h","avgc_d","avgc_a"]].to_numpy(dtype=float))
    rows=[]
    for name,prices,selected in (("Primary closing policy",market,test),
        ("Average closing in 2025/26",alt,test),("Exclude 2025/26",market,test&~late.to_numpy())):
        m=ranked_probability_score(prices[selected],y[selected])
        g=ranked_probability_score(goals[selected],y[selected])
        interval=cluster_bootstrap_mean(g-m,frame.loc[selected,"season"].to_numpy())
        rows.append({"benchmark":name,"n":int(selected.sum()),"market_rps":float(m.mean()),
                     "goals_rps":float(g.mean()),**vars(interval)})
    result=pd.DataFrame(rows);result.to_csv(out/"benchmark_sensitivity.csv",index=False)
    print(result.to_string(index=False))
if __name__=="__main__":main()
