"""Regression checks for scientifically consequential defects found in review."""
import numpy as np
import pandas as pd
import pytest
from seriea.models.dixon_coles import DixonColesForecaster
from seriea.models.shots import ShotsForecaster
from seriea.models.market import market_forecasts
from seriea.models.blend import MarketModelPool
from test_models import synthetic_matches

def test_unobserved_low_score_support_is_checked():
    model = DixonColesForecaster()
    # lambda=mu=4, observed 2-2: negative 0-1 and 1-0 probabilities still matter.
    p = np.array([0., -np.log(4), -np.log(4), 0., -.5])
    loss = model._negative_log_likelihood(p, np.array([0]), np.array([1]),
        np.array([2]), np.array([2]), np.array([1.]), 2)
    assert loss > 1_000_000

def test_prediction_rejects_invalid_support_instead_of_clipping():
    data = synthetic_matches(4)
    model = DixonColesForecaster().fit(data, data.date.max()+pd.Timedelta(days=1))
    model._rho = -100
    with pytest.raises(ValueError, match="support"):
        model.predict_score_grid(data.head(1))

def test_shots_fit_is_poisson_and_unseen_rates_use_shared_fallback():
    data = synthetic_matches(10)
    data["home_sot"] = 3*data.home_goals+2
    data["away_sot"] = 3*data.away_goals+2
    model = ShotsForecaster().fit(data, data.date.max()+pd.Timedelta(days=1))
    assert model._shot_model.rho == 0
    fixtures = pd.DataFrame({"home":["Fiorentina","Unseen"],"away":["Unseen","Fiorentina"]})
    shot_h,shot_a = model._shot_model.predict_rates(fixtures)
    h,a = model._goal_rates(fixtures)
    assert np.allclose(h,shot_h*model._home_conversion)
    assert np.allclose(a,shot_a*model._away_conversion)
    ratings = model.shot_ratings()
    assert h[0] == pytest.approx(np.exp(model._home_advantage+ratings.loc["Fiorentina","attack"]
        -model._shot_model._mean_defence)*model._home_conversion)
    assert a[1] == pytest.approx(np.exp(ratings.loc["Fiorentina","attack"]
        -model._shot_model._mean_defence)*model._away_conversion)

def test_closing_policy_rejects_stale_and_preclosing():
    data = pd.DataFrame({"date":pd.to_datetime(["2025-07-22","2025-07-23","2026-01-01"]),
        "psc_h":[2.,2.,2.],"psc_d":[3.,3.,3.],"psc_a":[4.,4.,4.],
        "b365c_h":[1.8,1.8,np.nan],"b365c_d":[3.5,3.5,np.nan],"b365c_a":[4.,4.,np.nan],
        "b365_h":[2.,2.,2.],"b365_d":[3.,3.,3.],"b365_a":[4.,4.,4.]})
    forecasts=market_forecasts(data)
    assert forecasts.market_source.tolist()==["psc","b365c","missing"]
    assert forecasts.iloc[2,:3].isna().all()

def test_pool_can_choose_exact_endpoint():
    market=np.tile([.9,.05,.05],(20,1))
    model=np.tile([.1,.45,.45],(20,1))
    assert MarketModelPool().fit(market,model,np.array(["H"]*20)).weight==0.

def test_cluster_bootstrap_keeps_within_season_dependence():
    from seriea.evaluation.inference import cluster_bootstrap_mean,bootstrap_mean
    values=np.repeat([0.,1.],100)
    groups=np.repeat([2019,2020],100)
    block=cluster_bootstrap_mean(values,groups,resamples=1000)
    naive=bootstrap_mean(values,resamples=1000)
    assert block.lower==0. and block.upper==1.
    assert naive.lower>0. and naive.upper<1.

def test_nested_calibrator_recovers_simple_validation_signal():
    from seriea.evaluation.encompassing import ResidualLogitCalibrator
    market=np.tile([1/3]*3,(90,1))
    y=np.tile(["H","D","A"],30)
    signals=np.tile(np.eye(3)*.7+.1,(30,1))
    base=ResidualLogitCalibrator().fit(market,y)
    augmented=ResidualLogitCalibrator().fit(market,y,[signals])
    from seriea.evaluation.metrics import log_loss
    assert log_loss(augmented.transform(market,[signals]),y).mean()<log_loss(base.transform(market),y).mean()
    assert np.allclose(augmented.transform(market,[signals]).sum(axis=1),1.)
