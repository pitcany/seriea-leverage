"""A scoped comparison of market recalibration with added structural log odds.

Both variants estimate home/draw intercepts and one market log-odds coefficient.
The augmented variant adds one coefficient per structural forecast. Fits use
validation only; improvements on already-inspected historical test data are
exploratory. This is not a nonparametric test of informational completeness.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from seriea.evaluation.metrics import outcomes_to_indicator

class ResidualLogitCalibrator:
    def _features(self, market, signals):
        arrays=[np.asarray(market),*[np.asarray(s) for s in signals]]
        if any(a.shape!=arrays[0].shape for a in arrays):
            raise ValueError("Forecast arrays must align.")
        features=[]
        for array in arrays:
            lp=np.log(np.clip(array,1e-12,1))
            features.append(lp[:,:2]-lp[:,2,None])
        return np.stack(features,axis=2)

    def fit(self, market, outcomes, signals=()):
        x=self._features(market,signals)
        y=outcomes_to_indicator(outcomes)
        if len(y)==0 or y.shape!=market.shape:
            raise ValueError("Nonempty aligned forecasts and outcomes are required.")
        def objective(theta):
            logits=np.column_stack([theta[:2]+np.einsum("nck,k->nc",x,theta[2:]),np.zeros(len(x))])
            residual=softmax(logits,axis=1)-y
            loss=np.mean(logsumexp(logits,axis=1)-(y*logits).sum(axis=1))
            gradient=np.r_[residual[:,:2].mean(axis=0),
                np.einsum("nc,nck->k",residual[:,:2],x)/len(x)]
            return loss,gradient
        initial=np.r_[0.,0.,1.,np.zeros(len(signals))]
        result=minimize(objective,initial,jac=True,method="L-BFGS-B",
            bounds=[(-3,3),(-3,3),(0,5)]+[(-2,2)]*len(signals),
            options={"ftol":1e-12,"gtol":1e-9,"maxiter":1000})
        if not result.success:
            raise RuntimeError("Calibration fit failed: "+str(result.message))
        self.coefficients=result.x
        self.n_signals=len(signals)
        return self

    def transform(self, market, signals=()):
        if not hasattr(self,"coefficients"):
            raise RuntimeError("Fit the calibrator before transforming.")
        if len(signals)!=self.n_signals:
            raise ValueError("Signal count differs from fitting.")
        x=self._features(market,signals)
        logits=np.column_stack([self.coefficients[:2]+np.einsum("nck,k->nc",x,self.coefficients[2:]),
                                np.zeros(len(x))])
        return softmax(logits,axis=1)
