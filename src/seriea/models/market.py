"""Closing-price benchmark with explicit per-fixture provenance.

Use Pinnacle closing before the provider's 23 July 2025 stale-price warning,
then Bet365 closing. Average closing is an explicitly labelled fallback.
Pre-closing snapshots are never silently substituted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from seriea.odds.devig import devig_shin

PINNACLE_CUTOFF = pd.Timestamp("2025-07-23")

def market_forecasts(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return H/D/A probabilities, source and timing; missing books remain NaN."""
    probabilities = np.full((len(fixtures), 3), np.nan)
    sources = np.full(len(fixtures), "missing", dtype=object)
    for source in ("psc", "b365c", "avgc"):
        columns = [f"{source}_{o}" for o in ("h", "d", "a")]
        if not all(c in fixtures for c in columns):
            continue
        eligible = np.isnan(probabilities).any(axis=1)
        if source == "psc":
            # Without dates we cannot establish that Pinnacle is eligible.
            if "date" not in fixtures:
                continue
            eligible &= (pd.to_datetime(fixtures["date"]) < PINNACLE_CUTOFF).to_numpy()
        prices = fixtures[columns].to_numpy(dtype=float, na_value=np.nan)
        candidate = devig_shin(prices)
        use = eligible & np.isfinite(candidate).all(axis=1)
        probabilities[use] = candidate[use]
        sources[use] = source
    output = pd.DataFrame(probabilities, columns=["p_market_H", "p_market_D", "p_market_A"],
                          index=fixtures.index)
    output["market_source"] = sources
    output["market_timing"] = np.where(sources == "missing", "missing", "closing")
    return output

def market_probabilities(fixtures: pd.DataFrame) -> np.ndarray:
    """Shin de-vigged probabilities under the documented closing-only policy."""
    return market_forecasts(fixtures).iloc[:, :3].to_numpy()

class MarketForecaster:
    """Stateless closing-price benchmark."""
    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "MarketForecaster":
        return self
    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        return market_probabilities(fixtures)

def has_market_price(fixtures: pd.DataFrame) -> np.ndarray:
    return np.isfinite(market_probabilities(fixtures)).all(axis=1)
