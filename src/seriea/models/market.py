"""The betting market as a forecaster.

This is the benchmark that matters. A closing price aggregates the money and
opinion of every participant, and in liquid markets it is very hard to beat. The
2016 study fed these prices in as *features* and then compared its accuracy
favourably against the market's — which inverts the logic. If a model consumes
the market's own forecast and scores worse than it, the model is destroying
information rather than adding any.

Pinnacle closing prices are preferred where available (2012-13 onward) because
Pinnacle operates on low margins and high limits, making its close the sharpest
widely-published number. Bet365 opening prices are the fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seriea.odds.devig import devig_shin

#: Column triples in order of preference: Pinnacle closing, Pinnacle opening,
#: then Bet365 opening.
_PRICE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("psc_h", "psc_d", "psc_a"),
    ("ps_h", "ps_d", "ps_a"),
    ("b365_h", "b365_d", "b365_a"),
)


def market_probabilities(fixtures: pd.DataFrame) -> np.ndarray:
    """Derive margin-free market probabilities for each fixture.

    For every fixture the best available price source is used, falling back
    through Pinnacle closing, Pinnacle opening and Bet365 opening. Fixtures with
    no complete book yield a row of NaN.

    Args:
        fixtures: Canonical match frame carrying the price columns.

    Returns:
        Array of shape ``(len(fixtures), 3)`` in H-D-A column order.
    """
    result = np.full((len(fixtures), 3), np.nan)

    for columns in _PRICE_SOURCES:
        if not all(column in fixtures.columns for column in columns):
            continue

        unresolved = np.isnan(result).any(axis=1)
        if not unresolved.any():
            break

        prices = fixtures.loc[:, list(columns)].to_numpy(dtype=float, na_value=np.nan)
        candidate = devig_shin(prices)

        usable = unresolved & np.isfinite(candidate).all(axis=1)
        result[usable] = candidate[usable]

    return result


class MarketForecaster:
    """Passes through de-vigged market prices as the forecast.

    Stateless: the market needs no fitting. :meth:`fit` exists only to satisfy
    the :class:`~seriea.models.base.Forecaster` interface.
    """

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp) -> "MarketForecaster":
        """Accept the interface and learn nothing.

        Args:
            history: Ignored.
            as_of: Ignored.

        Returns:
            The forecaster, unchanged.
        """
        return self

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Return margin-free market probabilities.

        Args:
            fixtures: Canonical match frame carrying price columns.

        Returns:
            Array of shape ``(len(fixtures), 3)``. Rows without a complete book
            are NaN and must be excluded by the caller.
        """
        return market_probabilities(fixtures)


def has_market_price(fixtures: pd.DataFrame) -> np.ndarray:
    """Flag fixtures for which a complete book exists.

    Args:
        fixtures: Canonical match frame.

    Returns:
        Boolean array of shape ``(len(fixtures),)``.
    """
    return np.isfinite(market_probabilities(fixtures)).all(axis=1)
