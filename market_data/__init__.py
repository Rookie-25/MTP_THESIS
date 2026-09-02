"""Market data acquisition and curve construction for the XVA engine.

Deliberately layered so that everything mathematical is importable and
testable without a network connection or the optional data libraries:

* :class:`~market_data.fetcher.YieldCurve`,
  :class:`~market_data.fetcher.CreditCurve` and
  :func:`~market_data.fetcher.bootstrap_hazard_rates` are pure NumPy;
* the ``fetch_*`` functions are thin I/O wrappers that import ``yfinance`` and
  ``pandas_datareader`` lazily and return those same pure objects.

Nothing here reaches the network at import time.
"""

from market_data.fetcher import (
    CDSQuote,
    CreditCurve,
    VolSurfaceData,
    YieldCurve,
    bootstrap_hazard_rates,
    clean_option_chain,
    fetch_discount_curve,
    fetch_implied_vol_surface,
    fetch_sofr_curve,
    fetch_spot,
    fetch_treasury_curve,
    model_par_spread,
)

__all__ = [
    "CDSQuote",
    "CreditCurve",
    "VolSurfaceData",
    "YieldCurve",
    "bootstrap_hazard_rates",
    "clean_option_chain",
    "fetch_discount_curve",
    "fetch_implied_vol_surface",
    "fetch_sofr_curve",
    "fetch_spot",
    "fetch_treasury_curve",
    "model_par_spread",
]
