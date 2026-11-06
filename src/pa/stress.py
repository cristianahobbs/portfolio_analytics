"""Stress tests.

Historical: replay a past crisis window on today's weights (buy-and-hold
over the window).

Hypothetical: shocks are given in market units (equity %, rates bp, spreads
bp...), translated into factor returns, and mapped to assets through their
estimated factor betas. Two variants:
- pure: only the shocked factors move;
- correlated: the other factors move by their expected value given the shock,
  estimated from stressed-period covariances (E[f_o | f_s] = S_os S_ss^-1 f_s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .correlation import to_weekly
from .factors import exposures, factor_returns

HISTORICAL_SCENARIOS = {
    "Global Financial Crisis": ("2008-09-12", "2009-03-09"),
    "US downgrade / Euro crisis": ("2011-07-22", "2011-10-03"),
    "Taper tantrum": ("2013-05-21", "2013-06-24"),
    "China deval / oil crash": ("2015-08-17", "2016-02-11"),
    "Volmageddon / Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "SVB / regional banks": ("2023-03-08", "2023-03-17"),
}

# Units: Equity / USD / Commodity in % moves; Rates / Credit / Inflation in bp.
HYPOTHETICAL_SCENARIOS = {
    "Equities -20%": {"Equity": -0.20},
    "Rates +100bp": {"Rates": 100},
    "Credit spreads +200bp": {"Credit": 200},
    "Stagflation": {"Equity": -0.15, "Rates": 150, "Inflation": 50, "Commodity": 0.20},
    "Risk-off, flight to quality": {"Equity": -0.15, "Rates": -75, "Credit": 150, "USD": 0.05},
    "USD +10%": {"USD": 0.10},
    "Commodity spike +30%": {"Commodity": 0.30, "Inflation": 40},
}


def _factor_durations(cfg: Config) -> dict:
    a = cfg.assets
    f = cfg["factors"]

    def dur(ticker, col, default):
        return float(a.loc[ticker, col]) if ticker in a.index and a.loc[ticker, col] > 0 else default

    rates_t = f["Rates"] if isinstance(f["Rates"], str) else f["Rates"][0]
    credit_t = f["Credit"][0] if isinstance(f["Credit"], list) else f["Credit"]
    return {"Rates": dur(rates_t, "duration", 7.5),
            "Credit": dur(credit_t, "spread_duration", 3.5),
            "Inflation": dur(rates_t, "duration", 7.5)}


def shock_to_factor_returns(cfg: Config, shock: dict) -> pd.Series:
    """Translate market-unit shocks into factor returns."""
    d = _factor_durations(cfg)
    out = {}
    for k, v in shock.items():
        if k == "Rates":
            out[k] = -d["Rates"] * v / 1e4          # yields up -> bond factor down
        elif k == "Credit":
            out[k] = -d["Credit"] * v / 1e4         # spreads wider -> credit factor down
        elif k == "Inflation":
            out[k] = d["Inflation"] * v / 1e4       # breakevens wider -> TIPS beat nominals
        else:
            out[k] = v
    return pd.Series(out, dtype=float)


def historical_stress(asset_returns: pd.DataFrame, weights: pd.Series, nav: float,
                      benchmark: pd.Series | None = None, scenarios=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per scenario, per-asset contribution per scenario)."""
    scenarios = scenarios or HISTORICAL_SCENARIOS
    rows, contrib = {}, {}
    for name, (s, e) in scenarios.items():
        blk = asset_returns.loc[s:e, weights.index]
        if len(blk) < 3 or blk.isna().all().any():
            continue
        asset_ret = (1 + blk.fillna(0)).prod() - 1
        c = weights * asset_ret
        wealth = ((1 + blk.fillna(0)).cumprod() * weights).sum(axis=1)
        path_dd = (wealth / wealth.cummax().clip(lower=1.0) - 1).min()
        row = {"Start": s, "End": e, "Days": len(blk), "Portfolio": c.sum(),
               "P&L ($)": c.sum() * nav, "Worst point": min(path_dd, 0.0)}
        if benchmark is not None:
            row["Benchmark"] = (1 + benchmark.loc[s:e]).prod() - 1
        rows[name] = row
        contrib[name] = c
    return pd.DataFrame(rows).T, pd.DataFrame(contrib)


def stressed_factor_cov(cfg: Config, returns: pd.DataFrame, vix: pd.Series | None,
                        quantile: float = 0.8) -> pd.DataFrame:
    F = to_weekly(factor_returns(cfg, returns))
    if vix is not None:
        v = vix.resample("W-FRI").last().shift(1).reindex(F.index)
        F = F[v > v.quantile(quantile)]
    return F.cov()


def hypothetical_stress(cfg: Config, returns: pd.DataFrame, weights: pd.Series, nav: float,
                        vix: pd.Series | None = None, scenarios=None, betas: pd.DataFrame | None = None):
    """Returns (summary: pure vs correlated P&L, factor moves used, per-asset
    P&L under the correlated version)."""
    scenarios = scenarios or HYPOTHETICAL_SCENARIOS
    if betas is None:
        betas = exposures(cfg, returns, returns[weights.index])
    fnames = list(cfg["factors"].keys())
    B = betas[fnames].astype(float)
    S = stressed_factor_cov(cfg, returns, vix)
    rows, moves, per_asset = {}, {}, {}
    for name, shock in scenarios.items():
        fs = shock_to_factor_returns(cfg, shock)
        pure = pd.Series(0.0, index=fnames)
        pure[fs.index] = fs
        s_idx = list(fs.index)
        o_idx = [f for f in fnames if f not in s_idx]
        corr = pure.copy()
        if o_idx:
            Sss = S.loc[s_idx, s_idx].to_numpy()
            Sos = S.loc[o_idx, s_idx].to_numpy()
            corr[o_idx] = Sos @ np.linalg.solve(Sss, fs.to_numpy())
        a_pure = B @ pure
        a_corr = B @ corr
        rows[name] = {"Pure shock": float(weights @ a_pure),
                      "With correlated moves": float(weights @ a_corr),
                      "P&L ($), correlated": float(weights @ a_corr) * nav}
        moves[name] = corr
        per_asset[name] = weights * a_corr
    return pd.DataFrame(rows).T, pd.DataFrame(moves).T, pd.DataFrame(per_asset)
