"""Currency view for a non-USD investor (e.g. GBP).

fx_rate is quoted as USD per unit of investor currency (e.g. GBPUSD).
Unhedged: (1 + r_usd) * (1 + fx change in investor's favour) - 1.
Hedged: r_usd + hedge carry (the interest differential locked in by rolling
forwards); a hedge ratio h blends the two.
"""
from __future__ import annotations

import pandas as pd


def investor_returns(r_usd: pd.Series, fx_rate: pd.Series, hedge_ratio: float = 0.0,
                     hedge_carry: float = 0.0, periods: int = 252) -> pd.Series:
    fx = fx_rate.reindex(r_usd.index).ffill()
    # a rising GBPUSD means USD assets are worth fewer GBP
    fx_ret = fx.shift(1) / fx - 1
    unhedged = (1 + r_usd) * (1 + fx_ret.fillna(0.0)) - 1
    hedged = r_usd + hedge_carry / periods
    return (hedge_ratio * hedged + (1 - hedge_ratio) * unhedged).rename(f"{hedge_ratio:.0%} hedged")


def hedge_ratio_table(r_usd: pd.Series, fx_rate: pd.Series, hedge_carry: float = 0.0,
                      ratios=(0.0, 0.25, 0.5, 0.75, 1.0), periods: int = 252):
    from .returns import ann_return, ann_vol, max_drawdown

    rows = {}
    for h in ratios:
        r = investor_returns(r_usd, fx_rate, h, hedge_carry, periods)
        rows[f"{h:.0%} hedged"] = {"Annual return": ann_return(r, periods),
                                   "Annual volatility": ann_vol(r, periods),
                                   "Max drawdown": max_drawdown(r)}
    return pd.DataFrame(rows).T
