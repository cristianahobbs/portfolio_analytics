"""Liquidity: how long would it take to exit, and what does it cost?

Days to liquidate = position / (participation rate x ADV). Under stress, ADV
is cut and spreads widen. Liquidity-adjusted VaR adds the cost of exiting
(half the bid-ask spread plus square-root market impact) to market VaR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def liquidity_table(weights: pd.Series, nav: float, adv_usd: pd.Series, spread_bps: pd.Series,
                    daily_vol: pd.Series, participation: float = 0.2, stress_adv_haircut: float = 0.5,
                    stress_spread_mult: float = 3.0, impact_coef: float = 1.0) -> pd.DataFrame:
    pos = (weights * nav).abs()
    adv = adv_usd.reindex(weights.index)
    sp = spread_bps.reindex(weights.index) / 1e4
    days = pos / (participation * adv)
    days_s = pos / (participation * adv * (1 - stress_adv_haircut))
    # square-root impact: cost ~ k * sigma_daily * sqrt(order size / ADV)
    impact = impact_coef * daily_vol.reindex(weights.index) * np.sqrt(pos / adv)
    impact_s = impact_coef * daily_vol.reindex(weights.index) * np.sqrt(
        pos / (adv * (1 - stress_adv_haircut)))
    out = pd.DataFrame({
        "Position ($)": pos,
        "ADV ($)": adv,
        "% of ADV": pos / adv,
        "Days to liquidate": days,
        "Days (stressed)": days_s,
        "Exit cost": sp / 2 + impact,
        "Exit cost (stressed)": sp * stress_spread_mult / 2 + impact_s,
    })
    return out


def liquidity_profile(table: pd.DataFrame, weights: pd.Series,
                      buckets=(1, 2, 5, 10, 20), stressed: bool = False) -> pd.Series:
    """Share of the portfolio that can be sold within each horizon, selling
    each position at the participation rate in parallel."""
    days = table["Days (stressed)" if stressed else "Days to liquidate"]
    w = weights.abs() / weights.abs().sum()
    return pd.Series({f"Within {b} day{'s' if b > 1 else ''}": float((w * np.minimum(1, b / days)).sum())
                      for b in buckets})


def days_to_liquidate_fraction(table: pd.DataFrame, weights: pd.Series, fraction: float = 0.9,
                               stressed: bool = False) -> float:
    days = table["Days (stressed)" if stressed else "Days to liquidate"].fillna(np.inf)
    w = weights.abs() / weights.abs().sum()
    grid = np.unique(np.concatenate([[0.0], days.to_numpy()]))
    for d in grid:
        if (w * np.minimum(1, d / days)).sum() >= fraction - 1e-12:
            return float(d)
    return float("inf")


def liquidity_adjusted_var(var: float, table: pd.DataFrame, weights: pd.Series,
                           stressed: bool = False) -> dict:
    col = "Exit cost (stressed)" if stressed else "Exit cost"
    cost = float((weights.abs() * table[col]).sum())
    return {"Market VaR": var, "Exit cost": cost, "Liquidity-adjusted VaR": var + cost}
