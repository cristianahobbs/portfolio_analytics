"""Trade proposal: what would it take to move from the current portfolio to a
target, what would it cost, and what changes?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .risk import active_risk


def _clean_target(current: pd.Series, target: pd.Series, min_trade: float, cash: str | None):
    """Skip trades smaller than `min_trade` of NAV (not worth the ticket);
    cash absorbs the difference so weights still sum to one."""
    t = target.reindex(current.index).fillna(0.0).copy()
    small = (t - current).abs() < min_trade
    if cash in t.index:
        small[cash] = False
    t[small] = current[small]
    if cash in t.index:
        t[cash] += 1.0 - t.sum()
    return t


def portfolio_metrics(w: pd.Series, cov_union: pd.DataFrame, w_bench: pd.Series,
                      equity_betas: pd.Series | None = None, mu: pd.Series | None = None,
                      nav: float = 1.0, periods: int = 252) -> dict:
    wu = w.reindex(cov_union.index).fillna(0.0)
    vol = float(np.sqrt(wu @ cov_union @ wu))
    d = vol / np.sqrt(periods)
    z = stats.norm.ppf(0.95)
    ar = active_risk(w, w_bench, cov_union)
    m = {"Volatility": vol,
         "1-day VaR 95% ($)": z * d * nav,
         "1-day ES 97.5% ($)": d * stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025 * nav,
         "Tracking error": ar["tracking_error"],
         "Beta to benchmark": ar["beta_to_benchmark"],
         "Largest position": float(w.max()),
         "Effective N": float(1 / (w ** 2).sum())}
    if equity_betas is not None:
        m["Equity beta"] = float(w @ equity_betas.reindex(w.index).fillna(0.0))
    if mu is not None:
        m["Exp. excess return"] = float(w @ mu.reindex(w.index).fillna(0.0))
        m["Exp. Sharpe"] = m["Exp. excess return"] / vol if vol > 0 else np.nan
    return m


def trade_proposal(current: pd.Series, target: pd.Series, cov_union: pd.DataFrame,
                   w_bench: pd.Series, nav: float, cost_bps: float | pd.Series = 10.0,
                   spread_bps: pd.Series | None = None, min_trade: float = 0.0025,
                   cash: str | None = None, equity_betas: pd.Series | None = None,
                   mu: pd.Series | None = None, names: pd.Series | None = None) -> dict:
    """Returns {'trades': table, 'metrics': before/after table, 'summary': dict}."""
    tgt = _clean_target(current, target, min_trade, cash)
    trade = (tgt - current).where(lambda x: x.abs() > 1e-9, 0.0)
    bps =cost_bps.reindex(current.index) if isinstance(cost_bps, pd.Series) else \
        pd.Series(cost_bps, index=current.index, dtype=float)
    if spread_bps is not None:
        bps = bps + spread_bps.reindex(current.index).fillna(0.0) / 2
    cost = trade.abs() * nav * bps / 1e4
    trades = pd.DataFrame({
        "Current": current, "Target": tgt, "Trade": trade, "Trade ($)": trade * nav,
        "Action": np.where(trade > 1e-9, "BUY", np.where(trade < -1e-9, "SELL", "–")),
        "Est. cost ($)": cost})
    trades.index.name = "Ticker"
    if names is not None:
        trades.insert(0, "Name", names.reindex(current.index))
    trades = trades.reindex(trades["Trade"].abs().sort_values(ascending=False).index)

    before = portfolio_metrics(current, cov_union, w_bench, equity_betas, mu, nav)
    after = portfolio_metrics(tgt, cov_union, w_bench, equity_betas, mu, nav)
    metrics = pd.DataFrame({"Before": before, "After": after})
    metrics["Change"] = metrics["After"] - metrics["Before"]

    total_cost = float(cost.sum())
    summary = {"One-way turnover": float(trade.abs().sum() / 2),
               "Number of trades": int((trade.abs() > 1e-9).sum()),
               "Total cost ($)": total_cost,
               "Cost (bps of NAV)": total_cost / nav * 1e4}
    if mu is not None:
        gain = after["Exp. excess return"] - before["Exp. excess return"]
        summary["Exp. return gain (per year)"] = gain
        summary["Years to recover cost"] = (total_cost / nav) / gain if gain > 0 else np.inf
    return {"trades": trades, "metrics": metrics, "summary": summary, "target": tgt}
