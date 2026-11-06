"""Walk-forward backtest with drifting weights, rebalancing rules and costs.

Timing (no look-ahead): on decision date d the strategy sees returns up to
and including d's close, trades at d's close, and the new weights first earn
returns on d+1. Transaction costs are charged on d+1.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import optimise as opt
from . import returns as R
from .config import Config
from .risk import ledoit_wolf_cov

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series                 # net of costs
    gross_returns: pd.Series
    weights: pd.DataFrame              # end-of-day weights after any trade
    turnover: pd.Series                # one-way turnover on each trade date
    costs: pd.Series                   # daily cost drag
    decisions: list = field(default_factory=list)

    def summary(self, rf=0.0, benchmark=None, periods: int = 252) -> pd.Series:
        s = R.performance_summary(self.returns, rf, benchmark, periods)
        years = len(self.returns) / periods
        s["Annual turnover"] = self.turnover.sum() / years
        s["Annual cost drag"] = self.costs.sum() / years
        s["Trades"] = int((self.turnover > 0).sum())
        return s


def walk_forward(asset_returns: pd.DataFrame, strategy: Strategy, lookback: int = 756,
                 rebalance: str = "ME", cost_bps: float | pd.Series = 10.0,
                 threshold: float = 0.0, name: str = "strategy") -> BacktestResult:
    rets = asset_returns.fillna(0.0)
    dates, cols = rets.index, rets.columns
    X = rets.to_numpy()
    decisions = set(d for d in R.period_ends(dates, rebalance) if dates.get_loc(d) >= lookback - 1)
    if not decisions:
        raise ValueError("Not enough history for the lookback window")
    bps = (cost_bps.reindex(cols).to_numpy() if isinstance(cost_bps, pd.Series)
           else np.full(len(cols), cost_bps)) / 1e4

    n = len(dates)
    gross = np.full(n, np.nan)
    cost = np.zeros(n)
    W = np.full((n, len(cols)), np.nan)
    turnover = {}
    w = None
    for i, dt in enumerate(dates):
        if w is not None:
            port = w @ X[i]
            gross[i] = port
            w = w * (1 + X[i]) / (1 + port)
        if dt in decisions:
            hist = rets.iloc[i - lookback + 1: i + 1]
            target = strategy(hist).reindex(cols).fillna(0.0).to_numpy()
            current = np.zeros(len(cols)) if w is None else w
            trade = np.abs(target - current)
            if w is None or threshold <= 0 or trade.max() > threshold:
                turnover[dt] = trade.sum()   # includes the initial build
                if i + 1 < n:
                    cost[i + 1] += trade @ bps
                w = target
        if w is not None:
            W[i] = w

    first = min(decisions)
    keep = dates > first
    gross_s = pd.Series(gross, index=dates)[keep]
    cost_s = pd.Series(cost, index=dates)[keep]
    return BacktestResult(
        name=name,
        returns=(gross_s - cost_s).rename(name),
        gross_returns=gross_s.rename(name),
        weights=pd.DataFrame(W, index=dates, columns=cols)[dates >= first],
        turnover=pd.Series(turnover, dtype=float).rename("Turnover"),
        costs=cost_s.rename("Cost"),
        decisions=sorted(decisions),
    )


# --------------------------------------------------------------------------
# Strategy factories
# --------------------------------------------------------------------------
def static_strategy(weights: pd.Series) -> Strategy:
    return lambda hist: weights


def optimised_strategy(cfg: Config, method: str, mu_method: str | None = None) -> Strategy:
    """Re-estimates covariance (Ledoit-Wolf) and expected returns from the
    lookback window only, then optimises under the config constraints."""
    cons = opt.Constraints.from_config(cfg)
    cash = cfg.cash_ticker
    state = {"x0": cfg.weights.reindex(cons.tickers).to_numpy()}

    def strat(hist: pd.DataFrame) -> pd.Series:
        hist = hist[cons.tickers]
        excess = hist.sub(hist[cash], axis=0) if cash in hist else hist
        cov = ledoit_wolf_cov(hist, cfg.trading_days)
        mu = opt.expected_returns(cfg, excess, cov, mu_method) if method == "max_sharpe" else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = opt.optimise(method, mu, cov, cons, state["x0"])
        state["x0"] = w.to_numpy()
        return w

    return strat


def run_strategies(cfg: Config, asset_returns: pd.DataFrame, methods=opt.METHODS,
                   include_current: bool = True, **kw) -> dict[str, BacktestResult]:
    bt = cfg["backtest"]
    params = dict(lookback=bt["lookback_days"], rebalance=bt["rebalance"],
                  cost_bps=bt["cost_bps"], threshold=bt.get("threshold", 0.0))
    params.update(kw)
    rets = asset_returns[cfg.tickers]
    out = {}
    if include_current:
        out["current (static)"] = walk_forward(rets, static_strategy(cfg.weights),
                                               name="current (static)", **params)
    for m in methods:
        out[m] = walk_forward(rets, optimised_strategy(cfg, m), name=m, **params)
    return out


def cost_sensitivity(cfg: Config, asset_returns: pd.DataFrame, strategy: Strategy,
                     costs_bps=(0, 5, 10, 25, 50), rf=0.0) -> pd.DataFrame:
    bt = cfg["backtest"]
    rows = {}
    for c in costs_bps:
        res = walk_forward(asset_returns[cfg.tickers], strategy, bt["lookback_days"],
                           bt["rebalance"], c, bt.get("threshold", 0.0))
        s = res.summary(rf, periods=cfg.trading_days)
        rows[f"{c} bps"] = s[["Annual return", "Sharpe", "Annual turnover", "Annual cost drag"]]
    return pd.DataFrame(rows).T


def rebalance_comparison(cfg: Config, asset_returns: pd.DataFrame, weights: pd.Series,
                         rf=0.0) -> pd.DataFrame:
    """Calendar vs threshold rebalancing of a fixed target, after costs."""
    bt = cfg["backtest"]
    rets = asset_returns[cfg.tickers]
    rules = {"Monthly": ("ME", 0.0), "Quarterly": ("QE", 0.0), "Annual": ("YE", 0.0),
             "Monthly check, 3% band": ("ME", 0.03), "Monthly check, 5% band": ("ME", 0.05)}
    rows = {}
    for label, (freq, band) in rules.items():
        res = walk_forward(rets, static_strategy(weights), bt["lookback_days"], freq,
                           bt["cost_bps"], band)
        s = res.summary(rf, periods=cfg.trading_days)
        rows[label] = s[["Annual return", "Annual volatility", "Sharpe", "Annual turnover",
                         "Annual cost drag", "Trades"]]
    # never rebalance: buy and hold from the same start date
    first = R.period_ends(rets.index, "ME")
    first = first[rets.index.get_indexer(first) >= bt["lookback_days"] - 1][0]
    bh = rets[rets.index > first]
    wealth = (1 + bh).cumprod() * weights
    bh_r = wealth.sum(axis=1).pct_change().fillna(wealth.sum(axis=1).iloc[0] - 1)
    s = R.performance_summary(bh_r, rf, periods=cfg.trading_days)
    rows["Never (buy & hold)"] = pd.Series({**s[["Annual return", "Annual volatility", "Sharpe"]],
                                            "Annual turnover": 0.0, "Annual cost drag": 0.0,
                                            "Trades": 0})
    return pd.DataFrame(rows).T
