"""One object holding the data every view needs, so the dashboard, tear sheet
and notebooks compute things the same way."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from . import attribution, factors, liquidity, risk
from . import returns as R
from .config import Config
from .data import MarketData, load_market_data
from .drawdown import drawdown_series


@dataclass
class Analysis:
    cfg: Config
    md: MarketData

    @classmethod
    def load(cls, cfg: Config, source: str | None = None) -> "Analysis":
        return cls(cfg, load_market_data(cfg, source))

    # --- return series ---------------------------------------------------
    @cached_property
    def all_returns(self) -> pd.DataFrame:
        """Daily returns. A holding that listed after the start date takes its
        proxy's returns before listing (see `backfilled`); its own returns are
        used from the first full day of trading."""
        cols = [c for c in self.md.prices.columns if c not in (self.cfg["vix_ticker"],
                                                               self.cfg["fx"]["fx_ticker"])]
        r = self.md.returns(cols)
        for t, proxy in self.cfg.proxies.items():
            if t in r and proxy in r:
                pre = r[t].isna() & (r.index <= self.md.prices[t].first_valid_index())
                r.loc[pre, t] = r.loc[pre, proxy]
        return r.dropna(subset=self.cfg.tickers)

    @cached_property
    def backfilled(self) -> pd.DataFrame:
        """Holdings whose early history comes from a proxy, and until when."""
        rows = {}
        start = self.all_returns.index[0]
        for t, proxy in self.cfg.proxies.items():
            listed = self.md.prices[t].first_valid_index()
            if listed is not None and listed > start:
                rows[t] = {"Proxy": proxy, "Own data from": listed,
                           "Share of history from proxy": float((self.all_returns.index <= listed).mean())}
        return pd.DataFrame(rows).T

    @property
    def asset_returns(self) -> pd.DataFrame:
        return self.all_returns[self.cfg.tickers]

    @property
    def periods(self) -> int:
        return self.cfg.trading_days

    @cached_property
    def rf(self) -> pd.Series:
        return self.all_returns[self.cfg.cash_ticker].rename("rf")

    @cached_property
    def bod_weights(self) -> pd.DataFrame:
        return attribution.drifting_weights(self.asset_returns, self.cfg.weights,
                                            self.cfg["backtest"]["rebalance"])

    @cached_property
    def portfolio(self) -> pd.Series:
        """Gross of fees, rebalanced to target weights at each period end."""
        return R.portfolio_returns(self.asset_returns, self.cfg.weights,
                                   self.cfg["backtest"]["rebalance"])

    @cached_property
    def portfolio_net(self) -> pd.Series:
        return R.net_of_fees(self.portfolio, self.cfg["fees"]["management_fee"], self.periods
                             ).rename("portfolio (net)")

    @cached_property
    def benchmark(self) -> pd.Series:
        bw = self.cfg.benchmark_weights
        return R.portfolio_returns(self.all_returns[bw.index], bw, "ME").rename("benchmark")

    @property
    def vix(self) -> pd.Series | None:
        v = self.cfg["vix_ticker"]
        return self.md.prices[v] if v in self.md.prices else None

    # --- risk inputs -----------------------------------------------------
    @cached_property
    def cov(self) -> pd.DataFrame:
        """Annualised Ledoit-Wolf covariance over the last 3 years."""
        return risk.ledoit_wolf_cov(self.asset_returns.iloc[-756:], self.periods)

    @property
    def cov_daily(self) -> pd.DataFrame:
        return self.cov / self.periods

    @cached_property
    def cov_union(self) -> pd.DataFrame:
        """Covariance over portfolio and benchmark holdings (e.g. adds AGG),
        needed for active risk and tracking error."""
        tickers = list(dict.fromkeys(self.cfg.tickers + list(self.cfg.benchmark_weights.index)))
        return risk.ledoit_wolf_cov(self.all_returns[tickers].iloc[-756:], self.periods)

    @cached_property
    def group_union(self) -> pd.Series:
        g = self.cfg.group.copy()
        for name, spec in self.cfg["benchmark"]["groups"].items():
            if spec["proxy"] not in g.index:
                g[spec["proxy"]] = name
        return g

    @cached_property
    def active(self) -> dict:
        return risk.active_risk(self.cfg.weights, self.cfg.benchmark_weights, self.cov_union,
                                self.group_union)

    @cached_property
    def excess_returns(self) -> pd.DataFrame:
        return self.asset_returns.sub(self.rf, axis=0)

    @cached_property
    def asset_factor_exposures(self) -> pd.DataFrame:
        return factors.exposures(self.cfg, self.all_returns, self.asset_returns)

    @cached_property
    def equity_betas(self) -> pd.Series:
        eq = self.cfg["factors"]["Equity"]
        b = self.asset_returns.apply(lambda c: R.beta(c, self.all_returns[eq]))
        return b.rename("Equity beta")

    @cached_property
    def liquidity(self) -> pd.DataFrame:
        cfg = self.cfg
        adv = (self.md.adv_usd() if self.md.source == "live" else cfg.assets["adv_usd"])
        adv = adv.reindex(cfg.tickers).fillna(cfg.assets["adv_usd"])
        L = cfg["liquidity"]
        return liquidity.liquidity_table(cfg.weights, cfg["nav"], adv, cfg.assets["spread_bps"],
                                         self.asset_returns.iloc[-252:].std(),
                                         L["participation_rate"], L["stress_adv_haircut"],
                                         L["stress_spread_mult"])

    # --- headline numbers --------------------------------------------------
    def current_limit_metrics(self) -> dict:
        p = self.portfolio
        recent = p.iloc[-self.cfg["risk"]["var_window"]:]
        te = (p - self.benchmark).iloc[-252:].std() * np.sqrt(self.periods)
        return {
            "var_1d_95": risk.historical_var(recent, 0.95),
            "es_1d_975": risk.historical_es(recent, 0.975),
            "max_drawdown": float(drawdown_series(p).iloc[-1]),
            "max_position": float(self.cfg.weights.max()),
            "equity_beta": float(self.cfg.weights @ self.equity_betas),
            "tracking_error": float(te),
            "days_to_liquidate_90pct": liquidity.days_to_liquidate_fraction(
                self.liquidity, self.cfg.weights, 0.9),
        }
