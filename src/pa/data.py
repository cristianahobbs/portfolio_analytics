"""Market data: live (Yahoo Finance + FRED) with a parquet cache, or a synthetic
market with regime switches and fat tails for offline use and tests.

Prices are dividend/split-adjusted closes (auto_adjust=True), so returns are
total returns, not price returns.
"""
from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Config

log = logging.getLogger(__name__)
CACHE_DIR = ROOT / "data" / "cache"


@dataclass
class MarketData:
    prices: pd.DataFrame        # adjusted closes (VIX and FX are levels)
    volumes: pd.DataFrame       # shares traded
    cpi: pd.Series | None       # monthly CPI index (None if unavailable)
    source: str                 # "live" | "synthetic"

    def returns(self, tickers: list[str] | None = None) -> pd.DataFrame:
        p = self.prices if tickers is None else self.prices[tickers]
        return p.pct_change(fill_method=None).iloc[1:]

    def adv_usd(self, window: int = 60) -> pd.Series:
        """Average daily traded value over the last `window` days."""
        traded = (self.prices * self.volumes).iloc[-window:]
        return traded.mean()


# --------------------------------------------------------------------------
# Live data
# --------------------------------------------------------------------------
def _cache_path(tickers, start, end) -> Path:
    key = hashlib.md5(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"yahoo_{key}.parquet"


def fetch_yahoo(tickers: list[str], start: str, end: str | None, use_cache: bool = True):
    end = end or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    path = _cache_path(tickers, start, end)
    if use_cache and path.exists():
        df = pd.read_parquet(path)
    else:
        import yfinance as yf

        df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False, group_by="column", threads=True)
        if df is None or df.empty:
            raise RuntimeError("Yahoo Finance returned no data")
        df = df[["Close", "Volume"]]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    prices, volumes = df["Close"], df["Volume"]
    missing = [t for t in tickers if t not in prices or prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No data for {missing}")
    return prices[tickers], volumes[tickers]


def fetch_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])), name=series_id)
    return s.dropna()


def _align(prices: pd.DataFrame, volumes: pd.DataFrame, anchor: str):
    """Keep the anchor's trading days; fill short gaps from other calendars
    (VIX, FX) forward. Never back-fill: that would leak future prices."""
    idx = prices.index[prices[anchor].notna()]
    prices = prices.reindex(idx).ffill(limit=5)
    volumes = volumes.reindex(idx).fillna(0.0)
    return prices, volumes


# --------------------------------------------------------------------------
# Synthetic market
# --------------------------------------------------------------------------
FACTORS = ["EQ", "RATES", "CREDIT", "CMD", "USD", "INFL", "GOLD"]
# annual vol / drift of each factor in the calm regime
_F_VOL = np.array([0.15, 0.06, 0.04, 0.18, 0.07, 0.03, 0.15])
_F_MU = np.array([0.09, 0.025, 0.02, 0.03, 0.0, 0.01, 0.05])
# crisis regime: higher vol, risk-off drifts
_CRISIS_VOL_MULT = np.array([2.8, 1.6, 2.5, 2.0, 1.5, 1.5, 1.6])
_CRISIS_MU = np.array([-0.60, 0.15, -0.25, -0.40, 0.10, -0.05, 0.10])
_CALM_CORR = np.array([
    [1.0, -0.2, 0.4, 0.3, -0.1, 0.1, 0.0],
    [-0.2, 1.0, -0.2, -0.1, 0.0, 0.0, 0.2],
    [0.4, -0.2, 1.0, 0.2, 0.0, 0.1, 0.0],
    [0.3, -0.1, 0.2, 1.0, -0.3, 0.3, 0.3],
    [-0.1, 0.0, 0.0, -0.3, 1.0, 0.0, -0.3],
    [0.1, 0.0, 0.1, 0.3, 0.0, 1.0, 0.2],
    [0.0, 0.2, 0.0, 0.3, -0.3, 0.2, 1.0],
])
_CRISIS_CORR = _CALM_CORR.copy()
for (i, j), v in {(0, 1): -0.5, (0, 2): 0.75, (0, 3): 0.6, (1, 2): -0.4}.items():
    _CRISIS_CORR[i, j] = _CRISIS_CORR[j, i] = v

# ticker -> (factor loadings dict, idiosyncratic annual vol, extra annual drift)
_LOADINGS = {
    "SPY": ({"EQ": 1.0}, 0.01, 0.0),
    "EFA": ({"EQ": 0.9, "USD": -0.5}, 0.06, -0.02),
    "EEM": ({"EQ": 1.1, "USD": -0.7, "CMD": 0.1}, 0.10, -0.02),
    "AAPL": ({"EQ": 1.2}, 0.25, 0.10),
    "JPM": ({"EQ": 1.25, "CREDIT": 0.6}, 0.20, 0.0),
    "JNJ": ({"EQ": 0.6}, 0.14, 0.0),
    "XOM": ({"EQ": 0.8, "CMD": 0.5}, 0.18, -0.02),
    "PG": ({"EQ": 0.5}, 0.13, 0.0),
    "CAT": ({"EQ": 1.15, "CMD": 0.3}, 0.22, 0.01),
    "IEF": ({"RATES": 1.0}, 0.01, 0.0),
    "TLT": ({"RATES": 2.2}, 0.03, 0.0),
    "AGG": ({"RATES": 0.75, "CREDIT": 0.15}, 0.01, 0.0),
    "LQD": ({"RATES": 1.1, "CREDIT": 0.5, "EQ": 0.03}, 0.015, 0.0),
    "HYG": ({"RATES": 0.4, "CREDIT": 1.2, "EQ": 0.1}, 0.02, 0.0),
    "TIP": ({"RATES": 0.9, "INFL": 1.0}, 0.01, 0.0),
    "UUP": ({"USD": 1.0}, 0.01, -0.01),
    "DBC": ({"CMD": 1.0}, 0.03, -0.02),
    "GLD": ({"GOLD": 1.0, "USD": -0.3}, 0.03, 0.0),
}


def synthetic_market(tickers: list[str], start: str, end: str | None = None, seed: int = 7,
                     adv: dict | None = None, vix: str = "^VIX", fx: str = "GBPUSD=X") -> MarketData:
    """Two-regime factor model with Student-t shocks. Calm/crisis regimes follow
    a Markov chain, so crises cluster and correlations jump in a crisis, as
    they do in real markets."""
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    idx = pd.bdate_range(start, end)
    n, k, dt = len(idx), len(FACTORS), 1 / 252

    # regime path: calm persists ~2y, crisis ~3 months
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.998 if regime[t - 1] == 0 else 0.985
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]

    chol = [np.linalg.cholesky(_CALM_CORR), np.linalg.cholesky(_CRISIS_CORR)]
    dof = 5
    z = rng.standard_t(dof, size=(n, k)) / np.sqrt(dof / (dof - 2))
    f = np.empty((n, k))
    for s in (0, 1):
        m = regime == s
        vol = _F_VOL * (_CRISIS_VOL_MULT if s else 1.0)
        mu = _CRISIS_MU if s else _F_MU
        f[m] = mu * dt + (z[m] @ chol[s].T) * vol * np.sqrt(dt)
    fac = pd.DataFrame(f, index=idx, columns=FACTORS)

    rets = {}
    for t in tickers:
        if t in (vix, fx):
            continue
        if t == "BIL":
            rets[t] = pd.Series(0.015 * dt + rng.normal(0, 0.0003, n), index=idx)
            continue
        load, idio, drift = _LOADINGS.get(t, ({"EQ": 1.0}, 0.2, 0.0))
        r = sum(b * fac[name] for name, b in load.items())
        stress = np.where(regime == 1, 1.5, 1.0)
        r = r + drift * dt + rng.standard_normal(n) * idio * np.sqrt(dt) * stress
        rets[t] = r
    rets = pd.DataFrame(rets)
    prices = 100 * (1 + rets).cumprod()

    if vix in tickers:
        eq = fac["EQ"]
        ewma = np.sqrt((eq ** 2).ewm(alpha=0.06).mean() * 252)
        prices[vix] = (100 * ewma * 1.1 + rng.normal(0, 1.0, n)).clip(lower=9)
    if fx in tickers:
        fx_r = -0.8 * fac["USD"] + rng.normal(0, 0.004, n)
        prices[fx] = 1.6 * (1 + fx_r).cumprod()
    prices = prices[tickers]

    adv = adv or {}
    vols = pd.DataFrame({t: adv.get(t, 1e9) / prices[t] * rng.lognormal(0, 0.3, n)
                         for t in tickers}, index=idx)
    for t in (vix, fx):
        if t in vols:
            vols[t] = 0.0
    cpi = (pd.Series(rng.normal(0.0022, 0.002, len(pd.date_range(start, end, freq="MS"))),
                     index=pd.date_range(start, end, freq="MS")).add(1).cumprod() * 200)
    return MarketData(prices, vols[tickers], cpi.rename("CPI"), "synthetic")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def load_market_data(cfg: Config, source: str | None = None, use_cache: bool = True) -> MarketData:
    source = source or cfg.get("data_source", "auto")
    tickers = cfg.all_tickers()
    anchor = cfg.tickers[0]
    if source in ("auto", "live"):
        try:
            prices, volumes = fetch_yahoo(tickers, cfg["start"], cfg.get("end"), use_cache)
            prices, volumes = _align(prices, volumes, anchor)
            try:
                cpi = fetch_fred(cfg.get("inflation_series", "CPIAUCSL"))
            except Exception as e:  # CPI is optional: only real returns need it
                log.warning("CPI download failed (%s); real returns disabled", e)
                cpi = None
            return MarketData(prices, volumes, cpi, "live")
        except Exception as e:
            if source == "live":
                raise
            warnings.warn(f"Live data unavailable ({e}); using synthetic market data.")
    return synthetic_market(tickers, cfg["start"], cfg.get("end"),
                            adv=cfg.assets["adv_usd"].to_dict(),
                            vix=cfg["vix_ticker"], fx=cfg["fx"]["fx_ticker"])
