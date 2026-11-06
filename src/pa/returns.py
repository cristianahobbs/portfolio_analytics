"""Performance metrics. Inputs are periodic simple returns (daily by default).

Conventions: annualised return is geometric (CAGR); volatility is the
annualised standard deviation; Sharpe uses excess returns over a risk-free
series (or a constant). All ratios are annualised with `periods`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649


# --------------------------------------------------------------------------
# Building portfolio return series
# --------------------------------------------------------------------------
_PERIOD = {"ME": "M", "QE": "Q", "YE": "Y", "W": "W", "W-FRI": "W"}


def period_ends(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day in `index` of each calendar period (ME, QE, YE, W)."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period(_PERIOD.get(freq, freq))).max().to_numpy())


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series,
                      rebalance: str | None = "ME") -> pd.Series:
    """Portfolio returns with weights that drift between rebalance dates.

    rebalance=None gives a constant-mix (daily rebalanced) portfolio. Otherwise
    the portfolio is reset to target weights at the close of each period end,
    which matches how a real portfolio behaves between trades.
    """
    r = asset_returns[weights.index].fillna(0.0)
    w_target = weights.to_numpy(dtype=float)
    if rebalance is None:
        return (r @ w_target).rename("portfolio")
    reb_dates = set(period_ends(r.index, rebalance))
    out = np.empty(len(r))
    w = w_target.copy()
    vals = r.to_numpy()
    for i, (dt, row) in enumerate(zip(r.index, vals)):
        port = w @ row
        out[i] = port
        w = w * (1 + row) / (1 + port)
        if dt in reb_dates:
            w = w_target.copy()
    return pd.Series(out, index=r.index, name="portfolio")


def net_of_fees(r: pd.Series, annual_fee: float, periods: int = 252) -> pd.Series:
    return (1 + r) * (1 - annual_fee) ** (1 / periods) - 1


def real_returns(r: pd.Series, cpi: pd.Series) -> pd.Series:
    """Deflate by CPI. Monthly CPI is spread over each month's trading days
    geometrically (published with a lag, so this is an ex-post measure)."""
    infl_m = cpi.pct_change().dropna()
    month = r.index.to_period("M")
    days = pd.Series(1, index=r.index).groupby(month).transform("count")
    m_infl = infl_m.copy()
    m_infl.index = m_infl.index.to_period("M")
    daily_infl = (1 + month.map(m_infl).to_numpy(dtype=float)) ** (1 / days.to_numpy()) - 1
    daily_infl = pd.Series(daily_infl, index=r.index).ffill().fillna(0.0)
    return (1 + r) / (1 + daily_infl) - 1


def cumulative(r: pd.Series | pd.DataFrame):
    return (1 + r.fillna(0)).cumprod()


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------
def ann_return(r: pd.Series, periods: int = 252) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (periods / len(r)) - 1)


def ann_vol(r: pd.Series, periods: int = 252) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def _excess(r: pd.Series, rf) -> pd.Series:
    if isinstance(rf, pd.Series):
        return (r - rf.reindex(r.index).fillna(0.0)).dropna()
    return r - rf


def sharpe(r: pd.Series, rf=0.0, periods: int = 252) -> float:
    """rf: per-period risk-free return (Series aligned to r, or a constant)."""
    ex = _excess(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf=0.0, periods: int = 252) -> float:
    ex = _excess(r, rf)
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * np.sqrt(periods)) if dd > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    w = cumulative(r)
    return float((w / w.cummax() - 1).min())


def calmar(r: pd.Series, periods: int = 252) -> float:
    mdd = max_drawdown(r)
    return float(ann_return(r, periods) / abs(mdd)) if mdd < 0 else np.nan


def omega(r: pd.Series, threshold: float = 0.0) -> float:
    d = r - threshold
    losses = -d[d < 0].sum()
    return float(d[d > 0].sum() / losses) if losses > 0 else np.nan


def hit_rate(r: pd.Series) -> float:
    return float((r > 0).mean())


# --------------------------------------------------------------------------
# Benchmark-relative
# --------------------------------------------------------------------------
def beta(r: pd.Series, b: pd.Series) -> float:
    x = pd.concat([r, b], axis=1).dropna()
    return float(np.cov(x.iloc[:, 0], x.iloc[:, 1])[0, 1] / x.iloc[:, 1].var())


def downside_beta(r: pd.Series, b: pd.Series) -> float:
    """Beta measured only on days the benchmark fell."""
    x = pd.concat([r, b], axis=1).dropna()
    x = x[x.iloc[:, 1] < 0]
    return beta(x.iloc[:, 0], x.iloc[:, 1])


def capture_ratios(r: pd.Series, b: pd.Series, freq: str = "ME") -> tuple[float, float]:
    """Up/down capture on monthly returns: portfolio's average return in
    benchmark up (down) months divided by the benchmark's."""
    m = (1 + pd.concat([r, b], axis=1).dropna()).resample(freq).prod() - 1
    up, dn = m[m.iloc[:, 1] > 0], m[m.iloc[:, 1] < 0]
    upc = up.iloc[:, 0].mean() / up.iloc[:, 1].mean() if len(up) else np.nan
    dnc = dn.iloc[:, 0].mean() / dn.iloc[:, 1].mean() if len(dn) else np.nan
    return float(upc), float(dnc)


def active_stats(r: pd.Series, b: pd.Series, periods: int = 252) -> dict:
    x = pd.concat([r, b], axis=1).dropna()
    active = x.iloc[:, 0] - x.iloc[:, 1]
    te = active.std(ddof=1) * np.sqrt(periods)
    ar = ann_return(x.iloc[:, 0], periods) - ann_return(x.iloc[:, 1], periods)
    bt = beta(x.iloc[:, 0], x.iloc[:, 1])
    # Jensen's alpha (annualised, arithmetic, rf ignored here)
    alpha = (x.iloc[:, 0].mean() - bt * x.iloc[:, 1].mean()) * periods
    return {"Active return": ar, "Tracking error": te,
            "Information ratio": ar / te if te > 0 else np.nan,
            "Beta": bt, "Alpha (ann.)": alpha,
            "Correlation": float(x.corr().iloc[0, 1])}


# --------------------------------------------------------------------------
# Statistical confidence in the Sharpe ratio
# --------------------------------------------------------------------------
def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0, rf=0.0,
                         periods: int = 252) -> float:
    """P(true Sharpe > sr_benchmark) given sample length, skew and kurtosis
    (Bailey & Lopez de Prado 2012). sr_benchmark is annualised."""
    ex = _excess(r, rf).dropna()
    n = len(ex)
    sr = ex.mean() / ex.std(ddof=1)                 # per-period Sharpe
    sr_star = sr_benchmark / np.sqrt(periods)
    g3 = stats.skew(ex)
    g4 = stats.kurtosis(ex, fisher=False)          # raw kurtosis (normal = 3)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpe_std: float, rf=0.0,
                    periods: int = 252) -> float:
    """PSR against the Sharpe you'd expect from the best of `n_trials` strategies
    with no skill. trial_sharpe_std: std dev of the trials' annualised Sharpes."""
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0, rf, periods)
    sd = trial_sharpe_std / np.sqrt(periods)
    e_max = sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                  + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe(r, e_max * np.sqrt(periods), rf, periods)


def stationary_bootstrap_indices(n: int, block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: random-length blocks keep the
    volatility clustering that an i.i.d. bootstrap would destroy."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < 1 / block
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def sharpe_confidence_interval(r: pd.Series, rf=0.0, n_boot: int = 1000, level: float = 0.95,
                               block: float = 20, periods: int = 252, seed: int = 0):
    ex = _excess(r, rf).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        s = ex[stationary_bootstrap_indices(len(ex), block, rng)]
        draws[i] = s.mean() / s.std(ddof=1) * np.sqrt(periods)
    a = (1 - level) / 2
    return float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


# --------------------------------------------------------------------------
# Summary tables
# --------------------------------------------------------------------------
def performance_summary(r: pd.Series, rf=0.0, benchmark: pd.Series | None = None,
                        periods: int = 252) -> pd.Series:
    r = r.dropna()
    out = {
        "Annual return": ann_return(r, periods),
        "Annual volatility": ann_vol(r, periods),
        "Sharpe": sharpe(r, rf, periods),
        "Sortino": sortino(r, rf, periods),
        "Calmar": calmar(r, periods),
        "Omega": omega(r),
        "Max drawdown": max_drawdown(r),
        "Skew": float(stats.skew(r)),
        "Excess kurtosis": float(stats.kurtosis(r)),
        "Hit rate": hit_rate(r),
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "Prob. Sharpe > 0": probabilistic_sharpe(r, 0.0, rf, periods),
    }
    if benchmark is not None:
        out.update(active_stats(r, benchmark, periods))
        out["Downside beta"] = downside_beta(r, benchmark)
        out["Up capture"], out["Down capture"] = capture_ratios(r, benchmark)
    return pd.Series(out, name=r.name)


def calendar_returns(r: pd.Series | pd.DataFrame, freq: str = "YE"):
    return (1 + r).resample(freq).prod() - 1


def trailing_returns(r: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Factsheet-style table: MTD, QTD, YTD, 1y, then annualised 3y/5y/10y and
    since inception. Horizons longer than the history are left out."""
    if isinstance(r, pd.Series):
        r = r.to_frame()
    end = r.index[-1]
    first = r.index[0]
    out = {}

    def total(block):
        return (1 + block).prod() - 1

    for label, freq in (("MTD", "M"), ("QTD", "Q"), ("YTD", "Y")):
        out[label] = total(r[r.index.to_period(freq) == end.to_period(freq)])
    for years in (1, 3, 5, 10):
        start = end - pd.DateOffset(years=years)
        if start < first:
            continue
        block = r[r.index > start]
        label = "1 year" if years == 1 else f"{years} years (ann.)"
        out[label] = total(block) if years == 1 else (1 + total(block)) ** (1 / years) - 1
    n_years = (end - first).days / 365.25
    out[f"Since {first:%b %Y} (ann.)"] = (1 + total(r)) ** (1 / n_years) - 1
    return pd.DataFrame(out).T


def rolling_metrics(r: pd.Series, window: int = 252, rf=0.0, benchmark: pd.Series | None = None,
                    periods: int = 252) -> pd.DataFrame:
    ex = _excess(r, rf)
    out = pd.DataFrame({
        "Return": (1 + r).rolling(window).apply(np.prod, raw=True) ** (periods / window) - 1,
        "Volatility": r.rolling(window).std() * np.sqrt(periods),
        "Sharpe": ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(periods),
    })
    if benchmark is not None:
        b = benchmark.reindex(r.index)
        out["Beta"] = r.rolling(window).cov(b) / b.rolling(window).var()
        out["Tracking error"] = (r - b).rolling(window).std() * np.sqrt(periods)
    return out
