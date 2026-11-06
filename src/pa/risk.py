"""Risk measures: volatility models, VaR/ES, covariance estimation, risk
decomposition, exposures and rate/credit sensitivities.

Sign convention: VaR and ES are reported as positive fractions of NAV
(a 1-day 95% VaR of 0.012 means a 1.2% loss is exceeded on 5% of days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Volatility models
# --------------------------------------------------------------------------
def ewma_vol(r: pd.Series, lam: float = 0.94, periods: int = 252) -> pd.Series:
    """RiskMetrics EWMA. Value at t is the forecast for t made with data up to
    t-1, so it can be used for decisions at t without look-ahead."""
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    return np.sqrt(var * periods).rename("EWMA vol")


def garch_vol(r: pd.Series, periods: int = 252, dist: str = "t"):
    """GARCH(1,1) with Student-t errors. Returns (conditional vol series,
    next-day forecast, parameters). Parameters are fitted on the full sample,
    so the in-sample path is descriptive, not a true out-of-sample forecast."""
    from arch import arch_model

    res = arch_model(r.dropna() * 100, mean="Constant", vol="GARCH", p=1, q=1,
                     dist=dist, rescale=False).fit(disp="off")
    cond = res.conditional_volatility / 100 * np.sqrt(periods)
    fc = float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]) / 100 * np.sqrt(periods))
    params = res.params.to_dict()
    params["persistence"] = params.get("alpha[1]", 0) + params.get("beta[1]", 0)
    return cond.rename("GARCH vol"), fc, params


def vol_target(r: pd.Series, target: float, lam: float = 0.94, max_leverage: float = 1.5,
               periods: int = 252, rf: pd.Series | float = 0.0):
    """Scale exposure each day by target / forecast vol (forecast from data to
    t-1). Unused capital earns rf; leverage above 1 pays rf."""
    lev = (target / ewma_vol(r, lam, periods)).clip(upper=max_leverage).fillna(1.0)
    rf_s = rf.reindex(r.index).fillna(0.0) if isinstance(rf, pd.Series) else rf
    scaled = lev * r + (1 - lev) * rf_s
    return scaled.rename("Vol-targeted"), lev.rename("Leverage")


# --------------------------------------------------------------------------
# VaR and Expected Shortfall
# --------------------------------------------------------------------------
def historical_var(r: pd.Series, alpha: float = 0.95) -> float:
    return float(-np.quantile(r.dropna(), 1 - alpha))


def historical_es(r: pd.Series, alpha: float = 0.95) -> float:
    r = r.dropna()
    q = np.quantile(r, 1 - alpha)
    return float(-r[r <= q].mean())


def parametric_var(mu: float, sigma: float, alpha: float = 0.95) -> float:
    return float(-(mu + sigma * stats.norm.ppf(1 - alpha)))


def parametric_es(mu: float, sigma: float, alpha: float = 0.95) -> float:
    z = stats.norm.ppf(1 - alpha)
    return float(-(mu - sigma * stats.norm.pdf(z) / (1 - alpha)))


def cornish_fisher_var(r: pd.Series, alpha: float = 0.95) -> float:
    """Normal VaR with the quantile adjusted for sample skew and kurtosis."""
    r = r.dropna()
    z = stats.norm.ppf(1 - alpha)
    s, k = stats.skew(r), stats.kurtosis(r)
    zcf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
           - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(-(r.mean() + zcf * r.std(ddof=1)))


def student_t_var_es(r: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    r = r.dropna()
    df, loc, scale = stats.t.fit(r)
    q = stats.t.ppf(1 - alpha, df)
    var = -(loc + scale * q)
    es = -(loc - scale * (df + q ** 2) / (df - 1) * stats.t.pdf(q, df) / (1 - alpha))
    return float(var), float(es)


def var_table(r: pd.Series, alphas=(0.95, 0.99), horizon: int = 1, nav: float | None = None):
    """VaR/ES by method. Multi-day figures use square-root-of-time scaling,
    which assumes i.i.d. returns and understates risk when vol clusters."""
    rows = {}
    mu, sd = r.mean(), r.std(ddof=1)
    for a in alphas:
        tv, te = student_t_var_es(r, a)
        rows[("Historical", a)] = (historical_var(r, a), historical_es(r, a))
        rows[("Normal", a)] = (parametric_var(mu, sd, a), parametric_es(mu, sd, a))
        rows[("Student-t", a)] = (tv, te)
        rows[("Cornish-Fisher", a)] = (cornish_fisher_var(r, a), np.nan)
    df = pd.DataFrame(rows, index=["VaR", "ES"]).T.astype(float) * np.sqrt(horizon)
    df.index.names = ["Method", "Confidence"]
    if nav:
        df["VaR ($)"], df["ES ($)"] = df["VaR"] * nav, df["ES"] * nav
    return df


# --------------------------------------------------------------------------
# Covariance estimation
# --------------------------------------------------------------------------
def sample_cov(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    return returns.cov() * periods


def ledoit_wolf_cov(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity, which stabilises
    the inverse that optimisers depend on. Closed form, same estimator as
    sklearn.covariance.LedoitWolf (checked in the tests) but much faster."""
    df = returns.dropna()
    x = df.to_numpy()
    n, p = x.shape
    x = x - x.mean(axis=0)
    S = x.T @ x / n
    mu = np.trace(S) / p
    delta = ((S - mu * np.eye(p)) ** 2).sum() / p
    beta = ((x ** 2).sum(axis=1) ** 2).sum() - n * (S ** 2).sum()
    beta = min(beta / (n ** 2 * p), delta)
    shrink = beta / delta if delta > 0 else 0.0
    cov = shrink * mu * np.eye(p) + (1 - shrink) * S
    return pd.DataFrame(cov * periods, index=df.columns, columns=df.columns)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94, periods: int = 252) -> pd.DataFrame:
    x = returns.dropna()
    cov = x.ewm(alpha=1 - lam, adjust=False).cov().iloc[-len(x.columns):]
    cov.index = x.columns
    return cov * periods


# --------------------------------------------------------------------------
# Risk decomposition
# --------------------------------------------------------------------------
def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility: contributions sum to total vol."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    s = cov.to_numpy()
    vol = float(np.sqrt(w @ s @ w))
    mrc = s @ w / vol
    crc = w * mrc
    out = pd.DataFrame({
        "Weight": w,
        "Standalone vol": np.sqrt(np.diag(s)),
        "Marginal contribution": mrc,
        "Vol contribution": crc,
        "% of risk": crc / vol,
    }, index=cov.index)
    out["Risk / weight"] = out["% of risk"] / out["Weight"].replace(0, np.nan)
    return out


def component_var(weights: pd.Series, cov_daily: pd.DataFrame, alpha: float = 0.95,
                  nav: float = 1.0) -> pd.DataFrame:
    """Parametric (zero-mean normal) marginal and component VaR. Component VaRs
    sum to portfolio VaR, so they show where the VaR comes from."""
    w = weights.reindex(cov_daily.index).fillna(0.0).to_numpy()
    s = cov_daily.to_numpy()
    z = stats.norm.ppf(alpha)
    vol = np.sqrt(w @ s @ w)
    marginal = z * (s @ w) / vol
    comp = w * marginal
    return pd.DataFrame({
        "Weight": w,
        "Marginal VaR": marginal,
        "Component VaR": comp,
        "Component VaR ($)": comp * nav,
        "% of VaR": comp / (z * vol),
    }, index=cov_daily.index)


def incremental_var(weights: pd.Series, cov_daily: pd.DataFrame, asset: str, delta: float,
                    funded_by: str | None = None, alpha: float = 0.95) -> dict:
    """VaR change from adding `delta` of NAV to `asset`, funded from
    `funded_by` (e.g. cash) or pro-rata from everything else."""
    w0 = weights.reindex(cov_daily.index).fillna(0.0)
    w1 = w0.copy()
    w1[asset] += delta
    if funded_by:
        w1[funded_by] -= delta
    else:
        others = w1.index != asset
        w1[others] -= delta * w0[others] / w0[others].sum()
    z, s = stats.norm.ppf(alpha), cov_daily.to_numpy()
    v0 = z * np.sqrt(w0 @ s @ w0)
    v1 = z * np.sqrt(w1 @ s @ w1)
    return {"VaR before": float(v0), "VaR after": float(v1), "Incremental VaR": float(v1 - v0),
            "new_weights": w1}


# --------------------------------------------------------------------------
# Active risk (relative to the benchmark)
# --------------------------------------------------------------------------
def active_risk(w_port: pd.Series, w_bench: pd.Series, cov: pd.DataFrame,
                groups: pd.Series | None = None) -> dict:
    """Ex-ante tracking error and where it comes from.

    Active weights a = w_port - w_bench over the union of holdings. TE is
    sqrt(a' S a); each position's contribution a_i (S a)_i / TE sums to TE.
    A position can add to TE even with a small active weight if it is
    volatile or poorly correlated with the benchmark."""
    idx = cov.index
    wp = w_port.reindex(idx).fillna(0.0)
    wb = w_bench.reindex(idx).fillna(0.0)
    a = wp - wb
    S = cov.to_numpy()
    Sa = S @ a.to_numpy()
    te = float(np.sqrt(a.to_numpy() @ Sa))
    ctr = a.to_numpy() * Sa / te
    table = pd.DataFrame({
        "Portfolio weight": wp, "Benchmark weight": wb, "Active weight": a,
        "Contribution to TE": ctr, "% of TE": ctr / te,
    }, index=idx)
    out = {"tracking_error": te, "table": table,
           "beta_to_benchmark": float(wp @ cov @ wb / (wb @ cov @ wb)),
           "portfolio_vol": float(np.sqrt(wp @ cov @ wp)),
           "benchmark_vol": float(np.sqrt(wb @ cov @ wb))}
    if groups is not None:
        g = groups.reindex(idx).fillna("Benchmark only")
        out["by_group"] = table.groupby(g.rename("Group"))[["Active weight", "Contribution to TE",
                                                            "% of TE"]].sum()
    return out


# --------------------------------------------------------------------------
# Exposures and sensitivities
# --------------------------------------------------------------------------
def exposures(weights: pd.Series | pd.DataFrame, betas: pd.Series | None = None):
    """Gross / net / long / short / beta-adjusted exposure. Accepts a single
    weight vector or a weights history (rows = dates)."""
    w = weights.to_frame().T if isinstance(weights, pd.Series) else weights
    out = pd.DataFrame({
        "Long": w.clip(lower=0).sum(axis=1),
        "Short": w.clip(upper=0).sum(axis=1),
        "Gross": w.abs().sum(axis=1),
        "Net": w.sum(axis=1),
    })
    if betas is not None:
        out["Beta-adjusted net"] = w.mul(betas.reindex(w.columns).fillna(0.0), axis=1).sum(axis=1)
    return out.iloc[0] if isinstance(weights, pd.Series) else out


def rate_credit_sensitivities(weights: pd.Series, duration: pd.Series, spread_duration: pd.Series,
                              nav: float) -> pd.DataFrame:
    """DV01 / CS01: $ P&L for a 1bp rise in yields / credit spreads (negative = loss)."""
    pos = weights * nav
    out = pd.DataFrame({
        "Position ($)": pos,
        "Duration": duration,
        "DV01 ($)": -pos * duration * 1e-4,
        "Spread duration": spread_duration,
        "CS01 ($)": -pos * spread_duration * 1e-4,
    })
    out.loc["Total"] = out.sum(numeric_only=True)
    out.loc["Total", ["Duration", "Spread duration"]] = [
        (weights * duration).sum(), (weights * spread_duration).sum()]
    return out
