"""Correlations: static, rolling, and conditional on market regime."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def to_weekly(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound to weekly. Weekly returns reduce the bias from markets that
    close at different times (e.g. Europe/Asia vs US), which pushes daily
    correlations towards zero."""
    return returns.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)


def rolling_correlation(returns: pd.DataFrame, reference: str, window: int = 126) -> pd.DataFrame:
    """Rolling correlation of every column with `reference`."""
    cols = [c for c in returns.columns if c != reference]
    return returns[cols].rolling(window).corr(returns[reference])


def average_pairwise_correlation(returns: pd.DataFrame, window: int = 126) -> pd.Series:
    """Mean of the off-diagonal rolling correlations: one diversification gauge."""
    n = returns.shape[1]
    rc = returns.rolling(window).corr()
    out = rc.groupby(level=0).apply(lambda m: (m.to_numpy().sum() - n) / (n * (n - 1)))
    return out.rename("Average correlation")


def regime_correlations(returns: pd.DataFrame, stress_indicator: pd.Series,
                        quantile: float = 0.8) -> dict:
    """Correlation matrices in calm vs stressed periods. Stress = indicator above
    its `quantile` (e.g. VIX), using the previous day's value so the regime is
    known in advance."""
    ind = stress_indicator.reindex(returns.index).shift(1)
    thresh = ind.quantile(quantile)
    stressed = ind > thresh
    return {"calm": returns[~stressed & ind.notna()].corr(),
            "stressed": returns[stressed].corr(),
            "threshold": float(thresh), "share_stressed": float(stressed.mean())}


# --------------------------------------------------------------------------
# Two-state Gaussian hidden Markov model (Baum-Welch)
# --------------------------------------------------------------------------
def fit_hmm_2state(x: pd.Series, n_iter: int = 200, tol: float = 1e-6, seed: int = 0):
    """Fit a 2-regime HMM to a return series. Returns a dict with
    parameters, smoothed P(high-vol regime) for each date, and log-likelihood.
    State 1 is always the higher-volatility regime.

    The smoothed probabilities use the whole sample (forward-backward), so they
    describe history. Use `filtered` for a real-time view."""
    y = x.dropna().to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    mu = np.array([y.mean(), y.mean()]) + rng.normal(0, y.std() * 0.01, 2)
    sd = np.array([y.std() * 0.7, y.std() * 1.8])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.8, 0.2])
    prev_ll = -np.inf
    for _ in range(n_iter):
        B = stats.norm.pdf(y[:, None], mu, sd) + 1e-300
        # forward (scaled)
        alpha = np.empty((n, 2))
        c = np.empty(n)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        # backward
        beta = np.empty((n, 2))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        # M-step
        pi = gamma[0]
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        w = gamma.sum(axis=0)
        mu = (gamma * y[:, None]).sum(axis=0) / w
        sd = np.sqrt((gamma * (y[:, None] - mu) ** 2).sum(axis=0) / w)
        ll = np.log(c).sum()
        if ll - prev_ll < tol:
            break
        prev_ll = ll
    hi = int(np.argmax(sd))
    order = [1 - hi, hi]
    idx = x.dropna().index
    return {
        "mean": mu[order], "vol": sd[order], "transition": A[np.ix_(order, order)],
        "smoothed": pd.Series(gamma[:, hi], index=idx, name="P(high-vol regime)"),
        "filtered": pd.Series(alpha[:, hi], index=idx, name="P(high-vol regime), real-time"),
        "loglik": float(ll),
        "expected_duration_days": 1 / (1 - np.diag(A[np.ix_(order, order)])),
    }
