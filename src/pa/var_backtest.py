"""Does the VaR model work? Rolling out-of-sample VaR forecasts compared with
realised losses, then formal tests of the breach count and of breach clustering.

- Kupiec POF test: is the breach frequency consistent with 1 - alpha?
- Christoffersen independence test: are breaches independent, or do they
  cluster (a sign the model reacts too slowly to changing volatility)?
- Conditional coverage = both together (chi-square, 2 d.o.f.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rolling_var_forecast(r: pd.Series, method: str = "historical", window: int = 500,
                         alpha: float = 0.95, lam: float = 0.94) -> pd.Series:
    """1-day VaR forecast for each day t using only returns up to t-1."""
    z = stats.norm.ppf(1 - alpha)
    if method == "historical":
        v = -r.rolling(window).quantile(1 - alpha)
    elif method == "normal":
        v = -(r.rolling(window).mean() + z * r.rolling(window).std())
    elif method == "ewma":
        v = -z * np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())
        v.iloc[:window] = np.nan  # same burn-in as the other methods
    elif method == "filtered_historical":
        # Filtered historical simulation: rescale past returns by current EWMA
        # vol, then take the empirical quantile of the rescaled returns.
        sig = np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())
        std_r = r / sig.shift(1)
        v = -std_r.rolling(window).quantile(1 - alpha) * sig
    else:
        raise ValueError(method)
    return v.shift(1).rename(f"VaR ({method})")


def kupiec_pof(breaches: pd.Series, alpha: float) -> dict:
    x = breaches.dropna().astype(bool)
    n, k = len(x), int(x.sum())
    p = 1 - alpha
    phat = k / n if n else 0.0

    def ll(prob):
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - k) * np.log(1 - prob) + k * np.log(prob)

    lr = -2 * (ll(p) - ll(phat))
    return {"Observations": n, "Breaches": k, "Expected": n * p, "Breach rate": phat,
            "Kupiec LR": lr, "Kupiec p-value": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen_independence(breaches: pd.Series) -> dict:
    x = breaches.dropna().astype(int).to_numpy()
    prev, cur = x[:-1], x[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def xlogy(a, b):
        return a * np.log(b) if a > 0 else 0.0

    ll_ind = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr = -2 * (ll_null - ll_ind)
    return {"P(breach | breach yesterday)": pi1, "P(breach | none yesterday)": pi0,
            "Christoffersen LR": lr, "Christoffersen p-value": float(1 - stats.chi2.cdf(lr, 1))}


def basel_traffic_light(breaches_last_250: int) -> str:
    """Basel zones for a 99% VaR over 250 days (0-4 green, 5-9 amber, 10+ red)."""
    return "green" if breaches_last_250 <= 4 else "amber" if breaches_last_250 <= 9 else "red"


def backtest_var(r: pd.Series, methods=("historical", "normal", "ewma", "filtered_historical"),
                 window: int = 500, alpha: float = 0.95, lam: float = 0.94):
    """Returns (summary table, DataFrame of returns and each method's VaR path)."""
    paths = {"Return": r}
    rows = {}
    for m in methods:
        v = rolling_var_forecast(r, m, window, alpha, lam)
        paths[m] = v
        valid = v.notna()
        br = (r[valid] < -v[valid])
        res = kupiec_pof(br, alpha)
        res.update(christoffersen_independence(br))
        lr_cc = res["Kupiec LR"] + res["Christoffersen LR"]
        res["Cond. coverage p-value"] = float(1 - stats.chi2.cdf(lr_cc, 2))
        res["Verdict"] = "pass" if res["Cond. coverage p-value"] > 0.05 else "reject"
        rows[m] = res
    return pd.DataFrame(rows).T, pd.DataFrame(paths)
