"""Portfolio construction under realistic constraints.

Expected returns are *excess* returns over cash throughout, so the cash asset
has an expected excess return of ~0 and Sharpe ratios need no rf term.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .config import Config


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------
@dataclass
class Constraints:
    tickers: list[str]
    lower: np.ndarray
    upper: np.ndarray
    asset_class: pd.Series
    class_ranges: dict
    cash_ticker: str | None = None
    cash_min: float = 0.0
    allow_short: bool = False
    max_gross: float = 1.0

    @classmethod
    def from_config(cls, c: Config) -> "Constraints":
        k = c.constraints
        upper = c.assets["max_weight"].to_numpy(dtype=float)
        lower = -upper if k.get("allow_short") else np.full(len(upper), k.get("min_weight", 0.0))
        return cls(c.tickers, lower, upper, c.asset_class, k.get("asset_class_ranges", {}),
                   c.cash_ticker, k.get("cash_min", 0.0), k.get("allow_short", False),
                   k.get("max_gross", 1.0))

    def scipy(self, fix_cash: bool = False):
        lower, upper = self.lower.copy(), self.upper.copy()
        if self.cash_ticker in self.tickers:
            i = self.tickers.index(self.cash_ticker)
            lower[i] = max(lower[i], self.cash_min)
            if fix_cash:
                upper[i] = lower[i]
        bounds = list(zip(lower, upper))
        ones = np.ones(len(self.tickers))
        # linear constraints get constant analytic Jacobians (much faster SLSQP)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: ones}]
        cls = self.asset_class.reindex(self.tickers).to_numpy()
        for name, (lo, hi) in self.class_ranges.items():
            m = (cls == name).astype(float)
            if m.sum() == 0:
                continue
            cons.append({"type": "ineq", "fun": lambda w, m=m, lo=lo: m @ w - lo,
                         "jac": lambda w, m=m: m})
            cons.append({"type": "ineq", "fun": lambda w, m=m, hi=hi: hi - m @ w,
                         "jac": lambda w, m=m: -m})
        if self.allow_short:
            cons.append({"type": "ineq", "fun": lambda w: self.max_gross - np.abs(w).sum(),
                         "jac": lambda w: -np.sign(w)})
        return bounds, cons

    def violations(self, w: pd.Series, tol: float = 1e-4) -> list[str]:
        w = w.reindex(self.tickers).fillna(0.0)
        out = []
        if abs(w.sum() - 1) > tol:
            out.append(f"weights sum to {w.sum():.3f}")
        for t, lo, hi in zip(self.tickers, self.lower, self.upper):
            if w[t] < lo - tol or w[t] > hi + tol:
                out.append(f"{t} {w[t]:.1%} outside [{lo:.0%}, {hi:.0%}]")
        for name, (lo, hi) in self.class_ranges.items():
            s = w[self.asset_class.reindex(self.tickers) == name].sum()
            if s < lo - tol or s > hi + tol:
                out.append(f"{name} {s:.1%} outside [{lo:.0%}, {hi:.0%}]")
        if self.cash_ticker in w and w[self.cash_ticker] < self.cash_min - tol:
            out.append(f"cash {w[self.cash_ticker]:.1%} below minimum {self.cash_min:.0%}")
        if w.abs().sum() > self.max_gross + tol:
            out.append(f"gross {w.abs().sum():.2f} above {self.max_gross:.2f}")
        return out


# --------------------------------------------------------------------------
# Expected returns
# --------------------------------------------------------------------------
def historical_mean(excess_returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    return excess_returns.mean() * periods


def shrunk_mean(excess_returns: pd.DataFrame, intensity: float = 0.5, periods: int = 252):
    """Shrink each asset's mean towards the cross-sectional average, which
    reduces the estimation noise that dominates sample means."""
    mu = historical_mean(excess_returns, periods)
    return (1 - intensity) * mu + intensity * mu.mean()


def implied_returns(cov: pd.DataFrame, w_ref: pd.Series, risk_aversion: float = 2.5) -> pd.Series:
    """Reverse optimisation: the excess returns that make `w_ref` optimal."""
    return (risk_aversion * cov @ w_ref.reindex(cov.index).fillna(0.0)).rename("Implied")


def black_litterman(cov: pd.DataFrame, w_ref: pd.Series, views: list[dict],
                    risk_aversion: float = 2.5, tau: float = 0.05):
    """Black-Litterman posterior expected excess returns.

    views: dicts with type 'absolute' (asset, return) or 'relative'
    (long, short, return), plus confidence in (0, 1]. View uncertainty follows
    the common heuristic Omega_kk = (1-c)/c * (P tau Sigma P')_kk.
    Returns (posterior, prior, view table)."""
    assets = list(cov.index)
    pi = implied_returns(cov, w_ref, risk_aversion)
    if not views:
        return pi.rename("Posterior"), pi, pd.DataFrame()
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    conf = np.zeros(len(views))
    labels = []
    for k, v in enumerate(views):
        if v["type"] == "absolute":
            P[k, assets.index(v["asset"])] = 1.0
            labels.append(f"{v['asset']} returns {v['return']:.1%}")
        else:
            P[k, assets.index(v["long"])] = 1.0
            P[k, assets.index(v["short"])] = -1.0
            labels.append(f"{v['long']} beats {v['short']} by {v['return']:.1%}")
        Q[k] = v["return"]
        conf[k] = min(max(v.get("confidence", 0.5), 1e-3), 0.999)
    S = cov.to_numpy()
    tS = tau * S
    omega = np.diag((1 - conf) / conf * np.diag(P @ tS @ P.T))
    inv_tS = np.linalg.inv(tS)
    inv_om = np.linalg.inv(omega)
    M = np.linalg.inv(inv_tS + P.T @ inv_om @ P)
    post = M @ (inv_tS @ pi.to_numpy() + P.T @ inv_om @ Q)
    post = pd.Series(post, index=assets, name="Posterior")
    view_tbl = pd.DataFrame({"View": labels, "View return": Q, "Confidence": conf,
                             "Prior implied": P @ pi.to_numpy(), "Posterior": P @ post.to_numpy()})
    return post, pi, view_tbl


def expected_returns(cfg: Config, excess_returns: pd.DataFrame, cov: pd.DataFrame,
                     method: str | None = None) -> pd.Series:
    er = cfg["expected_returns"]
    method = method or er["method"]
    if method == "historical":
        return historical_mean(excess_returns, cfg.trading_days)
    if method == "shrunk":
        return shrunk_mean(excess_returns, periods=cfg.trading_days)
    if method == "black_litterman":
        return black_litterman(cov, cfg.weights, er.get("views", []),
                               er.get("risk_aversion", 2.5), er.get("tau", 0.05))[0]
    raise ValueError(method)


# --------------------------------------------------------------------------
# Optimisers
# --------------------------------------------------------------------------
def _solve(obj, jac, x0, cons: Constraints, fix_cash=False):
    bounds, c = cons.scipy(fix_cash)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(obj, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=c,
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success:
        warnings.warn(f"Optimiser did not converge: {res.message}")
    return pd.Series(res.x, index=cons.tickers).round(10)


def min_volatility(cov: pd.DataFrame, cons: Constraints, x0=None) -> pd.Series:
    S = cov.loc[cons.tickers, cons.tickers].to_numpy() * 1e4   # scaled for the solver
    x0 = np.full(len(S), 1 / len(S)) if x0 is None else np.asarray(x0)
    return _solve(lambda w: w @ S @ w, lambda w: 2 * S @ w, x0, cons)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, cons: Constraints, x0=None) -> pd.Series:
    S = cov.loc[cons.tickers, cons.tickers].to_numpy()
    m = mu.reindex(cons.tickers).to_numpy()
    x0 = np.full(len(S), 1 / len(S)) if x0 is None else np.asarray(x0)

    def obj(w):
        return -(w @ m) / np.sqrt(w @ S @ w)

    def jac(w):
        v = np.sqrt(w @ S @ w)
        return -(m / v - (w @ m) * (S @ w) / v ** 3)

    return _solve(obj, jac, x0, cons)


def risk_parity(cov: pd.DataFrame, cons: Constraints, x0=None) -> pd.Series:
    """Equal risk contribution across risky assets; cash is held at its minimum.
    Where constraints bind, contributions are as equal as the limits allow."""
    S = cov.loc[cons.tickers, cons.tickers].to_numpy()
    risky = np.array([t != cons.cash_ticker for t in cons.tickers])
    target = risky / risky.sum()

    def obj(w):
        Sw = S @ w
        rc = w * Sw / (w @ Sw)
        return ((rc - target) ** 2).sum() * 1e3

    def jac(w):
        Sw = S @ w
        v = w @ Sw
        rc = w * Sw / v
        # d rc_i / d w_j = (delta_ij Sw_i + w_i S_ij) / v - 2 rc_i Sw_j / v
        J = (np.diag(Sw) + w[:, None] * S) / v - 2 * np.outer(rc, Sw) / v
        return 2e3 * J.T @ (rc - target)

    # inverse-vol start is close to the answer and helps convergence
    iv = risky / np.sqrt(np.diag(S))
    x0 = iv / iv.sum() if x0 is None else np.asarray(x0)
    return _solve(obj, jac, x0, cons, fix_cash=True)


def hrp(cov: pd.DataFrame, cons: Constraints | None = None) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado 2016). Clusters assets by
    correlation, then splits risk top-down. Needs no matrix inversion or
    expected returns. Applies only the cash minimum; other constraints are not
    imposed, and the comparison table shows any breaches."""
    tickers = list(cov.index) if cons is None else cons.tickers
    cash = cons.cash_ticker if cons else None
    risky = [t for t in tickers if t != cash]
    C = cov.loc[risky, risky].to_numpy()
    sd = np.sqrt(np.diag(C))
    corr = C / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), "single")))
    wv = np.ones(len(risky))

    def cluster_var(ix):
        c = C[np.ix_(ix, ix)]
        ivp = 1 / np.diag(c)
        ivp /= ivp.sum()
        return ivp @ c @ ivp

    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            a, b = c[: len(c) // 2], c[len(c) // 2:]
            va, vb = cluster_var(a), cluster_var(b)
            alpha = 1 - va / (va + vb)
            wv[a] *= alpha
            wv[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    cash_w = cons.cash_min if (cons and cash in tickers) else 0.0
    w = pd.Series(wv, index=risky) * (1 - cash_w)
    if cash in tickers:
        w[cash] = cash_w
    return w.reindex(tickers).fillna(0.0)


def equal_weight(cons: Constraints) -> pd.Series:
    risky = [t for t in cons.tickers if t != cons.cash_ticker]
    w = pd.Series(0.0, index=cons.tickers)
    cash_w = cons.cash_min if cons.cash_ticker in cons.tickers else 0.0
    w[risky] = (1 - cash_w) / len(risky)
    if cons.cash_ticker in cons.tickers:
        w[cons.cash_ticker] = cash_w
    return w


METHODS = ("max_sharpe", "min_vol", "risk_parity", "hrp", "equal_weight")


def optimise(method: str, mu: pd.Series, cov: pd.DataFrame, cons: Constraints, x0=None):
    if method == "max_sharpe":
        return max_sharpe(mu, cov, cons, x0)
    if method == "min_vol":
        return min_volatility(cov, cons, x0)
    if method == "risk_parity":
        return risk_parity(cov, cons, x0)
    if method == "hrp":
        return hrp(cov, cons)
    if method == "equal_weight":
        return equal_weight(cons)
    raise ValueError(method)


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------
def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> dict:
    w = w.reindex(cov.index).fillna(0.0)
    vol = float(np.sqrt(w @ cov @ w))
    ret = float(w @ mu.reindex(cov.index))
    rc = w * (cov @ w) / vol ** 2
    sd = np.sqrt(np.diag(cov))
    return {"Exp. excess return": ret, "Volatility": vol,
            "Sharpe": ret / vol if vol > 0 else np.nan,
            "Diversification ratio": float(w @ sd / vol),
            "Effective N (weights)": float(1 / (w ** 2).sum()),
            "Max risk contribution": float(rc.max())}


def compare_portfolios(mu, cov, cons: Constraints, current: pd.Series | None = None,
                       methods=METHODS):
    weights = {}
    if current is not None:
        weights["current"] = current.reindex(cons.tickers).fillna(0.0)
    x0 = None if current is None else weights["current"].to_numpy()
    for m in methods:
        weights[m] = optimise(m, mu, cov, cons, x0)
    W = pd.DataFrame(weights)
    S = pd.DataFrame({k: portfolio_stats(w, mu, cov) for k, w in W.items()})
    S.loc["Constraint breaches"] = [len(cons.violations(W[k])) for k in W]
    return W, S


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, cons: Constraints, n: int = 25):
    t = cons.tickers
    S = cov.loc[t, t].to_numpy()
    m = mu.reindex(t).to_numpy()
    w_min = min_volatility(cov, cons).to_numpy()
    bounds, base = cons.scipy()
    hi = minimize(lambda w: -(w @ m), w_min, method="SLSQP", bounds=bounds, constraints=base)
    r_lo, r_hi = w_min @ m, hi.x @ m
    pts = []
    x0 = w_min
    for target in np.linspace(r_lo, r_hi, n):
        c = base + [{"type": "eq", "fun": lambda w, tr=target: w @ m - tr}]
        res = minimize(lambda w: w @ S @ w * 1e4, x0, method="SLSQP", bounds=bounds,
                       constraints=c, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            x0 = res.x
            pts.append({"Return": target, "Volatility": float(np.sqrt(res.x @ S @ res.x)),
                        **dict(zip(t, res.x))})
    return pd.DataFrame(pts)


def return_sensitivity(mu: pd.Series, cov: pd.DataFrame, cons: Constraints, bump: float = 0.01,
                       method: str = "max_sharpe", x0=None) -> pd.DataFrame:
    """Weight changes (rows: asset whose expected return is raised by `bump`;
    columns: resulting change in each weight). Large entries mean the allocation
    depends on return forecasts nobody can make that precisely."""
    base = optimise(method, mu, cov, cons, x0)
    rows = {}
    for t in cons.tickers:
        m2 = mu.copy()
        m2[t] += bump
        rows[t] = optimise(method, m2, cov, cons, base.to_numpy()) - base
    return pd.DataFrame(rows).T


def resampled_weights(mu: pd.Series, cov: pd.DataFrame, cons: Constraints, n_obs: int,
                      n_draws: int = 50, method: str = "max_sharpe", seed: int = 0,
                      periods: int = 252) -> pd.DataFrame:
    """Michaud-style resampling: redraw expected returns from their sampling
    distribution (N(mu, cov / years of data)) and re-optimise. The spread of
    the resulting weights shows how uncertain the 'optimal' allocation is."""
    rng = np.random.default_rng(seed)
    years = n_obs / periods
    t = cons.tickers
    draws = rng.multivariate_normal(mu.reindex(t).to_numpy(), cov.loc[t, t].to_numpy() / years,
                                    size=n_draws)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in draws:
            out.append(optimise(method, pd.Series(d, index=t), cov, cons))
    return pd.DataFrame(out).reset_index(drop=True)
