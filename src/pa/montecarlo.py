"""Monte Carlo simulation of portfolio value over a horizon.

Weights are held constant (daily rebalanced) in every model. Each model's
assumptions are in ASSUMPTIONS so the dashboard can show them next to results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSUMPTIONS = {
    "normal": "Daily asset returns are multivariate normal with historical mean and covariance, "
              "independent across days. Fat tails and volatility clustering are ignored, so this "
              "understates tail risk; use it as the baseline.",
    "student_t": "Multivariate Student-t with the same covariance but fat tails (low degrees of "
                 "freedom). Large joint moves are more frequent than under the normal. Days are still "
                 "independent.",
    "bootstrap": "Resamples blocks of actual historical days (stationary bootstrap, average block 20 "
                 "days). Keeps real fat tails, cross-asset co-movement and some volatility "
                 "clustering, but can't produce anything worse than history.",
    "jump": "Normal diffusion plus random crash days (Poisson). A crash day replays a historical "
            "equity tail day (worst 1%), scaled up by 1.5x, across all assets at once. This is an "
            "extreme scenario beyond the historical record.",
}


def _draw_normal(rng, mu, L, n):
    return mu + rng.standard_normal((n, len(mu))) @ L.T


def simulate(asset_returns: pd.DataFrame, weights: pd.Series, model: str = "normal",
             horizon: int = 252, n_paths: int = 5000, seed: int = 42, t_dof: float = 4.0,
             jump_intensity: float = 3.0, jump_scale: float = 1.5,
             equity_ticker: str = "SPY", periods: int = 252) -> np.ndarray:
    """Returns an (n_paths, horizon) array of portfolio daily returns."""
    rng = np.random.default_rng(seed)
    R = asset_returns[weights.index].dropna()
    w = weights.to_numpy(dtype=float)
    mu = R.mean().to_numpy()
    cov = R.cov().to_numpy()
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mu)))
    out = np.empty((n_paths, horizon))

    if model == "bootstrap":
        # stationary bootstrap, vectorised across paths
        hist = R.to_numpy() @ w
        n = len(hist)
        idx = np.empty((n_paths, horizon), dtype=int)
        idx[:, 0] = rng.integers(n, size=n_paths)
        new_block = rng.random((n_paths, horizon)) < 1 / 20
        starts = rng.integers(n, size=(n_paths, horizon))
        for t in range(1, horizon):
            idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
        return hist[idx]

    if model == "jump":
        eq = R[equity_ticker] if equity_ticker in R else R.to_numpy() @ w
        tail_days = R[eq <= np.quantile(eq, 0.01)].to_numpy()
        p_jump = jump_intensity / periods

    for t in range(horizon):
        if model == "normal":
            x = _draw_normal(rng, mu, L, n_paths)
        elif model == "student_t":
            z = _draw_normal(rng, np.zeros_like(mu), L * np.sqrt((t_dof - 2) / t_dof), n_paths)
            x = mu + z / np.sqrt(rng.chisquare(t_dof, n_paths) / t_dof)[:, None]
        elif model == "jump":
            x = _draw_normal(rng, mu, L, n_paths)
            hit = rng.random(n_paths) < p_jump
            if hit.any():
                x[hit] = tail_days[rng.integers(len(tail_days), size=hit.sum())] * jump_scale
        else:
            raise ValueError(model)
        out[:, t] = x @ w
    return out


def summarise(sim: np.ndarray, nav: float = 1.0, alphas=(0.95, 0.99)) -> dict:
    wealth = np.cumprod(1 + sim, axis=1)
    terminal = wealth[:, -1] - 1
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0), axis=1)
    mdd = (wealth / peak - 1).min(axis=1)
    stats = {"Mean return": terminal.mean(), "Median return": np.median(terminal),
             "Std dev": terminal.std(), "P(loss)": (terminal < 0).mean(),
             "P(loss > 10%)": (terminal < -0.10).mean(), "P(loss > 20%)": (terminal < -0.20).mean(),
             "Median max drawdown": np.median(mdd), "5% worst max drawdown": np.quantile(mdd, 0.05)}
    for a in alphas:
        q = np.quantile(terminal, 1 - a)
        stats[f"VaR {a:.0%}"] = -q
        stats[f"ES {a:.0%}"] = -terminal[terminal <= q].mean()
        stats[f"VaR {a:.0%} ($)"] = -q * nav
    fan = pd.DataFrame({f"p{int(p * 100)}": np.quantile(wealth, p, axis=0)
                        for p in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)})
    return {"stats": pd.Series(stats), "fan": fan, "terminal": terminal, "max_drawdown": mdd}
