"""Factor exposures: what macro bets does the portfolio really make?

Factors are built from tradable ETFs (config `factors`): a single ticker is
its return; [long, short] is a long-short spread. Regressions run on weekly
returns by default to reduce the bias from different market closing times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .correlation import to_weekly


def factor_returns(cfg: Config, returns: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, spec in cfg["factors"].items():
        if isinstance(spec, list):
            out[name] = returns[spec[0]] - returns[spec[1]]
        else:
            out[name] = returns[spec]
    return pd.DataFrame(out)


def ols(y: pd.Series, X: pd.DataFrame) -> dict:
    d = pd.concat([y, X], axis=1).dropna()
    yv = d.iloc[:, 0].to_numpy()
    Xv = np.column_stack([np.ones(len(d)), d.iloc[:, 1:].to_numpy()])
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ coef
    dof = len(yv) - Xv.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xv.T @ Xv)))
    r2 = 1 - resid.var() / yv.var()
    names = ["Alpha"] + list(X.columns)
    return {"coef": pd.Series(coef, index=names), "t": pd.Series(coef / se, index=names),
            "r2": float(r2), "resid_vol": float(np.sqrt(s2)), "n": len(yv)}


def exposures(cfg: Config, returns: pd.DataFrame, target: pd.DataFrame | pd.Series,
              weekly: bool = True) -> pd.DataFrame:
    """Factor betas, t-stats and R^2 for each column of `target`. Also splits
    each series' variance into systematic (R^2) and idiosyncratic (1 - R^2)."""
    F = factor_returns(cfg, returns)
    T = target.to_frame() if isinstance(target, pd.Series) else target
    if weekly:
        F, T = to_weekly(F), to_weekly(T)
    periods = 52 if weekly else cfg.trading_days
    rows = {}
    for c in T.columns:
        res = ols(T[c], F)
        row = res["coef"].drop("Alpha").to_dict()
        row.update({f"t({k})": v for k, v in res["t"].drop("Alpha").items()})
        row["Alpha (ann.)"] = res["coef"]["Alpha"] * periods
        row["R²"] = res["r2"]
        row["Idiosyncratic share"] = 1 - res["r2"]
        row["Idiosyncratic vol"] = res["resid_vol"] * np.sqrt(periods)
        rows[c] = row
    return pd.DataFrame(rows).T


def factor_risk_decomposition(cfg: Config, returns: pd.DataFrame, port: pd.Series,
                              weekly: bool = True) -> pd.DataFrame:
    """Split portfolio variance into each factor's contribution
    (beta_k * cov(f_k, beta'f)) plus the residual."""
    F = factor_returns(cfg, returns)
    y = port
    if weekly:
        F, y = to_weekly(F), to_weekly(port.to_frame()).iloc[:, 0]
    d = pd.concat([y.rename("y"), F], axis=1).dropna()
    res = ols(d["y"], d[F.columns])
    b = res["coef"].drop("Alpha")
    cov = d[F.columns].cov()
    sys_var = float(b @ cov @ b)
    contrib = b * (cov @ b)
    total = float(d["y"].var())
    out = pd.DataFrame({"Beta": b, "Variance contribution": contrib})
    out.loc["Residual (idiosyncratic)"] = [np.nan, total - sys_var]
    out["% of variance"] = out["Variance contribution"] / total
    return out


def rolling_betas(cfg: Config, returns: pd.DataFrame, port: pd.Series, window: int = 104,
                  weekly: bool = True) -> pd.DataFrame:
    """Rolling multi-factor betas (window in weeks if weekly)."""
    F = factor_returns(cfg, returns)
    y = port
    if weekly:
        F, y = to_weekly(F), to_weekly(port.to_frame()).iloc[:, 0]
    d = pd.concat([y.rename("y"), F], axis=1).dropna()
    out = {}
    for end in range(window, len(d) + 1):
        blk = d.iloc[end - window: end]
        out[d.index[end - 1]] = ols(blk["y"], blk[F.columns])["coef"].drop("Alpha")
    return pd.DataFrame(out).T
