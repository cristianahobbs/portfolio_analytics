"""Performance attribution.

- Return contribution by asset (what made the money).
- Brinson-Fachler: active return vs the benchmark split into allocation
  (over/underweighting groups), selection (what was held inside each group,
  e.g. single stocks vs SPY) and interaction.

Single-period effects are linked across periods with Carino smoothing, so
they add up exactly to the compounded total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .returns import period_ends


def drifting_weights(asset_returns: pd.DataFrame, weights: pd.Series,
                     rebalance: str | None = "ME") -> pd.DataFrame:
    """Beginning-of-day weights for a portfolio rebalanced to `weights` at each
    period end (None = constant mix)."""
    r = asset_returns[weights.index].fillna(0.0)
    if rebalance is None:
        return pd.DataFrame(np.tile(weights.to_numpy(), (len(r), 1)), index=r.index,
                            columns=weights.index)
    reb = set(period_ends(r.index, rebalance))
    target = weights.to_numpy(dtype=float)
    w = target.copy()
    out = np.empty((len(r), len(target)))
    for i, (dt, row) in enumerate(zip(r.index, r.to_numpy())):
        out[i] = w
        w = w * (1 + row) / (1 + w @ row)
        if dt in reb:
            w = target.copy()
    return pd.DataFrame(out, index=r.index, columns=weights.index)


def _carino_k(r: np.ndarray | float, b: np.ndarray | float = 0.0):
    r, b = np.asarray(r, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (np.log1p(r) - np.log1p(b)) / (r - b)
    return np.where(np.isclose(r, b), 1 / (1 + r), k)


def contribution(asset_returns: pd.DataFrame, bod_weights: pd.DataFrame,
                 freq: str | None = "YE") -> pd.DataFrame:
    """Each asset's contribution to portfolio return. Per period, daily
    contributions are Carino-linked so they sum to the compounded return."""
    c = bod_weights * asset_returns[bod_weights.columns].fillna(0.0)
    port = c.sum(axis=1)

    def link(block: pd.DataFrame, p: pd.Series) -> pd.Series:
        total = (1 + p).prod() - 1
        k = _carino_k(p.to_numpy())
        K = _carino_k(total)
        return block.mul(k, axis=0).sum() / K

    if freq is None:
        return link(c, port).to_frame("Contribution").T
    grp = c.index.to_period({"YE": "Y", "QE": "Q", "ME": "M"}[freq])
    return pd.DataFrame({p: link(c[grp == p], port[grp == p]) for p in grp.unique()}).T


def brinson(cfg: Config, asset_returns: pd.DataFrame, bod_weights: pd.DataFrame,
            freq: str = "ME") -> dict:
    """Brinson-Fachler by `group` (config), computed monthly and Carino-linked.

    Benchmark: group weights and proxies from config, reset monthly.
    Returns dict with 'by_group', 'by_period', 'by_asset' tables and totals."""
    groups = cfg.group
    bench = cfg["benchmark"]["groups"]
    wb = pd.Series({g: v["weight"] for g, v in bench.items()}, dtype=float)
    proxy = {g: v["proxy"] for g, v in bench.items()}
    r = asset_returns.fillna(0.0)
    period = r.index.to_period({"ME": "M", "QE": "Q"}[freq])

    rows, asset_rows, Rp, Rb = [], [], [], []
    for p in period.unique():
        m = period == p
        rp_assets = (1 + r.loc[m, cfg.tickers]).prod() - 1
        w0 = bod_weights.loc[m].iloc[0]
        # within-period drift is captured by the asset period returns, given
        # BOD weights at the start of the period and no trades inside it
        wp = w0.groupby(groups).sum().reindex(wb.index).fillna(0.0)
        contrib = (w0 * rp_assets).groupby(groups).sum().reindex(wb.index).fillna(0.0)
        rp = (contrib / wp.replace(0, np.nan))
        rb = pd.Series({g: (1 + r.loc[m, proxy[g]]).prod() - 1 for g in wb.index})
        rp = rp.fillna(rb)          # empty group: treat as holding the proxy
        rb_tot = float((wb * rb).sum())
        rp_tot = float(contrib.sum())
        alloc = (wp - wb) * (rb - rb_tot)
        sel = wb * (rp - rb)
        inter = (wp - wb) * (rp - rb)
        rows.append(pd.DataFrame({"Allocation": alloc, "Selection": sel, "Interaction": inter,
                                  "period": p}))
        # per-asset active contribution vs its group proxy: w_i (r_i - rb_g)
        g_of = groups.reindex(cfg.tickers)
        asset_rows.append((w0 * (rp_assets - rb.reindex(g_of).to_numpy())).rename(p))
        Rp.append(rp_tot)
        Rb.append(rb_tot)

    Rp, Rb = np.array(Rp), np.array(Rb)
    total_p, total_b = np.prod(1 + Rp) - 1, np.prod(1 + Rb) - 1
    k = _carino_k(Rp, Rb) / _carino_k(total_p, total_b)
    eff = pd.concat(rows)
    pmap = dict(zip(period.unique(), k))
    eff[["Allocation", "Selection", "Interaction"]] = eff[["Allocation", "Selection",
                                                          "Interaction"]].mul(eff["period"].map(pmap), axis=0)
    by_group = eff.groupby(level=0)[["Allocation", "Selection", "Interaction"]].sum()
    by_group["Total"] = by_group.sum(axis=1)
    by_group.loc["Total"] = by_group.sum()
    eff["year"] = [pp.year for pp in eff["period"]]
    by_year = eff.groupby("year")[["Allocation", "Selection", "Interaction"]].sum()
    by_year["Total"] = by_year.sum(axis=1)
    by_asset = pd.DataFrame(asset_rows).mul(k, axis=0).sum().rename("Active contribution vs group proxy")
    by_asset = pd.concat([by_asset, groups.rename("Group")], axis=1)
    return {"by_group": by_group, "by_year": by_year, "by_asset": by_asset,
            "portfolio_return": float(total_p), "benchmark_return": float(total_b),
            "active_return": float(total_p - total_b)}
