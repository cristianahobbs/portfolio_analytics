"""Drawdowns: depth, time from peak to trough, and time to recover."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return (wealth / wealth.cummax() - 1).rename("Drawdown")


def drawdown_table(r: pd.Series, top: int = 10) -> pd.DataFrame:
    """One row per drawdown episode, deepest first. Durations are in trading
    days. An episode still underwater at the end has no recovery date."""
    dd = drawdown_series(r)
    under = dd < 0
    if not under.any():
        return pd.DataFrame()
    # label consecutive underwater stretches
    episode = (under != under.shift()).cumsum()[under]
    rows = []
    for _, seg in dd[under].groupby(episode):
        start_pos = dd.index.get_loc(seg.index[0])
        peak = dd.index[start_pos - 1] if start_pos > 0 else seg.index[0]
        trough = seg.idxmin()
        end_pos = dd.index.get_loc(seg.index[-1])
        recovered = end_pos + 1 < len(dd)
        recovery = dd.index[end_pos + 1] if recovered else pd.NaT
        rows.append({
            "Peak": peak,
            "Trough": trough,
            "Recovery": recovery,
            "Depth": float(seg.min()),
            "Peak to trough (days)": int(dd.index.get_loc(trough) - dd.index.get_loc(peak)),
            "Trough to recovery (days)": (int(end_pos + 1 - dd.index.get_loc(trough))
                                          if recovered else np.nan),
            "Total duration (days)": int(end_pos + 1 - dd.index.get_loc(peak)) if recovered
            else int(len(dd) - 1 - dd.index.get_loc(peak)),
            "Recovered": recovered,
        })
    out = pd.DataFrame(rows).sort_values("Depth").head(top).reset_index(drop=True)
    out.index += 1
    return out


def time_underwater(r: pd.Series) -> float:
    """Share of days spent below a previous peak."""
    return float((drawdown_series(r) < 0).mean())
