"""Risk limits monitor: current utilisation with green/amber/red status, plus
a history of breaches for the limits that can be tracked daily."""
from __future__ import annotations

import pandas as pd

from .drawdown import drawdown_series
from .var_backtest import rolling_var_forecast


def status(value: float, limit: float, amber: float) -> str:
    u = abs(value) / limit
    return "red" if u >= 1 else "amber" if u >= amber else "green"


def limit_report(cfg, metrics: dict) -> pd.DataFrame:
    """metrics: name -> current value, for names matching keys in cfg['limits']."""
    lim = cfg["limits"]
    amber = lim.get("amber_level", 0.8)
    labels = {"var_1d_95": "1-day VaR 95%", "es_1d_975": "1-day ES 97.5%",
              "max_drawdown": "Drawdown from peak", "max_position": "Largest position",
              "equity_beta": "Equity beta", "tracking_error": "Tracking error",
              "days_to_liquidate_90pct": "Days to sell 90%"}
    rows = []
    for key, val in metrics.items():
        if key not in lim:
            continue
        rows.append({"Limit": labels.get(key, key), "Current": val, "Limit value": lim[key],
                     "Utilisation": abs(val) / lim[key], "Status": status(val, lim[key], amber)})
    return pd.DataFrame(rows).set_index("Limit")


def breach_history(cfg, port: pd.Series, window: int = 500) -> pd.DataFrame:
    """Daily utilisation of the VaR and drawdown limits through history."""
    lim = cfg["limits"]
    var = rolling_var_forecast(port, "filtered_historical", window, 0.95,
                               cfg["risk"]["ewma_lambda"])
    dd = drawdown_series(port).abs()
    out = pd.DataFrame({"VaR utilisation": var / lim["var_1d_95"],
                        "Drawdown utilisation": dd / lim["max_drawdown"]})
    return out.dropna()
