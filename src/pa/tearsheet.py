"""One-page HTML tear sheet for an investment committee.

Usage:  python -m pa.tearsheet [--source auto|live|synthetic] [--out reports/tearsheet.html]
Print to PDF from the browser (the layout is print-friendly).
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from . import attribution as A
from . import drawdown as D
from . import limits as LIM
from . import returns as R
from . import risk as RK
from . import stress as S
from .analysis import Analysis
from .config import ROOT, load_config

BLUE, ORANGE, RED, MUTED = "#2a78d6", "#eb6834", "#e34948", "#898781"
STATUS = {"green": ("#0ca30c", "✅ OK"), "amber": ("#c98500", "⚠️ Near"), "red": ("#d03b3b", "⛔ Breach")}


def _fig_html(fig, first: bool) -> str:
    fig.update_layout(template="plotly_white", height=260, margin=dict(l=40, r=10, t=30, b=30),
                      font=dict(family="system-ui, Segoe UI, sans-serif", size=11),
                      legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                       config={"displayModeBar": False})


def _table(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.1%}") -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        cells = []
        for c, v in r.items():
            f = fmt.get(c, default)
            cells.append(f"<td>{f.format(v) if isinstance(v, (int, float)) and pd.notna(v) else html.escape(str(v))}</td>")
        rows.append(f"<tr><th>{html.escape(str(idx))}</th>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build(a: Analysis) -> str:
    cfg, port, bench, rf, P = a.cfg, a.portfolio, a.benchmark, a.rf, a.periods
    nav = cfg["nav"]
    s = R.performance_summary(port, rf, bench, P)
    sb = R.performance_summary(bench, rf, None, P)
    var95 = RK.historical_var(port.iloc[-500:])
    kpis = [("Annual return", f"{s['Annual return']:.1%}", f"bench {sb['Annual return']:.1%}"),
            ("Volatility", f"{s['Annual volatility']:.1%}", f"bench {sb['Annual volatility']:.1%}"),
            ("Sharpe", f"{s['Sharpe']:.2f}", f"bench {sb['Sharpe']:.2f}"),
            ("Max drawdown", f"{s['Max drawdown']:.1%}", f"bench {sb['Max drawdown']:.1%}"),
            ("Information ratio", f"{s['Information ratio']:.2f}", f"TE {s['Tracking error']:.1%}"),
            ("1-day VaR 95%", f"${var95 * nav / 1e6:,.2f}m", f"{var95:.2%} of NAV")]
    kpi_html = "".join(f"<div class='kpi'><div class='l'>{k}</div><div class='v'>{v}</div>"
                       f"<div class='s'>{sub}</div></div>" for k, v, sub in kpis)

    g = go.Figure()
    for name, r, col in (("Portfolio", port, BLUE), ("Benchmark", bench, ORANGE)):
        g.add_trace(go.Scatter(x=r.index, y=R.cumulative(r), name=name, line=dict(width=2, color=col)))
    g.update_layout(title="Growth of $1")
    dd = go.Figure()
    for name, r, col in (("Portfolio", port, BLUE), ("Benchmark", bench, ORANGE)):
        dd.add_trace(go.Scatter(x=r.index, y=D.drawdown_series(r), name=name, line=dict(width=1.5, color=col)))
    dd.update_layout(title="Drawdown", yaxis_tickformat=".0%")

    rc = RK.risk_contributions(cfg.weights, a.cov)
    rcf = go.Figure()
    rcf.add_trace(go.Bar(x=rc.index, y=rc["Weight"], name="Weight", marker_color=BLUE))
    rcf.add_trace(go.Bar(x=rc.index, y=rc["% of risk"], name="Share of risk", marker_color=ORANGE))
    rcf.update_layout(title="Weight vs risk contribution", barmode="group", yaxis_tickformat=".0%",
                      hovermode="closest")

    cal = R.calendar_returns(pd.DataFrame({"Portfolio": port, "Benchmark": bench})).iloc[-8:]
    cal.index = cal.index.year
    cal["Active"] = cal["Portfolio"] - cal["Benchmark"]
    hs, _ = S.historical_stress(a.asset_returns, cfg.weights, nav, bench)
    hy, _, _ = S.hypothetical_stress(cfg, a.all_returns, cfg.weights, nav, a.vix,
                                     betas=a.asset_factor_exposures)
    br = A.brinson(cfg, a.all_returns, a.bod_weights)["by_group"]
    lr = LIM.limit_report(cfg, a.current_limit_metrics())
    lim_rows = "".join(
        f"<tr><th>{k}</th><td>{r['Current']:.2%}</td><td>{r['Limit value']:.2%}</td>"
        f"<td>{r['Utilisation']:.0%}</td><td style='color:{STATUS[r['Status']][0]}'>{STATUS[r['Status']][1]}</td></tr>"
        if "Days" not in k and "beta" not in k else
        f"<tr><th>{k}</th><td>{r['Current']:.2f}</td><td>{r['Limit value']:.2f}</td>"
        f"<td>{r['Utilisation']:.0%}</td><td style='color:{STATUS[r['Status']][0]}'>{STATUS[r['Status']][1]}</td></tr>"
        for k, r in lr.iterrows())
    holdings = cfg.assets[["name", "asset_class"]].assign(Weight=cfg.weights,
                                                          **{"Share of risk": rc["% of risk"]})

    asof = port.index[-1].strftime("%d %B %Y")
    warn = ("<p class='warn'>⚠️ Synthetic data: illustrative only.</p>" if a.md.source == "synthetic" else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(cfg['name'])} – Tear sheet</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;background:#fcfcfb;margin:24px auto;max-width:1100px;padding:0 16px;font-size:13px}}
h1{{font-size:22px;margin:0}} h2{{font-size:14px;margin:18px 0 6px;border-bottom:1px solid #e1e0d9;padding-bottom:4px}}
.sub{{color:#52514e;margin:4px 0 14px}} .warn{{color:#d03b3b;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}} .kpi{{border:1px solid #e1e0d9;border-radius:8px;padding:8px 10px}}
.kpi .l{{color:#52514e;font-size:11px}} .kpi .v{{font-size:20px;font-weight:600}} .kpi .s{{color:#898781;font-size:11px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{padding:3px 6px;text-align:right;border-bottom:1px solid #eeede8}}
tbody th{{text-align:left;font-weight:500}} thead th{{color:#52514e;font-weight:500}}
.foot{{color:#898781;font-size:11px;margin-top:16px}}
@media (max-width:800px){{.kpis{{grid-template-columns:repeat(2,1fr)}} .grid{{grid-template-columns:1fr}}}}
@media print{{body{{margin:0}} .grid{{break-inside:avoid}}}}
</style></head><body>
<h1>{html.escape(cfg['name'])}</h1>
<p class="sub">As of {asof} · NAV ${nav / 1e6:,.0f}m · {a.md.source} data since {port.index[0]:%b %Y} · benchmark
{', '.join(f'{w:.0%} {t}' for t, w in cfg.benchmark_weights.items())}</p>{warn}
<div class="kpis">{kpi_html}</div>
<div class="grid"><div>{_fig_html(g, True)}</div><div>{_fig_html(dd, False)}</div></div>
<div class="grid">
 <div><h2>Holdings</h2>{_table(holdings, {'name': '{}', 'asset_class': '{}'})}</div>
 <div>{_fig_html(rcf, False)}
  <h2>Risk limits</h2><table><thead><tr><th></th><th>Current</th><th>Limit</th><th>Used</th><th>Status</th></tr></thead><tbody>{lim_rows}</tbody></table>
 </div>
</div>
<div class="grid">
 <div><h2>Calendar returns</h2>{_table(cal.sort_index(ascending=False))}</div>
 <div><h2>Attribution vs benchmark (cumulative, Brinson-Fachler)</h2>{_table(br)}</div>
</div>
<div class="grid">
 <div><h2>Historical stress (today's weights)</h2>{_table(hs[['Portfolio', 'Benchmark', 'P&L ($)']], {'P&L ($)': '${:,.0f}'})}</div>
 <div><h2>Hypothetical shocks</h2>{_table(hy, {'P&L ($), correlated': '${:,.0f}'})}</div>
</div>
<p class="foot">Total returns, gross of fees, rebalanced monthly. VaR: historical, last 500 days. Covariance: Ledoit-Wolf, 3 years.
Past performance is not a guide to future returns. Single-stock holdings were selected with hindsight. See docs/methodology.md.</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "tearsheet.html"))
    args = ap.parse_args()
    a = Analysis.load(load_config(args.config), args.source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(a), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
