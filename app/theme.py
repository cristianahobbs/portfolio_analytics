"""Chart styling: validated categorical order, thin marks, recessive axes.
Colours follow the entity (a fixed colour per name), never its rank."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#898781"
GRID = "rgba(137,135,129,0.25)"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]   # red - gray - blue
SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
STATUS = {"green": ("#0ca30c", "✅ Within limit"), "amber": ("#fab219", "⚠️ Near limit"),
          "red": ("#d03b3b", "⛔ Breach")}

# Fixed colours for recurring entities so they never repaint between charts
ENTITY = {"portfolio": SERIES[0], "benchmark": SERIES[1], "current (static)": SERIES[0],
          "max_sharpe": SERIES[1], "min_vol": SERIES[2], "risk_parity": SERIES[3],
          "hrp": SERIES[4], "equal_weight": SERIES[5], "current": SERIES[0]}

pio.templates["pa"] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13),
    colorway=SERIES,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED, zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=True, zerolinecolor=MUTED, zerolinewidth=1),
    # title at the very top, legend in its own row underneath, so they never overlap
    title=dict(x=0, xanchor="left", xref="paper", y=1, yanchor="top", yref="container",
               pad=dict(t=6), font=dict(size=15)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=12)),
    margin=dict(l=10, r=10, t=72, b=10),
    hovermode="x unified",
    bargap=0.25,
))
pio.templates.default = "plotly+pa"


def color_for(name: str, i: int = 0) -> str:
    return ENTITY.get(name, SERIES[i % len(SERIES)])


def line_fig(df, title: str = "", yfmt: str = ".0%", height: int = 380, colors: dict | None = None):
    """Line chart, one trace per column, 2px lines, unified hover."""
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        col = (colors or {}).get(c, color_for(str(c), i))
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=str(c), mode="lines",
                                 line=dict(width=2, color=col),
                                 hovertemplate=f"%{{y:{yfmt}}}<extra>{c}</extra>"))
    fig.update_layout(title=title, height=height, yaxis_tickformat=yfmt,
                      showlegend=len(df.columns) > 1)
    return fig


def bar_fig(s, title: str = "", fmt: str = ".1%", height: int = 360, horizontal: bool = True,
            signed: bool = False):
    """Single-series bar chart. signed=True colours by polarity (diverging poles)."""
    s = s.dropna()
    colors = ([SERIES[0] if v >= 0 else SERIES[7] for v in s] if signed else SERIES[0])
    kw = dict(x=s.values, y=[str(i) for i in s.index], orientation="h") if horizontal else \
        dict(x=[str(i) for i in s.index], y=s.values)
    fig = go.Figure(go.Bar(**kw, marker=dict(color=colors, line=dict(width=0)),
                           hovertemplate=f"%{{{'y' if horizontal else 'x'}}}: %{{{'x' if horizontal else 'y'}:{fmt}}}<extra></extra>"))
    fig.update_layout(title=title, height=height, hovermode="closest", showlegend=False)
    if horizontal:
        fig.update_xaxes(tickformat=fmt, gridcolor=GRID, showgrid=True)
        fig.update_yaxes(autorange="reversed", showgrid=False)
    else:
        fig.update_yaxes(tickformat=fmt)
    return fig


def grouped_bar(df, title: str = "", fmt: str = ".0%", height: int = 400, colors: dict | None = None):
    """Grouped bars: index on the x-axis, one colour per column."""
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Bar(x=[str(x) for x in df.index], y=df[c], name=str(c),
                             marker=dict(color=(colors or {}).get(c, color_for(str(c), i)),
                                         line=dict(width=0)),
                             hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra>{c}</extra>"))
    fig.update_layout(title=title, height=height, barmode="group", yaxis_tickformat=fmt,
                      hovermode="closest", bargroupgap=0.08)
    return fig


def heatmap(df, title: str = "", zmin=-1, zmax=1, fmt: str = ".2f", diverging: bool = True,
            height: int = 520):
    fig = go.Figure(go.Heatmap(
        z=df.values, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
        zmin=zmin, zmax=zmax, colorscale=DIVERGING if diverging else SEQUENTIAL,
        xgap=2, ygap=2, text=df.values, texttemplate=f"%{{text:{fmt}}}", textfont=dict(size=10),
        hovertemplate=f"%{{y}} / %{{x}}: %{{z:{fmt}}}<extra></extra>"))
    fig.update_layout(title=title, height=height, hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig
