"""Interactive portfolio analytics dashboard.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
# works without `pip install -e .` (e.g. on Streamlit Community Cloud)
sys.path[:0] = [str(APP_DIR), str(APP_DIR.parent / "src")]
warnings.filterwarnings("ignore")

from pa import attribution as A  # noqa: E402
from pa import backtest as B  # noqa: E402
from pa import correlation as C  # noqa: E402
from pa import drawdown as D  # noqa: E402
from pa import factors as F  # noqa: E402
from pa import fx as FX  # noqa: E402
from pa import limits as LIM  # noqa: E402
from pa import liquidity as LQ  # noqa: E402
from pa import montecarlo as MC  # noqa: E402
from pa import optimise as O  # noqa: E402
from pa import returns as R  # noqa: E402
from pa import risk as RK  # noqa: E402
from pa import stress as S  # noqa: E402
from pa import trades as TR  # noqa: E402
from pa import var_backtest as VB  # noqa: E402
from pa.insights import build_thesis, thesis_markdown  # noqa: E402
from pa.analysis import Analysis  # noqa: E402
from pa.config import load_config  # noqa: E402
from pa.data import load_market_data  # noqa: E402
from theme import (DIVERGING, SERIES, STATUS, bar_fig, color_for, grouped_bar,  # noqa: E402
                   heatmap, line_fig)

st.set_page_config(page_title="Portfolio Analytics", page_icon="📊", layout="wide")


# =============================================================================
# Data & caching
# =============================================================================
@st.cache_resource(show_spinner="Loading market data…")
def market(source: str):
    return load_market_data(load_config(), source)


@st.cache_resource(show_spinner="Computing analytics…")
def analysis(weights_key: tuple, source: str, nav: float) -> Analysis:
    base = load_config()
    cfg = base.with_weights(pd.Series(dict(weights_key)))
    cfg.raw["nav"] = nav
    return Analysis(cfg, market(source))


@st.cache_resource(show_spinner="Running walk-forward backtests (about 20 seconds the first time)…")
def backtests(weights_key, source, nav, rebalance, cost_bps, lookback):
    a = analysis(weights_key, source, nav)
    return B.run_strategies(a.cfg, a.asset_returns, rebalance=rebalance, cost_bps=cost_bps,
                            lookback=lookback)


@st.cache_data(show_spinner="Simulating…")
def monte_carlo(weights_key, source, nav, model, horizon, n_paths, seed, t_dof, jump_intensity):
    a = analysis(weights_key, source, nav)
    sim = MC.simulate(a.asset_returns.iloc[-2520:], a.cfg.weights, model, horizon, n_paths, seed,
                      t_dof, jump_intensity, periods=a.periods)
    out = MC.summarise(sim, nav)
    return out["stats"], out["fan"], out["terminal"]


@st.cache_resource(show_spinner="Writing the investment thesis…")
def thesis(weights_key, source, nav):
    return build_thesis(analysis(weights_key, source, nav))


@st.cache_resource(show_spinner="Running cost sensitivity…")
def cost_sens(weights_key, source, nav, method):
    a = analysis(weights_key, source, nav)
    return B.cost_sensitivity(a.cfg, a.asset_returns, B.optimised_strategy(a.cfg, method), rf=a.rf)


@st.cache_resource(show_spinner=False)
def slow(name: str, weights_key, source, nav, *args):
    """Cache for slower per-portfolio calculations, so widget changes don't recompute them."""
    a = analysis(weights_key, source, nav)
    cfg, port, bench = a.cfg, a.portfolio, a.benchmark
    if name == "sharpe_ci":
        return R.sharpe_confidence_interval(port, a.rf, n_boot=500, periods=a.periods)
    if name == "brinson":
        return A.brinson(cfg, a.all_returns, a.bod_weights)
    if name == "factor_decomp":
        return F.factor_risk_decomposition(cfg, a.all_returns, port)
    if name == "factor_decomp_active":
        return F.factor_risk_decomposition(cfg, a.all_returns, (port - bench).rename("Active"))
    if name == "factor_exposures":
        return F.exposures(cfg, a.all_returns, port.rename("Portfolio"))
    if name == "rolling_betas":
        return F.rolling_betas(cfg, a.all_returns, port, 104)
    if name == "frontier":
        mu_method, = args
        cons = O.Constraints.from_config(cfg)
        mu = O.expected_returns(cfg, a.excess_returns.iloc[-756:], a.cov, mu_method)
        return O.efficient_frontier(mu, a.cov, cons, 25)
    if name == "hyp_stress":
        return S.hypothetical_stress(cfg, a.all_returns, cfg.weights, nav, a.vix,
                                     betas=a.asset_factor_exposures)
    if name == "rebalance_cmp":
        return B.rebalance_comparison(cfg, a.asset_returns, cfg.weights, a.rf)
    raise ValueError(name)


@st.cache_data(show_spinner="Fitting regime model…")
def hmm(weights_key, source, nav):
    a = analysis(weights_key, source, nav)
    res = C.fit_hmm_2state(a.portfolio)
    return {k: v for k, v in res.items()}


@st.cache_data(show_spinner="Re-optimising under perturbed inputs…")
def sensitivity(weights_key, source, nav, method, bump, n_draws):
    a = analysis(weights_key, source, nav)
    cons = O.Constraints.from_config(a.cfg)
    win = a.excess_returns.iloc[-756:]
    mu = O.expected_returns(a.cfg, win, a.cov)
    sens = O.return_sensitivity(mu, a.cov, cons, bump, method, a.cfg.weights.to_numpy())
    res = O.resampled_weights(mu, a.cov, cons, len(win), n_draws, method)
    return sens, res


# =============================================================================
# Formatting helpers
# =============================================================================
PCT_ROWS = {"Annual return", "Annual volatility", "Max drawdown", "Hit rate", "Best day",
            "Worst day", "Active return", "Tracking error", "Alpha (ann.)", "Prob. Sharpe > 0",
            "Annual turnover", "Annual cost drag", "Exp. excess return", "Volatility",
            "Max risk contribution", "Up capture", "Down capture"}


def fmt_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise formatting of a metrics table (rows = metrics) as strings."""
    out = df.copy().astype(object)
    for r in df.index:
        for c in df.columns:
            v = df.loc[r, c]
            if isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                out.loc[r, c] = f"{v:.1%}" if r in PCT_ROWS else f"{v:.2f}"
    return out


def show(df, pct=(), money=(), num=(), days=(), height="auto", **kw):
    fmt = {c: "{:.1%}" for c in pct if c in df}
    fmt.update({c: (lambda v: f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}") for c in money if c in df})
    fmt.update({c: "{:.2f}" for c in num if c in df})
    fmt.update({c: "{:.1f}" for c in days if c in df})
    st.dataframe(df.style.format(fmt, na_rep="–"), height=height, width="stretch", **kw)


def kpi(col, label, value, help=None, delta=None):
    col.metric(label, value, delta=delta, help=help)


def plot(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# =============================================================================
# Sidebar: portfolio definition
# =============================================================================
base_cfg = load_config()
st.sidebar.title("📊 Portfolio")
source = st.sidebar.selectbox("Data", ["auto", "synthetic", "live"], index=0,
                              help="auto = Yahoo Finance + FRED, falling back to a synthetic market offline")
nav = st.sidebar.number_input("NAV ($)", value=float(base_cfg["nav"]), step=1e7, format="%.0f")

st.sidebar.subheader("Holdings")
edit = base_cfg.assets[["name", "asset_class", "weight"]].copy()
edit["weight"] = (edit["weight"] * 100).round(2)
edited = st.sidebar.data_editor(
    edit, column_config={"weight": st.column_config.NumberColumn("Weight %", min_value=-100.0,
                                                                 max_value=100.0, step=0.5),
                         "name": st.column_config.TextColumn("Name", disabled=True),
                         "asset_class": st.column_config.TextColumn("Class", disabled=True)},
    width="stretch", height=460)
total = edited["weight"].sum()
weights = edited["weight"] / 100
if abs(total - 100) > 0.01:
    st.sidebar.warning(f"Weights sum to {total:.1f}%. They are rescaled to 100% for the analysis.")
    weights = weights / weights.sum()
weights_key = tuple((t, round(float(w), 6)) for t, w in weights.items())

a = analysis(weights_key, source, nav)
cfg = a.cfg
if a.md.source == "synthetic":
    st.warning("Showing **synthetic** market data (live download unavailable or not selected). "
               "Numbers illustrate the method, not real history.", icon="⚠️")

port, bench, rf = a.portfolio, a.benchmark, a.rf
P = a.periods
st.title(cfg["name"])
st.caption(f"{a.md.source.title()} data · {port.index[0]:%d %b %Y} – {port.index[-1]:%d %b %Y} · "
           f"Benchmark: " + ", ".join(f"{w:.0%} {t}" for t, w in cfg.benchmark_weights.items())
           + f" · rebalanced {cfg['backtest']['rebalance'].replace('ME', 'monthly')}")

TAB_NAMES = ["Thesis", "Overview", "Active risk", "Benchmark & attribution", "Risk", "Drawdowns",
             "Correlations & regimes", "Factors", "Optimisation", "Trade proposal", "Backtest",
             "Stress tests", "Monte Carlo", "Liquidity & limits", "FX", "Methodology"]
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# =============================================================================
# 0. Investment thesis
# =============================================================================
with tabs["Thesis"]:
    th = thesis(weights_key, source, nav)
    s0 = R.performance_summary(port, rf, bench, P)
    c = st.columns(5)
    kpi(c[0], "Annual return", f"{s0['Annual return']:.1%}", delta=f"{s0['Active return']:+.1%} vs benchmark")
    kpi(c[1], "Sharpe", f"{s0['Sharpe']:.2f}")
    kpi(c[2], "Ex-ante tracking error", f"{a.active['tracking_error']:.1%}")
    kpi(c[3], "Beta to benchmark", f"{a.active['beta_to_benchmark']:.2f}")
    kpi(c[4], "Worst crisis (replayed)", f"{S.historical_stress(a.asset_returns, cfg.weights, nav)[0]['Portfolio'].astype(float).min():.1%}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(thesis_markdown(th, "###"))
    with right:
        st.subheader("Trailing returns")
        tr = R.trailing_returns(pd.DataFrame({"Portfolio": port, "Benchmark": bench}), P)
        tr["Active"] = tr["Portfolio"] - tr["Benchmark"]
        show(tr, pct=tr.columns)
        st.subheader("Where the active risk is")
        tet = a.active["table"]["% of TE"].sort_values()
        plot(bar_fig(tet[tet.abs() > 0.005], "Share of tracking error", signed=True, height=380))
        st.caption("See the *Active risk* and *Trade proposal* tabs for details.")

# =============================================================================
# 1. Overview
# =============================================================================
with tabs["Overview"]:
    summ = R.performance_summary(port, rf, bench, P)
    c = st.columns(6)
    kpi(c[0], "Annual return", f"{summ['Annual return']:.1%}",
        delta=f"{summ['Active return']:+.1%} vs benchmark")
    kpi(c[1], "Volatility", f"{summ['Annual volatility']:.1%}")
    kpi(c[2], "Sharpe", f"{summ['Sharpe']:.2f}")
    kpi(c[3], "Max drawdown", f"{summ['Max drawdown']:.1%}")
    kpi(c[4], "Information ratio", f"{summ['Information ratio']:.2f}")
    kpi(c[5], "1-day VaR 95%", f"${RK.historical_var(port.iloc[-500:]) * nav:,.0f}",
        help="Historical, last 500 days")

    show_real = a.md.cpi is not None
    series = {"portfolio": port, "portfolio (net of fees)": a.portfolio_net, "benchmark": bench}
    if show_real:
        series["portfolio (real, net)"] = R.real_returns(a.portfolio_net, a.md.cpi)
    growth = pd.DataFrame({k: R.cumulative(v) for k, v in series.items()})
    plot(line_fig(growth, "Growth of $1", yfmt=".2f", height=420,
                  colors={"portfolio": SERIES[0], "portfolio (net of fees)": SERIES[2],
                          "benchmark": SERIES[1], "portfolio (real, net)": SERIES[6]}))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Performance summary")
        table = pd.DataFrame({"Portfolio": summ, "Benchmark": R.performance_summary(bench, rf, None, P)})
        st.dataframe(fmt_summary(table), width="stretch", height=560)
    with right:
        st.subheader("Is the Sharpe ratio real?")
        lo, hi = slow("sharpe_ci", weights_key, source, nav)
        st.markdown(f"**Sharpe {summ['Sharpe']:.2f}**, 95% bootstrap interval **[{lo:.2f}, {hi:.2f}]** "
                    "(stationary block bootstrap, which keeps volatility clustering).")
        st.markdown(f"Probabilistic Sharpe ratio: **{R.probabilistic_sharpe(port, 0, rf, P):.1%}** "
                    "chance that the true Sharpe is above zero, after adjusting for sample length, skew "
                    "and fat tails.")
        st.markdown(f"Probability it beats the benchmark's Sharpe "
                    f"({R.sharpe(bench, rf, P):.2f}): **{R.probabilistic_sharpe(port, R.sharpe(bench, rf, P), rf, P):.1%}**")
        st.caption("For the Deflated Sharpe Ratio, which corrects for testing several strategies, "
                   "see the Backtest tab.")
        st.subheader("Trailing returns")
        trr = R.trailing_returns(pd.DataFrame({"Portfolio": port, "Benchmark": bench}), P)
        trr["Active"] = trr["Portfolio"] - trr["Benchmark"]
        show(trr, pct=trr.columns)
        st.subheader("Calendar-year returns")
        cal = R.calendar_returns(pd.DataFrame({"Portfolio": port, "Benchmark": bench}))
        cal.index = cal.index.year
        cal["Active"] = cal["Portfolio"] - cal["Benchmark"]
        show(cal.sort_index(ascending=False), pct=cal.columns, height=320)

    st.subheader("Rolling 1-year metrics")
    roll = R.rolling_metrics(port, 252, rf, bench, P)
    c1, c2 = st.columns(2)
    with c1:
        plot(line_fig(roll[["Return"]].join(R.rolling_metrics(bench, 252, rf, None, P)[["Return"]]
                                             .rename(columns={"Return": "Benchmark"}))
                      .rename(columns={"Return": "portfolio", "Benchmark": "benchmark"}),
                      "Rolling 1y return", height=320))
    with c2:
        plot(line_fig(roll[["Sharpe"]].rename(columns={"Sharpe": "portfolio"}), "Rolling 1y Sharpe",
                      yfmt=".1f", height=320))

# =============================================================================
# Active risk
# =============================================================================
with tabs["Active risk"]:
    ar = a.active
    at_ = ar["table"]
    realised_te = (port - bench).iloc[-252:].std() * np.sqrt(P)
    c = st.columns(5)
    kpi(c[0], "Ex-ante tracking error", f"{ar['tracking_error']:.2%}",
        help="sqrt(a'Σa), a = active weights; Ledoit-Wolf covariance, last 3 years")
    kpi(c[1], "Realised TE (1 year)", f"{realised_te:.2%}")
    kpi(c[2], "Beta to benchmark", f"{ar['beta_to_benchmark']:.2f}")
    kpi(c[3], "Active share", f"{at_['Active weight'].abs().sum() / 2:.0%}",
        help="Half the sum of |active weights|: how different the holdings are from the benchmark")
    kpi(c[4], "Alpha needed for IR 0.5", f"{0.5 * ar['tracking_error']:.2%} a year",
        help="An information ratio of 0.5 is good. This is the outperformance the active risk has to earn")
    st.markdown("Tracking error measures how far the portfolio can drift from the benchmark. "
                "Each position's **contribution** = active weight × its marginal effect on TE, so the "
                "contributions add up to the total. A position can matter more than its active weight "
                "suggests if it is volatile, or less if it hedges other bets.")

    srt = at_.sort_values("% of TE", ascending=False)
    plot(grouped_bar(srt[["Active weight", "% of TE"]].rename(columns={"% of TE": "Share of TE"}),
                     "Active weight vs share of tracking error", height=400,
                     colors={"Active weight": SERIES[0], "Share of TE": SERIES[1]}))
    c1, c2 = st.columns([3, 2])
    with c1:
        big = (at_["Active weight"].abs() > 0.04) | (at_["% of TE"].abs() > 0.04)
        fig = go.Figure(go.Scatter(
            x=at_["Active weight"], y=at_["% of TE"], mode="markers+text",
            text=[t if b else "" for t, b in zip(at_.index, big)], customdata=at_.index,
            textposition="top center", marker=dict(size=11, color=SERIES[0], line=dict(width=2, color="white")),
            hovertemplate="%{customdata}: active %{x:+.1%}, %{y:.1%} of TE<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#898781", width=1))
        fig.update_layout(height=420, hovermode="closest", title="Bets map: size of bet vs risk it adds",
                          xaxis_title="Active weight", yaxis_title="Share of tracking error",
                          xaxis_tickformat="+.0%", yaxis_tickformat=".0%")
        plot(fig)
    with c2:
        show(ar["by_group"].sort_values("% of TE", ascending=False),
             pct=["Active weight", "Contribution to TE", "% of TE"])
        show(srt, pct=["Portfolio weight", "Benchmark weight", "Active weight", "Contribution to TE", "% of TE"],
             height=380)
        st.caption("Contribution to TE is in percentage points of tracking error; the column sums to the total.")

    st.subheader("What drives the active risk: factors")
    active_r = (port - bench).rename("Active")
    fa = slow("factor_decomp_active", weights_key, source, nav)
    c1, c2 = st.columns(2)
    with c1:
        plot(bar_fig(fa["% of variance"], "Share of active variance by factor", height=320))
    with c2:
        plot(bar_fig(fa["Beta"].dropna(), "Active factor betas (portfolio − benchmark)", fmt=".2f",
                     signed=True, height=320))
    st.caption("Weekly regression of portfolio-minus-benchmark returns on the factor proxies. A negative "
               "equity beta here means the portfolio is structurally less exposed to equities than the benchmark.")

    st.subheader("Ex-ante vs realised")
    roll_te = pd.DataFrame({"Realised (63-day)": (port - bench).rolling(63).std() * np.sqrt(P),
                            "Realised (1-year)": (port - bench).rolling(252).std() * np.sqrt(P)})
    fig = line_fig(roll_te, "Rolling realised tracking error", height=320,
                   colors={"Realised (63-day)": SERIES[0], "Realised (1-year)": SERIES[2]})
    fig.add_hline(y=ar["tracking_error"], line=dict(color=SERIES[1], width=2, dash="dash"),
                  annotation_text=f"Ex-ante today {ar['tracking_error']:.1%}")
    plot(fig)

    st.subheader("What if…")
    c1, c2, c3 = st.columns(3)
    w_asset = c1.selectbox("Change the weight of", cfg.tickers, key="te_asset")
    w_delta = c2.slider("By (% of NAV)", -15.0, 15.0, 5.0, 0.5, key="te_delta") / 100
    w_fund = c3.selectbox("Funded from", ["pro-rata"] + cfg.tickers,
                          index=1 + cfg.tickers.index(cfg.cash_ticker), key="te_fund")
    new_w = RK.incremental_var(cfg.weights, a.cov_daily, w_asset, w_delta,
                               None if w_fund == "pro-rata" else w_fund)["new_weights"]
    new_ar = RK.active_risk(new_w, cfg.benchmark_weights, a.cov_union)
    c = st.columns(3)
    kpi(c[0], "Tracking error", f"{new_ar['tracking_error']:.2%}",
        delta=f"{(new_ar['tracking_error'] - ar['tracking_error']) * 1e4:+.0f} bp", )
    kpi(c[1], "Portfolio volatility", f"{new_ar['portfolio_vol']:.2%}",
        delta=f"{(new_ar['portfolio_vol'] - ar['portfolio_vol']) * 1e4:+.0f} bp")
    kpi(c[2], "Beta to benchmark", f"{new_ar['beta_to_benchmark']:.2f}",
        delta=f"{new_ar['beta_to_benchmark'] - ar['beta_to_benchmark']:+.2f}")

# =============================================================================
# Trade proposal
# =============================================================================
with tabs["Trade proposal"]:
    st.markdown("From the current portfolio to an optimised target: the trade list, the cost, and what "
                "changes in risk. Targets use Black-Litterman returns and Ledoit-Wolf risk under the config "
                "constraints.")
    c1, c2, c3, c4 = st.columns(4)
    tmethod = c1.selectbox("Target", ["max_sharpe", "min_vol", "risk_parity", "hrp"],
                           format_func={"max_sharpe": "Maximum Sharpe", "min_vol": "Minimum volatility",
                                        "risk_parity": "Risk parity", "hrp": "HRP"}.get)
    move = c2.slider("Move this far towards target", 10, 100, 100, 10, format="%d%%") / 100
    min_tr = c3.select_slider("Skip trades smaller than", [0.0, 0.0025, 0.005, 0.01],
                              value=0.0025, format_func="{:.2%}".format)
    tcost = c4.select_slider("Commission / impact (bps)", [0, 5, 10, 25], value=cfg["backtest"]["cost_bps"])
    cons_t = O.Constraints.from_config(cfg)
    mu_t = O.expected_returns(cfg, a.excess_returns.iloc[-756:], a.cov)
    full_target = O.optimise(tmethod, mu_t, a.cov, cons_t, cfg.weights.to_numpy())
    target = cfg.weights + move * (full_target - cfg.weights)
    tp = TR.trade_proposal(cfg.weights, target, a.cov_union, cfg.benchmark_weights, nav, tcost,
                           cfg.assets["spread_bps"], min_tr, cfg.cash_ticker, a.equity_betas, mu_t,
                           cfg.assets["name"])
    sm = tp["summary"]
    c = st.columns(5)
    kpi(c[0], "Trades", f"{sm['Number of trades']}")
    kpi(c[1], "One-way turnover", f"{sm['One-way turnover']:.1%}")
    kpi(c[2], "Estimated cost", f"${sm['Total cost ($)']:,.0f}", help="Commission/impact + half the bid-ask spread")
    kpi(c[3], "Cost (bps of NAV)", f"{sm['Cost (bps of NAV)']:.1f}")
    yrs = sm.get("Years to recover cost", np.inf)
    kpi(c[4], "Payback (model forecasts)", "never" if not np.isfinite(yrs) else
        f"{yrs * 12:.1f} months", help="Cost ÷ gain in expected excess return. Only as good as the forecasts")
    viol = cons_t.violations(tp["target"])
    if viol:
        st.warning("Target breaches constraints: " + "; ".join(viol), icon="⚠️")

    c1, c2 = st.columns([3, 2])
    with c1:
        tt = tp["trades"]
        show(tt, pct=["Current", "Target", "Trade"], money=["Trade ($)", "Est. cost ($)"], height=600)
        st.download_button("Download trade list (CSV)", tt.to_csv().encode("utf-8"),
                           file_name=f"trades_{tmethod}.csv", mime="text/csv")
    with c2:
        mt = tp["metrics"].copy()
        money_rows = [r for r in mt.index if "($)" in r]
        num_rows = ["Beta to benchmark", "Effective N", "Equity beta", "Exp. Sharpe"]
        disp = mt.astype(object)
        for r in mt.index:
            f = ("${:,.0f}" if r in money_rows else "{:.2f}" if r in num_rows else "{:.2%}")
            disp.loc[r] = [f.format(v) if r not in money_rows or k != "Change" else f"{v:+,.0f}"
                           for k, v in mt.loc[r].items()]
        st.dataframe(disp, width="stretch", height=420)
    cmp_w = pd.DataFrame({"Current": cfg.weights, "Target": tp["target"]})
    plot(grouped_bar(cmp_w, "Current vs target weights", height=380,
                     colors={"Current": SERIES[0], "Target": SERIES[1]}))
    st.caption("Optimised weights are sensitive to expected-return inputs (see Optimisation → sensitivity). "
               "Moving part of the way is a common way to limit that risk and the cost.")

# =============================================================================
# 2. Benchmark & attribution
# =============================================================================
with tabs["Benchmark & attribution"]:
    act = R.active_stats(port, bench, P)
    c = st.columns(5)
    kpi(c[0], "Active return (ann.)", f"{act['Active return']:+.2%}")
    kpi(c[1], "Tracking error", f"{act['Tracking error']:.2%}")
    kpi(c[2], "Information ratio", f"{act['Information ratio']:.2f}")
    kpi(c[3], "Beta", f"{act['Beta']:.2f}")
    up, dn = R.capture_ratios(port, bench)
    kpi(c[4], "Up / down capture", f"{up:.0%} / {dn:.0%}",
        help="Share of the benchmark's average up (down) month captured")

    c1, c2 = st.columns(2)
    with c1:
        plot(line_fig(roll[["Tracking error"]].rename(columns={"Tracking error": "portfolio"}),
                      "Rolling 1y tracking error", height=300))
    with c2:
        plot(line_fig(roll[["Beta"]].rename(columns={"Beta": "portfolio"}), "Rolling 1y beta to benchmark",
                      yfmt=".2f", height=300))

    st.subheader("Brinson-Fachler attribution")
    br = slow("brinson", weights_key, source, nav)
    st.markdown(f"Cumulative portfolio **{br['portfolio_return']:.1%}** vs benchmark "
                f"**{br['benchmark_return']:.1%}**, active **{br['active_return']:+.1%}**, split into "
                "*allocation* (over- or underweighting groups), *selection* (what was held inside each "
                "group vs that group's benchmark proxy) and *interaction*. The effects are Carino-linked "
                "across months, so they add up exactly to the active return.")
    c1, c2 = st.columns([1, 1])
    with c1:
        bg = br["by_group"]
        show(bg, pct=bg.columns)
        st.caption("A group with zero benchmark weight (e.g. Real Assets) shows its selection under "
                   "*interaction*. Selection + interaction = the group's total selection effect.")
    with c2:
        plot(grouped_bar(bg.drop("Total")[["Allocation", "Selection", "Interaction"]],
                         "Effects by group (cumulative)", fmt=".0%", height=340))
    st.subheader("Security selection: each holding vs its group proxy")
    st.caption("w × (holding return − group proxy return), linked. For the single stocks and "
               "international ETFs this shows whether they beat simply holding SPY.")
    ba = br["by_asset"].iloc[:, 0].sort_values()
    plot(bar_fig(ba, "Active contribution vs group proxy (cumulative)", signed=True, height=480))
    st.subheader("Attribution by year")
    show(br["by_year"].sort_index(ascending=False), pct=br["by_year"].columns, height=300)

    st.subheader("Return contribution by holding")
    contrib = A.contribution(a.asset_returns, a.bod_weights, "YE")
    contrib.index = [p.year for p in contrib.index]
    contrib["Portfolio"] = contrib.sum(axis=1)
    show(contrib.sort_index(ascending=False), pct=contrib.columns, height=420)

# =============================================================================
# 3. Risk
# =============================================================================
with tabs["Risk"]:
    win = st.select_slider("Estimation window", options=[250, 500, 750, 1250, len(port)], value=500,
                           format_func=lambda x: "Full history" if x == len(port) else f"{x} days")
    recent = port.iloc[-win:]
    st.subheader("Value at Risk and Expected Shortfall (1 day)")
    vt = RK.var_table(recent, (0.95, 0.99), 1, nav).reset_index()
    vt["Confidence"] = vt["Confidence"].map("{:.0%}".format)
    c1, c2 = st.columns([3, 2])
    with c1:
        show(vt.set_index(["Method", "Confidence"]), pct=["VaR", "ES"], money=["VaR ($)", "ES ($)"])
    with c2:
        st.markdown("**Expected Shortfall** is the average loss *on the days VaR is breached*: it "
                    "answers \"how bad is bad?\". Compare Normal with Historical and Student-t at 99%: "
                    "the normal distribution understates the tail.")
        h10 = RK.historical_var(recent, 0.99) * np.sqrt(10)
        st.metric("10-day 99% VaR (√10 scaling)", f"${h10 * nav:,.0f}",
                  help="Square-root-of-time scaling assumes independent days, so it understates risk "
                       "when volatility clusters")

    st.subheader("Does the VaR model work? Backtest")
    alpha = st.radio("Confidence", [0.95, 0.99], horizontal=True, format_func="{:.0%}".format)
    vb_summ, vb_paths = VB.backtest_var(port, window=cfg["risk"]["var_window"], alpha=alpha,
                                        lam=cfg["risk"]["ewma_lambda"])
    show(vb_summ[["Breaches", "Expected", "Breach rate", "Kupiec p-value",
                  "P(breach | breach yesterday)", "Christoffersen p-value", "Cond. coverage p-value",
                  "Verdict"]].astype({"Breaches": int}),
         pct=["Breach rate", "P(breach | breach yesterday)"],
         num=["Expected", "Kupiec p-value", "Christoffersen p-value", "Cond. coverage p-value"])
    st.caption("Kupiec: is the breach *count* right? Christoffersen: are breaches *independent*, or "
               "do they cluster because the model is slow to react? p < 0.05 = reject the model.")
    vm = st.selectbox("Show method", list(vb_summ.index), index=3)
    pp = vb_paths[["Return", vm]].dropna()
    breach = pp[pp["Return"] < -pp[vm]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pp.index, y=pp["Return"], name="Daily return", mode="lines",
                             line=dict(width=1, color="rgba(137,135,129,0.6)")))
    fig.add_trace(go.Scatter(x=pp.index, y=-pp[vm], name=f"−VaR ({vm})", mode="lines",
                             line=dict(width=2, color=SERIES[0])))
    fig.add_trace(go.Scatter(x=breach.index, y=breach["Return"], name="Breach", mode="markers",
                             marker=dict(size=8, color=SERIES[7], line=dict(width=2, color="white"))))
    fig.update_layout(height=380, yaxis_tickformat=".1%", title="Returns vs VaR forecast")
    plot(fig)

    st.subheader("Where does the risk come from?")
    rc = RK.risk_contributions(cfg.weights, a.cov)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = grouped_bar(rc[["Weight", "% of risk"]].rename(columns={"Weight": "Weight",
                                                                      "% of risk": "Share of risk"}),
                          "Weight vs share of portfolio volatility", height=380,
                          colors={"Weight": SERIES[0], "Share of risk": SERIES[1]})
        plot(fig)
    with c2:
        show(rc[["Weight", "Standalone vol", "% of risk", "Risk / weight"]].sort_values("% of risk",
                                                                                        ascending=False),
             pct=["Weight", "Standalone vol", "% of risk"], num=["Risk / weight"], height=380)
    st.caption("Ledoit-Wolf covariance, last 3 years. Risk/weight > 1 means the holding contributes "
               "more risk than its weight suggests.")

    st.subheader("Component and incremental VaR")
    cv = RK.component_var(cfg.weights, a.cov_daily, 0.95, nav)
    c1, c2 = st.columns([3, 2])
    with c1:
        show(cv.sort_values("Component VaR ($)", ascending=False),
             pct=["Weight", "Marginal VaR", "Component VaR", "% of VaR"], money=["Component VaR ($)"],
             height=360)
    with c2:
        st.markdown("**Pre-trade check: what if I add to a position?**")
        asset = st.selectbox("Add to", cfg.tickers, index=0)
        delta = st.slider("Size (% of NAV)", -10.0, 10.0, 2.0, 0.5) / 100
        fund = st.selectbox("Funded from", ["pro-rata from all others"] + cfg.tickers,
                            index=1 + cfg.tickers.index(cfg.cash_ticker))
        iv = RK.incremental_var(cfg.weights, a.cov_daily, asset, delta,
                                None if fund.startswith("pro") else fund)
        st.metric("Incremental 1-day VaR 95%", f"${iv['Incremental VaR'] * nav:+,.0f}",
                  delta=f"{iv['Incremental VaR'] / iv['VaR before']:+.1%} of current VaR",
                  delta_color="inverse")

    st.subheader("Volatility: forecasts and targeting")
    ev = RK.ewma_vol(port, cfg["risk"]["ewma_lambda"], P)
    gv, gfc, gpar = RK.garch_vol(port, P)
    realised = port.rolling(21).std().shift(-21) * np.sqrt(P)
    vols = pd.DataFrame({"EWMA forecast": ev, "GARCH(1,1) in-sample": gv,
                         "Realised next 21 days": realised}).dropna(how="all")
    plot(line_fig(vols, "Forecast vs realised volatility", height=360,
                  colors={"EWMA forecast": SERIES[0], "GARCH(1,1) in-sample": SERIES[2],
                          "Realised next 21 days": "rgba(137,135,129,0.8)"}))
    c = st.columns(4)
    kpi(c[0], "EWMA vol (today)", f"{ev.iloc[-1]:.1%}")
    kpi(c[1], "GARCH next-day forecast", f"{gfc:.1%}")
    kpi(c[2], "GARCH persistence", f"{gpar['persistence']:.3f}",
        help="alpha + beta: closer to 1 means volatility shocks fade more slowly")
    kpi(c[3], "Long-run vol (full sample)", f"{R.ann_vol(port, P):.1%}")
    target = st.slider("Volatility target", 0.04, 0.15, float(cfg["risk"]["vol_target"]), 0.01,
                       format="%.2f")
    vt_r, lev = RK.vol_target(port, target, cfg["risk"]["ewma_lambda"], cfg["risk"]["max_leverage"], P, rf)
    cmp = pd.DataFrame({"Original": R.performance_summary(port, rf, None, P),
                        "Vol-targeted": R.performance_summary(vt_r, rf, None, P)}).loc[
        ["Annual return", "Annual volatility", "Sharpe", "Max drawdown", "Excess kurtosis"]]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(fmt_summary(cmp), width="stretch")
    with c2:
        plot(line_fig(lev.to_frame("Leverage"), "Vol-target exposure (uses yesterday's forecast)",
                      yfmt=".2f", height=280))

    st.subheader("Exposures and rate/credit sensitivities")
    c1, c2 = st.columns([1, 2])
    with c1:
        ex = RK.exposures(cfg.weights, a.equity_betas)
        st.dataframe(ex.to_frame("Exposure").style.format("{:.1%}"), width="stretch")
        st.caption("Beta-adjusted net = Σ weight × equity beta: the portfolio's equivalent SPY exposure.")
    with c2:
        sens = RK.rate_credit_sensitivities(cfg.weights, cfg.assets["duration"],
                                            cfg.assets["spread_duration"], nav)
        sens = sens[(sens["Duration"] > 0) | (sens.index == "Total")]
        show(sens, money=["Position ($)", "DV01 ($)", "CS01 ($)"], num=["Duration", "Spread duration"])
        st.caption("DV01/CS01: $ change for a 1bp rise in yields / credit spreads. Durations are "
                   "approximate inputs from the config.")

# =============================================================================
# 4. Drawdowns
# =============================================================================
with tabs["Drawdowns"]:
    dd = pd.DataFrame({"portfolio": D.drawdown_series(port), "benchmark": D.drawdown_series(bench)})
    plot(line_fig(dd, "Drawdown from previous peak", height=380))
    c = st.columns(3)
    kpi(c[0], "Max drawdown", f"{dd['portfolio'].min():.1%}", delta=f"benchmark {dd['benchmark'].min():.1%}",
        )
    kpi(c[1], "Time spent below a previous peak", f"{D.time_underwater(port):.0%}")
    kpi(c[2], "Current drawdown", f"{dd['portfolio'].iloc[-1]:.1%}")
    st.subheader("Worst drawdowns")
    dt = D.drawdown_table(port, 10)
    for col in ("Peak", "Trough", "Recovery"):
        dt[col] = pd.to_datetime(dt[col]).dt.strftime("%d %b %Y").fillna("not yet")
    show(dt, pct=["Depth"], days=["Trough to recovery (days)"])
    st.caption("Durations in trading days. Recovery = first day back at the previous peak.")

# =============================================================================
# 5. Correlations & regimes
# =============================================================================
with tabs["Correlations & regimes"]:
    freq = st.radio("Return frequency", ["Weekly", "Daily"], horizontal=True,
                    help="Weekly reduces the bias from markets closing at different times")
    rr = C.to_weekly(a.asset_returns) if freq == "Weekly" else a.asset_returns
    plot(heatmap(rr.corr(), f"Correlation matrix ({freq.lower()}, full history)"))

    st.subheader("Does diversification hold up in a crisis?")
    if a.vix is not None:
        vix = a.vix.resample("W-FRI").last() if freq == "Weekly" else a.vix
        rg = C.regime_correlations(rr, vix, 0.8)
        c1, c2 = st.columns(2)
        with c1:
            plot(heatmap(rg["calm"], f"Calm (VIX ≤ {rg['threshold']:.0f})", height=480))
        with c2:
            plot(heatmap(rg["stressed"], f"Stressed (VIX > {rg['threshold']:.0f})", height=480))
        diff = rg["stressed"] - rg["calm"]
        plot(heatmap(diff, "Change in correlation: stressed − calm", zmin=-0.6, zmax=0.6, height=480))
        st.caption("Regime = previous period's VIX above its 80th percentile, so it is known in advance. "
                   "Blue cells became more correlated in stress.")

    st.subheader("Rolling correlations")
    ref = st.selectbox("Correlation with", cfg.tickers, index=0)
    picks = st.multiselect("Assets", [t for t in cfg.tickers if t != ref],
                           default=[t for t in ["TLT", "HYG", "GLD"] if t != ref])
    rc_ = C.rolling_correlation(a.asset_returns, ref, 126)[picks]
    plot(line_fig(rc_, f"6-month rolling correlation with {ref}", yfmt=".1f", height=360))

    st.subheader("Two-regime model (hidden Markov model)")
    h = hmm(weights_key, source, nav)
    c = st.columns(4)
    kpi(c[0], "Calm-regime vol", f"{h['vol'][0] * np.sqrt(P):.1%}")
    kpi(c[1], "Stressed-regime vol", f"{h['vol'][1] * np.sqrt(P):.1%}")
    kpi(c[2], "Average calm spell", f"{h['expected_duration_days'][0]:.0f} days")
    kpi(c[3], "P(stressed regime) today", f"{h['filtered'].iloc[-1]:.0%}",
        help="Real-time (filtered) probability, using data up to today only")
    plot(line_fig(pd.DataFrame({"Smoothed (hindsight)": h["smoothed"], "Real-time": h["filtered"]}),
                  "Probability of the high-volatility regime", height=320,
                  colors={"Smoothed (hindsight)": SERIES[0], "Real-time": SERIES[1]}))

# =============================================================================
# 6. Factors
# =============================================================================
with tabs["Factors"]:
    st.markdown("Weekly regressions on tradable factor proxies: "
                + ", ".join(f"**{k}** = {v if isinstance(v, str) else v[0] + ' − ' + v[1]}"
                            for k, v in cfg["factors"].items()))
    fd = slow("factor_decomp", weights_key, source, nav)
    fexp = slow("factor_exposures", weights_key, source, nav)
    c1, c2 = st.columns(2)
    with c1:
        plot(bar_fig(fd["Beta"].dropna(), "Portfolio factor betas", fmt=".2f", signed=True, height=340))
    with c2:
        plot(bar_fig(fd["% of variance"], "Share of portfolio variance by factor", height=340))
    st.caption(f"R² {fexp['R²'].iloc[0]:.0%}: the portfolio is mostly a combination of these macro bets. "
               "Variance shares can be negative when a factor hedges the others.")

    st.subheader("Holding-level exposures")
    ae = a.asset_factor_exposures
    fnames = list(cfg["factors"].keys())
    show(ae[fnames + ["R²", "Idiosyncratic share", "Idiosyncratic vol"]],
         num=fnames, pct=["R²", "Idiosyncratic share", "Idiosyncratic vol"], height=600)
    st.caption("Single stocks have large idiosyncratic shares: much of their risk is company-specific "
               "and isn't explained by macro factors. Factor ETFs regress perfectly on themselves (R² = 1).")

    st.subheader("Rolling factor betas (2-year window)")
    rb = slow("rolling_betas", weights_key, source, nav)
    pick = st.multiselect("Factors", fnames, default=["Equity", "Rates", "Credit"])
    plot(line_fig(rb[pick], "", yfmt=".2f", height=360))

# =============================================================================
# 7. Optimisation
# =============================================================================
with tabs["Optimisation"]:
    cons = O.Constraints.from_config(cfg)
    win = a.excess_returns.iloc[-756:]
    mu_method = st.radio("Expected returns", ["black_litterman", "shrunk", "historical"], horizontal=True,
                         format_func={"black_litterman": "Black-Litterman", "shrunk": "Shrunk historical",
                                      "historical": "Historical mean"}.get)
    mu = O.expected_returns(cfg, win, a.cov, mu_method)
    W, Sts = O.compare_portfolios(mu, a.cov, cons, cfg.weights)
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(heatmap(W, "Allocations by method", zmin=0, zmax=float(W.max().max()), fmt=".0%",
                     diverging=False, height=560))
    with c2:
        st.dataframe(fmt_summary(Sts), width="stretch")
        st.caption("Ledoit-Wolf covariance, 3-year window. Expected returns are in excess of cash. "
                   "HRP and equal-weight ignore the constraints (except cash), so their breaches are "
                   "counted above.")
    if mu_method == "black_litterman":
        st.subheader("Black-Litterman views")
        _, prior, views = O.black_litterman(a.cov, cfg.weights, cfg["expected_returns"]["views"],
                                            cfg["expected_returns"]["risk_aversion"],
                                            cfg["expected_returns"]["tau"])
        show(views, pct=["View return", "Confidence", "Prior implied", "Posterior"])
        plot(grouped_bar(pd.DataFrame({"Equilibrium (implied)": prior, "Posterior": mu}),
                         "Expected excess returns: prior vs posterior", fmt=".1%", height=340,
                         colors={"Equilibrium (implied)": SERIES[0], "Posterior": SERIES[1]}))

    st.subheader("Efficient frontier (with constraints)")
    ef = slow("frontier", weights_key, source, nav, mu_method)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ef["Volatility"], y=ef["Return"], mode="lines", name="Frontier",
                             line=dict(width=2, color=SERIES[0]),
                             hovertemplate="vol %{x:.1%}, return %{y:.2%}<extra>Frontier</extra>"))
    for i, k in enumerate(W.columns):
        s_ = Sts[k]
        fig.add_trace(go.Scatter(x=[s_["Volatility"]], y=[s_["Exp. excess return"]], mode="markers+text",
                                 name=k, text=[k], textposition="top center",
                                 marker=dict(size=11, color=color_for(k, i), line=dict(width=2, color="white")),
                                 hovertemplate=f"{k}: vol %{{x:.1%}}, return %{{y:.2%}}<extra></extra>"))
    fig.update_layout(height=420, xaxis_title="Volatility", yaxis_title="Expected excess return",
                      xaxis_tickformat=".0%", yaxis_tickformat=".1%", hovermode="closest", showlegend=False)
    plot(fig)

    st.subheader("How sensitive are the weights to return forecasts?")
    c1, c2, c3 = st.columns(3)
    smethod = c1.selectbox("Method", ["max_sharpe", "min_vol", "risk_parity"], index=0)
    bump = c2.select_slider("Bump", [0.005, 0.01, 0.02], value=0.01, format_func="{:+.1%}".format)
    ndraw = c3.select_slider("Resamples", [20, 50, 100], value=50)
    sens, res = sensitivity(weights_key, source, nav, smethod, bump, ndraw)
    if smethod == "max_sharpe":
        plot(heatmap(sens, f"Change in weights when one asset's expected return rises by {bump:.1%} "
                           "(row = asset bumped)", zmin=-0.15, zmax=0.15, fmt=".0%", height=560))
        st.caption(f"Average absolute weight change per {bump:.1%} bump: "
                   f"{sens.abs().sum(axis=1).mean() / 2:.0%} of the portfolio traded.")
    else:
        st.info("Min-vol and risk parity don't use expected returns, so they aren't affected by return "
                "forecasts. That is their main appeal.")
    fig = go.Figure()
    for i, t in enumerate(res.columns):
        fig.add_trace(go.Box(y=res[t], name=t, marker=dict(color=SERIES[0], size=4),
                             line=dict(width=1.5), boxpoints="outliers"))
    fig.update_layout(height=360, showlegend=False, yaxis_tickformat=".0%", hovermode="closest",
                      title="Resampled optimal weights: the spread shows estimation uncertainty")
    plot(fig)

# =============================================================================
# 8. Backtest
# =============================================================================
with tabs["Backtest"]:
    bt = cfg["backtest"]
    c1, c2, c3 = st.columns(3)
    rb_ = c1.selectbox("Rebalance", ["ME", "QE"], format_func={"ME": "Monthly", "QE": "Quarterly"}.get)
    cost = c2.select_slider("Transaction cost (bps, one-way)", [0, 5, 10, 25, 50], value=bt["cost_bps"])
    lb = c3.select_slider("Lookback", [504, 756, 1260], value=bt["lookback_days"],
                          format_func=lambda x: f"{x // 252} years")
    if not st.session_state.get("bt_on"):
        st.info("The walk-forward backtest re-optimises every strategy each month over ~19 years, which takes "
                "about 30 seconds the first time (then it is cached).", icon="⏱️")
        if st.button("Run walk-forward backtests", type="primary"):
            st.session_state["bt_on"] = True
            st.rerun()
    else:
        res = backtests(weights_key, source, nav, rb_, cost, lb)
        st.caption("Walk-forward: at each rebalance date the optimiser sees only the trailing lookback "
                   "window (covariance and expected returns re-estimated), trades at that close, and pays "
                   "costs on turnover. No future data enters any decision (checked by a unit test).")
        first = max(r.returns.index[0] for r in res.values())
        bench_bt = bench[bench.index >= first]
        perf = pd.DataFrame({k: R.cumulative(r.returns[r.returns.index >= first]) for k, r in res.items()})
        perf["benchmark"] = R.cumulative(bench_bt)
        plot(line_fig(perf, "Out-of-sample growth of $1 (after costs)", yfmt=".2f", height=420))
        tbl = pd.DataFrame({k: r.summary(rf, bench_bt, P) for k, r in res.items()})
        n_trials = len(res)
        trial_sd = tbl.loc["Sharpe"].std()
        tbl.loc["Deflated Sharpe (prob.)"] = [R.deflated_sharpe(r.returns, n_trials, trial_sd, rf, P)
                                              for r in res.values()]
        rows = ["Annual return", "Annual volatility", "Sharpe", "Deflated Sharpe (prob.)", "Sortino",
                "Max drawdown", "Information ratio", "Tracking error", "Annual turnover", "Annual cost drag", "Trades"]
        st.dataframe(fmt_summary(tbl.loc[rows]), width="stretch")
        st.caption(f"Deflated Sharpe = probability that the true Sharpe is above the best Sharpe you would expect from "
                   f"{n_trials} strategies with no skill. Below ~95% means the result could be luck from choosing "
                   "the best of several backtests.")

        pick = st.selectbox("Weights over time for", list(res.keys()), index=1)
        wt = res[pick].weights.resample("ME").last()
        fig = go.Figure()
        for i, t in enumerate(wt.columns):
            fig.add_trace(go.Scatter(x=wt.index, y=wt[t], name=t, stackgroup="one", mode="lines",
                                     line=dict(width=0.5, color="white"),
                                     fillcolor=SERIES[i % len(SERIES)],
                                     hovertemplate=f"{t}: %{{y:.1%}}<extra></extra>"))
        fig.update_layout(height=400, yaxis_tickformat=".0%", title=f"{pick}: allocation history")
        plot(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Is rebalancing worth the cost?")
            rbc = slow("rebalance_cmp", weights_key, source, nav)
            show(rbc, pct=["Annual return", "Annual volatility", "Annual turnover", "Annual cost drag"],
                 num=["Sharpe"])
            st.caption("A band rule trades only when a weight has drifted more than the band from its target, "
                       "which cuts turnover with little change in risk.")
        with c2:
            st.subheader("Cost sensitivity")
            cm = st.selectbox("Strategy", ["max_sharpe", "min_vol", "risk_parity", "hrp"], key="costm")
            cs = cost_sens(weights_key, source, nav, cm)
            show(cs, pct=["Annual return", "Annual turnover", "Annual cost drag"], num=["Sharpe"])

# =============================================================================
# 9. Stress tests
# =============================================================================
with tabs["Stress tests"]:
    st.subheader("Historical crises: today's weights replayed")
    hs, hc = S.historical_stress(a.asset_returns, cfg.weights, nav, bench)
    show(hs, pct=["Portfolio", "Worst point", "Benchmark"], money=["P&L ($)"])
    plot(grouped_bar(hs[["Portfolio", "Benchmark"]].astype(float), "Return over each crisis window",
                     height=380, colors={"Portfolio": SERIES[0], "Benchmark": SERIES[1]}))
    sc = st.selectbox("Contributions in", list(hc.columns))
    plot(bar_fig(hc[sc].sort_values(), f"Contribution by holding: {sc}", signed=True, height=480))

    st.subheader("Hypothetical shocks")
    hy, moves, per_asset = slow("hyp_stress", weights_key, source, nav)
    show(hy, pct=["Pure shock", "With correlated moves"], money=["P&L ($), correlated"])
    st.caption("**Pure**: only the named factor moves. **With correlated moves**: the other factors move "
               "as they typically did in stressed markets (VIX above its 80th percentile), given the shock. "
               "Asset P&L = factor betas × factor moves.")
    with st.expander("Factor moves used (correlated version)"):
        show(moves, pct=moves.columns)

    st.subheader("Build your own scenario")
    c = st.columns(6)
    custom = {"Equity": c[0].slider("Equities %", -40, 20, -10) / 100,
              "Rates": c[1].slider("Rates (bp)", -200, 300, 50),
              "Credit": c[2].slider("HY spreads (bp)", -100, 600, 100),
              "Inflation": c[3].slider("Breakevens (bp)", -100, 150, 0),
              "USD": c[4].slider("USD %", -15, 15, 0) / 100,
              "Commodity": c[5].slider("Commodities %", -40, 40, 0) / 100}
    custom = {k: v for k, v in custom.items() if v != 0}
    if custom:
        ch, _, cpa = S.hypothetical_stress(cfg, a.all_returns, cfg.weights, nav, a.vix,
                                           {"Custom": custom}, betas=a.asset_factor_exposures)
        k1, k2 = st.columns(2)
        kpi(k1, "Custom scenario P&L (pure)", f"{ch.loc['Custom', 'Pure shock']:.2%}",
            delta=f"${ch.loc['Custom', 'Pure shock'] * nav:,.0f}")
        fig = bar_fig((cfg.weights * (a.asset_factor_exposures[list(cfg['factors'])].astype(float)
                                      @ S.shock_to_factor_returns(cfg, custom).reindex(list(cfg['factors'])).fillna(0))
                       ).sort_values(), "Contribution by holding (pure shock)", signed=True, height=440)
        k2.metric("With correlated moves", f"{ch.loc['Custom', 'With correlated moves']:.2%}")
        plot(fig)

# =============================================================================
# 10. Monte Carlo
# =============================================================================
with tabs["Monte Carlo"]:
    mc = cfg["monte_carlo"]
    c1, c2, c3 = st.columns(3)
    horizon = c1.select_slider("Horizon", [21, 63, 126, 252, 504], value=mc["horizon_days"],
                               format_func=lambda d: f"{d} days")
    npaths = c2.select_slider("Paths", [1000, 2000, 5000, 10000], value=mc["n_paths"])
    model = c3.selectbox("Model", list(MC.ASSUMPTIONS), format_func={
        "normal": "Normal", "student_t": "Student-t (fat tails)", "bootstrap": "Historical bootstrap",
        "jump": "Normal + crash jumps (extreme)"}.get)
    st.info(f"**Assumptions:** {MC.ASSUMPTIONS[model]} Estimated on the last 10 years of data; weights "
            "held constant.", icon="ℹ️")
    stats_, fan, terminal = monte_carlo(weights_key, source, nav, model, horizon, npaths, mc["seed"],
                                        mc["t_dof"], mc["jump_intensity"])
    fig = go.Figure()
    x = np.arange(1, horizon + 1)
    for lo_, hi_, op in (("p1", "p99", 0.12), ("p5", "p95", 0.2), ("p25", "p75", 0.35)):
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[fan[hi_], fan[lo_][::-1]], fill="toself",
                                 fillcolor=f"rgba(42,120,214,{op})", line=dict(width=0),
                                 name=f"{lo_[1:]}–{hi_[1:]}th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=fan["p50"], name="Median", line=dict(width=2, color=SERIES[0])))
    fig.update_layout(height=420, title="Simulated growth of $1", xaxis_title="Trading days",
                      yaxis_tickformat=".2f")
    plot(fig)
    c1, c2 = st.columns([2, 3])
    with c1:
        s = stats_.copy()
        st.dataframe(pd.DataFrame({"Value": [f"${v:,.0f}" if "$" in k else f"{v:.1%}" for k, v in s.items()]},
                                  index=s.index), width="stretch", height=470)
    with c2:
        fig = go.Figure(go.Histogram(x=terminal, nbinsx=80, marker=dict(color=SERIES[0], line=dict(width=0)),
                                     hovertemplate="%{x:.1%}: %{y} paths<extra></extra>"))
        var95 = -stats_["VaR 95%"]
        fig.add_vline(x=var95, line=dict(color=SERIES[7], width=2, dash="dash"),
                      annotation_text=f"VaR 95% ({var95:.1%})")
        fig.update_layout(height=420, title="Distribution of horizon return", xaxis_tickformat=".0%",
                          hovermode="closest", bargap=0.02)
        plot(fig)
    st.subheader("All models side by side")
    rows = {}
    for m in MC.ASSUMPTIONS:
        sm, _, _ = monte_carlo(weights_key, source, nav, m, horizon, npaths, mc["seed"], mc["t_dof"],
                               mc["jump_intensity"])
        rows[m] = sm[["Mean return", "VaR 95%", "ES 95%", "VaR 99%", "ES 99%", "P(loss > 10%)",
                      "P(loss > 20%)", "5% worst max drawdown"]]
    show(pd.DataFrame(rows).T, pct=rows["normal"].index)
    st.caption("At long horizons, daily fat tails average out (normal ≈ Student-t). The bootstrap keeps "
               "clustering, so its tail is wider, and the jump model shows what history alone doesn't contain.")

# =============================================================================
# 11. Liquidity & limits
# =============================================================================
with tabs["Liquidity & limits"]:
    st.subheader("Risk limits")
    lr = LIM.limit_report(cfg, a.current_limit_metrics())
    cols = st.columns(len(lr))
    for col, (name, row) in zip(cols, lr.iterrows()):
        color, label = STATUS[row["Status"]]
        is_pct = "Days" not in name and "beta" not in name
        val = f"{row['Current']:.2%}" if is_pct else f"{row['Current']:.2f}"
        lim = f"{row['Limit value']:.1%}" if is_pct else f"{row['Limit value']:.2f}"
        col.markdown(f"<div style='border-left:4px solid {color};padding:4px 10px'>"
                     f"<div style='font-size:0.85em;opacity:0.75'>{name}</div>"
                     f"<div style='font-size:1.4em;font-weight:600'>{val}</div>"
                     f"<div style='font-size:0.8em'>limit {lim} · {row['Utilisation']:.0%} used</div>"
                     f"<div style='font-size:0.8em'>{label}</div></div>", unsafe_allow_html=True)
    st.caption(f"Amber at {cfg['limits']['amber_level']:.0%} of a limit. Limits are set in config/portfolio.yaml.")
    bh = LIM.breach_history(cfg, port, cfg["risk"]["var_window"])
    fig = line_fig(bh, "Limit utilisation through time (1.0 = at limit)", yfmt=".0%", height=340,
                   colors={"VaR utilisation": SERIES[0], "Drawdown utilisation": SERIES[1]})
    fig.add_hline(y=1.0, line=dict(color=STATUS["red"][0], width=1.5, dash="dash"))
    fig.add_hline(y=cfg["limits"]["amber_level"], line=dict(color=STATUS["amber"][0], width=1, dash="dot"))
    plot(fig)
    st.caption(f"Days in breach: VaR {(bh['VaR utilisation'] >= 1).sum()}, "
               f"drawdown {(bh['Drawdown utilisation'] >= 1).sum()} (of {len(bh)}).")

    st.subheader("Liquidity")
    part = st.slider("Max participation (% of daily volume)", 5, 50,
                     int(cfg["liquidity"]["participation_rate"] * 100), 5) / 100
    nav_mult = st.select_slider("Scale the fund", [1, 10, 50, 100], value=1,
                                format_func=lambda m: f"{m}× (${nav * m / 1e9:,.1f}bn)")
    adv = a.liquidity["ADV ($)"]
    L = cfg["liquidity"]
    lt = LQ.liquidity_table(cfg.weights, nav * nav_mult, adv, cfg.assets["spread_bps"],
                            a.asset_returns.iloc[-252:].std(), part, L["stress_adv_haircut"],
                            L["stress_spread_mult"])
    c1, c2 = st.columns([3, 2])
    with c1:
        show(lt.sort_values("Days (stressed)", ascending=False), money=["Position ($)", "ADV ($)"],
             pct=["% of ADV", "Exit cost", "Exit cost (stressed)"], num=["Days to liquidate", "Days (stressed)"],
             height=440)
    with c2:
        prof = pd.DataFrame({"Normal": LQ.liquidity_profile(lt, cfg.weights),
                             "Stressed": LQ.liquidity_profile(lt, cfg.weights, stressed=True)})
        plot(grouped_bar(prof, "Share of portfolio sellable", height=300,
                         colors={"Normal": SERIES[0], "Stressed": SERIES[1]}))
        v95 = RK.historical_var(port.iloc[-500:], 0.95)
        lv = LQ.liquidity_adjusted_var(v95, lt, cfg.weights, stressed=True)
        st.metric("Liquidity-adjusted VaR (stressed)", f"{lv['Liquidity-adjusted VaR']:.2%}",
                  delta=f"+{lv['Exit cost']:.2%} exit cost", delta_color="inverse")
    st.caption("Exit cost = half the bid-ask spread + square-root market impact (σ_daily × √(size/ADV)). "
               "Stress: ADV halved, spreads tripled. ADV is from the last 60 days of live volume data.")

# =============================================================================
# 12. FX
# =============================================================================
with tabs["FX"]:
    fxc = cfg["fx"]
    ccy = fxc["investor_currency"]
    st.markdown(f"The portfolio is in USD. For a **{ccy}-based investor**, returns also depend on "
                f"{fxc['fx_ticker'].replace('=X', '')}. Hedging removes the currency move but earns or pays "
                f"the interest-rate differential (set to {fxc['hedge_carry']:.1%} a year in the config).")
    fxr = a.md.prices[fxc["fx_ticker"]]
    carry = st.slider(f"Hedge carry (r_{ccy} − r_USD, % a year)", -3.0, 3.0, float(fxc["hedge_carry"] * 100), 0.25) / 100
    ht = FX.hedge_ratio_table(port, fxr, carry, periods=P)
    c1, c2 = st.columns([2, 3])
    with c1:
        show(ht, pct=ht.columns)
    with c2:
        curves = pd.DataFrame({f"{h:.0%} hedged": R.cumulative(FX.investor_returns(port, fxr, h, carry, P))
                               for h in (0.0, 0.5, 1.0)})
        plot(line_fig(curves, f"Growth of {ccy} 1", yfmt=".2f", height=340))
    st.caption("In risk-off episodes the USD tends to rally, so unhedged USD assets can *cushion* a non-USD "
               "investor's drawdowns. Compare the max drawdown column.")

# =============================================================================
# 13. Methodology
# =============================================================================
with tabs["Methodology"]:
    st.markdown((Path(__file__).resolve().parents[1] / "docs" / "methodology.md").read_text(encoding="utf-8"))
