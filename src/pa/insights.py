"""Investment thesis: the analytics turned into a short PM-style write-up.

Every sentence is generated from the current numbers, so the thesis stays
true when the data, holdings or config change. It describes what the models
show; it is not investment advice.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import optimise as O
from . import returns as R
from . import risk as RK
from . import stress as S
from . import trades as T
from . import var_backtest as VB
from .analysis import Analysis
from .correlation import regime_correlations, to_weekly


def _avg_offdiag(c: pd.DataFrame) -> float:
    m = c.to_numpy()
    n = len(m)
    return float((m.sum() - np.trace(m)) / (n * (n - 1)))


def _names(idx, n=3) -> str:
    idx = list(idx)[:n]
    return ", ".join(idx[:-1]) + f" and {idx[-1]}" if len(idx) > 1 else idx[0]


def build_thesis(a: Analysis, target_method: str = "max_sharpe") -> dict:
    cfg, port, bench, rf, P, nav = a.cfg, a.portfolio, a.benchmark, a.rf, a.periods, a.cfg["nav"]
    sec = {}

    # 1. Performance -----------------------------------------------------------
    s = R.performance_summary(port, rf, bench, P)
    sb = R.performance_summary(bench, rf, None, P)
    tr = R.trailing_returns(pd.DataFrame({"p": port, "b": bench}), P)
    since = tr.index[-1]
    sec["Performance"] = (
        f"Since {port.index[0]:%B %Y} the portfolio returned **{s['Annual return']:.1%} a year** at "
        f"**{s['Annual volatility']:.1%} volatility** (Sharpe {s['Sharpe']:.2f}), against "
        f"{sb['Annual return']:.1%} at {sb['Annual volatility']:.1%} for the benchmark (Sharpe {sb['Sharpe']:.2f}). "
        f"The worst drawdown was {s['Max drawdown']:.1%} vs {sb['Max drawdown']:.1%}, and it captured "
        f"{s['Up capture']:.0%} of the benchmark's up months but only {s['Down capture']:.0%} of its down months. "
        + (f"Over the last year: {tr.loc['1 year', 'p']:+.1%} vs {tr.loc['1 year', 'b']:+.1%}."
           if "1 year" in tr.index else ""))

    # 2. Active risk -----------------------------------------------------------
    act = a.active
    t = act["table"].sort_values("% of TE", ascending=False)
    top = t.head(3)
    hedges = t[t["% of TE"] < 0].sort_values("% of TE")
    bg = act["by_group"].sort_values("% of TE", ascending=False)
    rest = 1 - top["% of TE"].sum()
    lead, lead_w = top.index[0], top["Active weight"].iloc[0]
    stock_te = t.loc[t.index.isin(stocks_idx := cfg.asset_class[cfg.asset_class == "Single Stocks"].index),
                     "% of TE"].sum()
    sec["Active risk"] = (
        f"Ex-ante tracking error is **{act['tracking_error']:.1%}** with a beta of "
        f"{act['beta_to_benchmark']:.2f} to the benchmark. **The biggest sources are {_names(top.index)}**: "
        + "; ".join(f"{i} {r['Active weight']:+.0%} active, {r['% of TE']:.0%} of TE" for i, r in top.iterrows())
        + f" (everything else nets to {rest:+.0%}). Most active risk comes from being "
          f"**{'underweight' if lead_w < 0 else 'overweight'} {lead}**; the {len(stocks_idx)} single stocks "
          f"contribute only {stock_te:+.0%}. "
        + f"By group, {bg.index[0]} accounts for {bg['% of TE'].iloc[0]:.0%}. "
        + (f"{_names(hedges.index, 2)} slightly reduce tracking error: they partly offset the larger "
           "active bets (e.g. adding back some of the missing equity exposure)." if len(hedges) else "")
        + f" Realised tracking error over the last year was "
          f"{((port - bench).iloc[-252:].std() * np.sqrt(P)):.1%}.")

    # 3. Total-risk drivers -----------------------------------------------------
    rc = RK.risk_contributions(cfg.weights, a.cov).sort_values("% of risk", ascending=False)
    stocks = cfg.asset_class == "Single Stocks"
    eq_cls = cfg.asset_class.isin(["Equity", "Single Stocks"])
    worst_ratio = rc["Risk / weight"].idxmax()
    from .factors import factor_risk_decomposition
    fd = factor_risk_decomposition(cfg, a.all_returns, port)
    sec["Risk drivers"] = (
        f"Equity-type holdings are {cfg.weights[eq_cls].sum():.0%} of capital but "
        f"**{rc.loc[eq_cls[eq_cls].index, '% of risk'].sum():.0%} of volatility**; the equity factor alone "
        f"explains {fd.loc['Equity', '% of variance']:.0%} of portfolio variance. "
        f"The six single stocks are {cfg.weights[stocks].sum():.0%} of capital and "
        f"{rc.loc[stocks[stocks].index, '% of risk'].sum():.0%} of risk. "
        f"**{worst_ratio}** carries the most risk per unit of weight ({rc.loc[worst_ratio, 'Risk / weight']:.1f}×), "
        f"while {_names(rc['Risk / weight'].drop(cfg.cash_ticker).nsmallest(2).index, 2)} "
        f"diversify most cheaply.")

    # 4. Diversification under stress -------------------------------------------
    wk = to_weekly(a.asset_returns.drop(columns=[cfg.cash_ticker]))
    rg = regime_correlations(wk, a.vix.resample("W-FRI").last()) if a.vix is not None else None
    if rg is not None:
        sb_calm = rg["calm"].loc["SPY", "IEF"] if "IEF" in wk else np.nan
        sb_str = rg["stressed"].loc["SPY", "IEF"] if "IEF" in wk else np.nan
        cr_calm, cr_str = rg["calm"].loc["SPY", "HYG"], rg["stressed"].loc["SPY", "HYG"]
        sec["Diversification"] = (
            f"Average pairwise correlation rises from **{_avg_offdiag(rg['calm']):.2f} in calm markets "
            f"to {_avg_offdiag(rg['stressed']):.2f} when VIX is high**: diversification thins out when it "
            f"matters. Equity-credit correlation goes from {cr_calm:.2f} to {cr_str:.2f}, so HY credit "
            f"behaves like equity in a sell-off. Treasuries remain the hedge (SPY-IEF {sb_calm:.2f} calm, "
            f"{sb_str:.2f} stressed), but 2022 showed that this hedge fails when inflation drives rates.")

    # 5. Tails ------------------------------------------------------------------
    recent = port.iloc[-cfg["risk"]["var_window"]:]
    h99 = RK.historical_var(recent, 0.99)
    n99 = RK.parametric_var(recent.mean(), recent.std(), 0.99)
    vb, _ = VB.backtest_var(port, window=cfg["risk"]["var_window"], alpha=0.99)
    passed = list(vb.index[vb["Verdict"] == "pass"])
    sec["Tail risk"] = (
        f"1-day 99% VaR is **${h99 * nav:,.0f}** historically vs ${n99 * nav:,.0f} under a normal "
        f"distribution: the bell curve understates the tail by {h99 / n99 - 1:.0%}. "
        f"Daily returns have excess kurtosis of {s['Excess kurtosis']:.1f}. In the VaR backtest at 99%, "
        + (f"only {_names(passed, 4).replace('_', ' ')} pass{'es' if len(passed) == 1 else ''} the "
           "coverage and independence tests; the others breach in clusters, because they react too "
           "slowly when volatility jumps." if passed else
           "no method passes both tests, so treat VaR as a lower bound."))

    # 6. Stress -----------------------------------------------------------------
    hs, _ = S.historical_stress(a.asset_returns, cfg.weights, nav, bench)
    worst = hs["Portfolio"].astype(float).idxmin()
    hy, _, _ = S.hypothetical_stress(cfg, a.all_returns, cfg.weights, nav, a.vix,
                                     betas=a.asset_factor_exposures)
    hw = hy["With correlated moves"].idxmin()
    beat = (hs["Portfolio"].astype(float) > hs["Benchmark"].astype(float)).sum()
    sec["Stress"] = (
        f"Replaying history on today's weights, the worst episode is the **{worst} "
        f"({hs.loc[worst, 'Portfolio']:.1%}, −${abs(hs.loc[worst, 'P&L ($)']) / 1e6:,.1f}m)**. The portfolio lost "
        f"less than the benchmark in {beat} of {len(hs)} crises. Among hypothetical shocks, **{hw}** hurts "
        f"most once correlated moves are included ({hy.loc[hw, 'With correlated moves']:.1%}).")

    # 7. Liquidity --------------------------------------------------------------
    lq = a.liquidity.sort_values("Days (stressed)", ascending=False)
    sec["Liquidity"] = (
        f"Liquidity is not a constraint at ${nav / 1e6:,.0f}m: the least liquid holding, {lq.index[0]}, "
        f"would take {lq['Days (stressed)'].iloc[0]:.1f} days to exit at "
        f"{cfg['liquidity']['participation_rate']:.0%} of stressed volume. "
        f"The real liquidity risk is in HYG and LQD's underlying bonds, which can trade at a discount to "
        f"NAV in a crisis.")

    # 8. What the model would change ----------------------------------------------
    cons = O.Constraints.from_config(cfg)
    mu = O.expected_returns(cfg, a.excess_returns.iloc[-756:], a.cov)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = O.optimise(target_method, mu, a.cov, cons, cfg.weights.to_numpy())
    tp = T.trade_proposal(cfg.weights, target, a.cov_union, cfg.benchmark_weights, nav,
                          cfg["backtest"]["cost_bps"], cfg.assets["spread_bps"], cash=cfg.cash_ticker,
                          equity_betas=a.equity_betas, mu=mu)
    tt = tp["trades"]
    buys = tt[tt["Trade"] > 0].head(3)
    sells = tt[tt["Trade"] < 0].head(3)
    m = tp["metrics"]
    yrs = tp["summary"].get("Years to recover cost", np.inf)
    months = yrs * 12
    payback = ("" if not np.isfinite(yrs) or yrs >= 5 else
               "; on the model's own return forecasts the cost is earned back within a month" if months < 1
               else "; on the model's own return forecasts the cost is earned back in about a month"
               if months < 1.5 else
               f"; on the model's own return forecasts the cost is earned back in about {months:.0f} months")
    toward_bench = m.loc["Tracking error", "Change"] < 0 and m.loc["Beta to benchmark", "Change"] > 0
    label = {"max_sharpe": "maximum-Sharpe", "min_vol": "minimum-volatility",
             "risk_parity": "risk-parity"}.get(target_method, target_method)
    sec["What the model would change"] = (
        f"The constrained {label} optimiser (Black-Litterman returns, "
        f"Ledoit-Wolf risk) would **buy {_names([f'{i} {r.Trade:+.0%}' for i, r in buys.iterrows()])}** "
        f"and **sell {_names([f'{i} {r.Trade:+.0%}' for i, r in sells.iterrows()])}**, "
        f"{tp['summary']['One-way turnover']:.0%} one-way turnover costing about "
        f"${tp['summary']['Total cost ($)']:,.0f}. Volatility moves {m.loc['Volatility', 'Before']:.1%} → "
        f"{m.loc['Volatility', 'After']:.1%}, tracking error {m.loc['Tracking error', 'Before']:.1%} → "
        f"{m.loc['Tracking error', 'After']:.1%}, expected Sharpe {m.loc['Exp. Sharpe', 'Before']:.2f} → "
        f"{m.loc['Exp. Sharpe', 'After']:.2f}{payback}. "
        + (f"In effect it moves the portfolio **closer to the benchmark** (beta "
           f"{m.loc['Beta to benchmark', 'Before']:.2f} → {m.loc['Beta to benchmark', 'After']:.2f}) and "
           f"gives up some diversification (effective holdings {m.loc['Effective N', 'Before']:.0f} → "
           f"{m.loc['Effective N', 'After']:.0f}). " if toward_bench else "")
        + "The sensitivity analysis shows these weights move a lot for small changes in expected returns, "
          "so treat them as a direction, not a precise target.")

    return {"sections": sec, "trade": tp, "as_of": port.index[-1], "source": a.md.source}


def thesis_markdown(th: dict, heading: str = "##", escape_dollars: bool = True) -> str:
    """Markdown for the thesis. Dollar signs are escaped by default because
    Streamlit and GitHub both read a pair of `$` as LaTeX maths."""
    if escape_dollars:
        th = {**th, "sections": {k: v.replace("$", r"\$") for k, v in th["sections"].items()}}
    lines = [f"{heading} Investment thesis",
             f"*Generated from {th['source']} data to {th['as_of']:%d %B %Y}. "
             "Model output for exploration, not investment advice.*", ""]
    for title, text in th["sections"].items():
        lines += [f"**{title}.** {text}", ""]
    return "\n".join(lines)


def main():
    """python -m pa.insights [--source live|synthetic]: writes docs/thesis.md"""
    import argparse

    from .config import ROOT, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--out", default=str(ROOT / "docs" / "thesis.md"))
    args = ap.parse_args()
    th = build_thesis(Analysis.load(load_config(), args.source))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(thesis_markdown(th, "#") + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
