"""Known-answer tests for the analytics. These run on synthetic data, so no
network access is needed."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pa import attribution as A
from pa import backtest as B
from pa import drawdown as D
from pa import optimise as O
from pa import returns as R
from pa import risk
from pa import var_backtest as VB
from pa.config import load_config
from pa.correlation import fit_hmm_2state
from pa.data import synthetic_market


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def market(cfg):
    return synthetic_market(cfg.all_tickers(), "2010-01-01", "2020-12-31", seed=1,
                            vix=cfg["vix_ticker"], fx=cfg["fx"]["fx_ticker"])


@pytest.fixture(scope="module")
def rets(cfg, market):
    return market.returns(cfg.tickers)


def bdays(n):
    return pd.bdate_range("2020-01-01", periods=n)


# ---------------------------------------------------------------- returns
def test_ann_return_constant_daily():
    r = pd.Series(0.001, index=bdays(252))
    assert R.ann_return(r) == pytest.approx(1.001 ** 252 - 1)


def test_sharpe_matches_definition():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000), index=bdays(1000))
    expected = (r - 0.0001).mean() / (r - 0.0001).std() * np.sqrt(252)
    assert R.sharpe(r, 0.0001) == pytest.approx(expected)


def test_max_drawdown_simple_path():
    # 100 -> 120 -> 90 -> 130: drawdown 25% from the 120 peak
    r = pd.Series([0.2, -0.25, 130 / 90 - 1], index=bdays(3))
    assert R.max_drawdown(r) == pytest.approx(-0.25)


def test_constant_mix_equals_weighted_sum():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(0, 0.01, (50, 2)), index=bdays(50), columns=["a", "b"])
    w = pd.Series({"a": 0.3, "b": 0.7})
    pd.testing.assert_series_equal(R.portfolio_returns(x, w, None), (x @ w).rename("portfolio"))


def test_probabilistic_sharpe_normal_case():
    # for a large normal sample PSR(0) is close to Phi(SR * sqrt(n-1))
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0004, 0.01, 2000), index=bdays(2000))
    sr = r.mean() / r.std()
    assert R.probabilistic_sharpe(r) == pytest.approx(stats.norm.cdf(sr * np.sqrt(1999)), abs=0.02)


def test_deflated_sharpe_is_lower_with_more_trials():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500), index=bdays(1500))
    assert R.deflated_sharpe(r, 100, 0.5) < R.deflated_sharpe(r, 2, 0.5)


# ---------------------------------------------------------------- risk
def test_parametric_var_closed_form():
    assert risk.parametric_var(0.0, 0.01, 0.99) == pytest.approx(0.01 * 2.326348, rel=1e-5)
    # normal ES at 97.5% ~= VaR at 99%
    assert risk.parametric_es(0.0, 0.01, 0.975) == pytest.approx(
        risk.parametric_var(0.0, 0.01, 0.99), rel=0.01)


def test_historical_var_on_large_normal_sample():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 200_000))
    assert risk.historical_var(r, 0.95) == pytest.approx(0.01 * 1.6449, rel=0.02)
    assert risk.historical_es(r, 0.95) == pytest.approx(0.01 * 2.0627, rel=0.02)


def test_cornish_fisher_equals_normal_without_skew_or_kurtosis():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 500_000))
    assert risk.cornish_fisher_var(r) == pytest.approx(
        risk.parametric_var(r.mean(), r.std(), 0.95), rel=0.01)


def test_risk_contributions_sum_to_vol():
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.01]], index=list("ab"), columns=list("ab"))
    w = pd.Series({"a": 0.5, "b": 0.5})
    rc = risk.risk_contributions(w, cov)
    assert rc["Vol contribution"].sum() == pytest.approx(np.sqrt(w @ cov @ w))
    assert rc["% of risk"].sum() == pytest.approx(1.0)
    # the riskier asset contributes more than its 50% weight
    assert rc.loc["a", "% of risk"] > 0.5


def test_component_var_sums_to_portfolio_var():
    cov = pd.DataFrame(np.diag([1e-4, 4e-4]), index=list("ab"), columns=list("ab"))
    w = pd.Series({"a": 0.6, "b": 0.4})
    cv = risk.component_var(w, cov, 0.95)
    assert cv["Component VaR"].sum() == pytest.approx(1.6449 * np.sqrt(w @ cov @ w), rel=1e-4)


def test_ledoit_wolf_matches_sklearn(rets):
    from sklearn.covariance import LedoitWolf

    x = rets.iloc[:500]
    ref = LedoitWolf().fit(x.to_numpy()).covariance_ * 252
    np.testing.assert_allclose(risk.ledoit_wolf_cov(x).to_numpy(), ref, rtol=1e-8)


def test_ewma_vol_has_no_look_ahead():
    r = pd.Series(np.r_[np.full(100, 0.01), 0.5], index=bdays(101))
    v = risk.ewma_vol(r)
    assert v.iloc[-1] == pytest.approx(0.01 * np.sqrt(252), rel=1e-6)  # shock not yet seen


# ---------------------------------------------------------------- drawdowns
def test_drawdown_table_durations():
    # peak day 1, trough day 3, recovered on day 5
    r = pd.Series([0.0, 0.1, -0.1, -0.1, 0.1, 0.2], index=bdays(6))
    t = D.drawdown_table(r)
    row = t.iloc[0]
    assert row["Peak"] == r.index[1] and row["Trough"] == r.index[3]
    assert row["Recovery"] == r.index[5]
    assert row["Peak to trough (days)"] == 2 and row["Trough to recovery (days)"] == 2


# ---------------------------------------------------------------- VaR backtest
def test_kupiec_accepts_correct_model():
    rng = np.random.default_rng(6)
    br = pd.Series(rng.random(5000) < 0.05)
    assert VB.kupiec_pof(br, 0.95)["Kupiec p-value"] > 0.01


def test_kupiec_rejects_bad_model():
    br = pd.Series(np.r_[np.ones(500), np.zeros(4500)].astype(bool))  # 10% breaches
    assert VB.kupiec_pof(br, 0.95)["Kupiec p-value"] < 0.001


def test_christoffersen_detects_clustering():
    br = pd.Series(np.tile(np.r_[np.ones(5), np.zeros(95)], 50).astype(bool))
    assert VB.christoffersen_independence(br)["Christoffersen p-value"] < 0.001


def test_rolling_var_uses_only_past():
    r = pd.Series(np.random.default_rng(7).normal(0, 0.01, 600), index=bdays(600))
    v1 = VB.rolling_var_forecast(r, "historical", 250)
    r2 = r.copy()
    r2.iloc[-1] = -0.5
    v2 = VB.rolling_var_forecast(r2, "historical", 250)
    assert v1.iloc[-1] == v2.iloc[-1]


# ---------------------------------------------------------------- optimisation
def test_optimisers_respect_constraints(cfg, rets):
    cons = O.Constraints.from_config(cfg)
    cov = risk.ledoit_wolf_cov(rets)
    mu = O.historical_mean(rets.sub(rets[cfg.cash_ticker], axis=0))
    for m in ("max_sharpe", "min_vol", "risk_parity"):
        w = O.optimise(m, mu, cov, cons)
        assert cons.violations(w) == [], (m, cons.violations(w))


def test_risk_parity_equalises_contributions_when_unconstrained():
    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.005], (2000, 3)), columns=list("abc"))
    cov = x.cov() * 252
    cons = O.Constraints(list("abc"), np.zeros(3), np.ones(3), pd.Series("x", index=list("abc")), {})
    rc = risk.risk_contributions(O.risk_parity(cov, cons), cov)["% of risk"]
    assert np.allclose(rc, 1 / 3, atol=1e-3)


def test_hrp_weights_sum_to_one(cfg, rets):
    cons = O.Constraints.from_config(cfg)
    w = O.hrp(risk.ledoit_wolf_cov(rets), cons)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    assert w[cfg.cash_ticker] == pytest.approx(cons.cash_min)


def test_black_litterman_no_views_returns_prior():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=list("ab"), columns=list("ab"))
    w = pd.Series({"a": 0.5, "b": 0.5})
    post, prior, _ = O.black_litterman(cov, w, [])
    pd.testing.assert_series_equal(post, prior.rename("Posterior"))


def test_black_litterman_moves_towards_confident_view():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=list("ab"), columns=list("ab"))
    w = pd.Series({"a": 0.5, "b": 0.5})
    view = [{"type": "absolute", "asset": "a", "return": 0.20, "confidence": 0.99}]
    post, prior, _ = O.black_litterman(cov, w, view)
    assert abs(post["a"] - 0.20) < abs(prior["a"] - 0.20) and post["a"] > 0.15


# ---------------------------------------------------------------- backtest
def test_backtest_has_no_look_ahead(cfg, rets):
    """Changing returns after a decision date must not change that decision."""
    strat = B.optimised_strategy(cfg, "min_vol")
    base = B.walk_forward(rets, strat, lookback=500, rebalance="QE", name="a")
    d = base.decisions[2]
    shocked = rets.copy()
    shocked.loc[shocked.index > d] *= -3
    strat2 = B.optimised_strategy(cfg, "min_vol")
    alt = B.walk_forward(shocked, strat2, lookback=500, rebalance="QE", name="b")
    pd.testing.assert_series_equal(base.weights.loc[d], alt.weights.loc[d], check_names=False,
                                   atol=1e-6)


def test_backtest_costs_reduce_returns(cfg, rets):
    s = B.static_strategy(cfg.weights)
    free = B.walk_forward(rets, s, 252, "ME", 0.0)
    costly = B.walk_forward(rets, s, 252, "ME", 50.0)
    assert (free.returns - costly.returns).sum() == pytest.approx(costly.costs.sum())
    assert costly.costs.sum() > 0


def test_static_backtest_matches_portfolio_returns(cfg, rets):
    res = B.walk_forward(rets, B.static_strategy(cfg.weights), 252, "ME", 0.0)
    ref = R.portfolio_returns(rets, cfg.weights, "ME").loc[res.returns.index]
    np.testing.assert_allclose(res.returns.to_numpy(), ref.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------- attribution
def test_contributions_sum_to_compounded_return(cfg, rets):
    bod = A.drifting_weights(rets, cfg.weights, "ME")
    port = R.portfolio_returns(rets, cfg.weights, "ME")
    by_year = A.contribution(rets, bod, "YE")
    yearly = R.calendar_returns(port, "YE")
    np.testing.assert_allclose(by_year.sum(axis=1).to_numpy(), yearly.to_numpy(), atol=1e-10)


def test_brinson_effects_sum_to_active_return(cfg, market):
    rets = market.returns()
    a_rets = rets[cfg.tickers]
    bod = A.drifting_weights(a_rets, cfg.weights, "ME")
    res = A.brinson(cfg, rets, bod)
    assert res["by_group"].loc["Total", "Total"] == pytest.approx(res["active_return"], abs=1e-10)
    port = R.portfolio_returns(a_rets, cfg.weights, "ME")
    assert res["portfolio_return"] == pytest.approx((1 + port).prod() - 1, rel=1e-10)


# ---------------------------------------------------------------- active risk & trades
def test_active_risk_contributions_sum_to_te():
    cov = pd.DataFrame([[0.04, 0.01, 0.0], [0.01, 0.02, 0.0], [0.0, 0.0, 0.01]],
                       index=list("abc"), columns=list("abc"))
    wp = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    wb = pd.Series({"a": 0.6, "b": 0.4})
    res = risk.active_risk(wp, wb, cov)
    act = (wp.reindex(cov.index).fillna(0) - wb.reindex(cov.index).fillna(0)).to_numpy()
    assert res["tracking_error"] == pytest.approx(np.sqrt(act @ cov.to_numpy() @ act))
    assert res["table"]["Contribution to TE"].sum() == pytest.approx(res["tracking_error"])
    assert res["table"]["% of TE"].sum() == pytest.approx(1.0)


def test_trade_proposal_costs_and_min_trade():
    from pa import trades as T

    idx = list("abc")
    cov = pd.DataFrame(np.diag([0.04, 0.02, 1e-6]), index=idx, columns=idx)
    cur = pd.Series({"a": 0.5, "b": 0.48, "c": 0.02})
    tgt = pd.Series({"a": 0.4, "b": 0.581, "c": 0.019})   # c trade below threshold
    res = T.trade_proposal(cur, tgt, cov, pd.Series({"a": 0.5, "b": 0.5}), nav=1e6, cost_bps=10,
                           min_trade=0.0025, cash="c")
    t = res["target"]
    assert t.sum() == pytest.approx(1.0)
    assert res["trades"].loc["a", "Trade ($)"] == pytest.approx(-100_000)
    # costs: |trades| x 10 bps
    assert res["summary"]["Total cost ($)"] == pytest.approx((t - cur).abs().sum() * 1e6 * 10 / 1e4)


def test_trailing_returns_ytd():
    idx = pd.bdate_range("2023-12-01", "2024-03-29")
    r = pd.Series(0.001, index=idx)
    tr = R.trailing_returns(r)
    n_ytd = (idx.year == 2024).sum()
    assert tr.loc["YTD"].iloc[0] == pytest.approx(1.001 ** n_ytd - 1)
    assert "1 year" not in tr.index   # not enough history


def test_thesis_builds_on_synthetic_data(cfg):
    from pa.analysis import Analysis
    from pa.insights import build_thesis, thesis_markdown

    a = Analysis.load(cfg, "synthetic")
    th = build_thesis(a)
    md = thesis_markdown(th)
    for section in ("Performance", "Active risk", "Stress", "What the model would change"):
        assert f"**{section}.**" in md


# ---------------------------------------------------------------- regimes
def test_hmm_separates_calm_and_volatile_regimes():
    rng = np.random.default_rng(9)
    calm, wild = rng.normal(0, 0.005, 500), rng.normal(0, 0.03, 200)
    r = pd.Series(np.r_[calm, wild, calm], index=bdays(1200))
    res = fit_hmm_2state(r)
    assert res["vol"][0] == pytest.approx(0.005, rel=0.2)
    assert res["vol"][1] == pytest.approx(0.03, rel=0.2)
    assert res["smoothed"].iloc[600] > 0.9 and res["smoothed"].iloc[100] < 0.1
