# Multi-Asset Portfolio Analytics

Risk, performance, attribution and portfolio-construction analytics for a multi-asset portfolio, with an interactive Streamlit dashboard and a one-page tear sheet.

The model portfolio is 15 ETFs and stocks across equities, government bonds, credit, commodities and cash, plus six single stocks (Apple, JPMorgan, J&J, Exxon, P&G, Caterpillar), measured against a 60/40 benchmark. Holdings, constraints, limits and views all live in [`config/portfolio.yaml`](config/portfolio.yaml).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

streamlit run app/dashboard.py  # interactive dashboard
python -m pa.tearsheet          # writes reports/tearsheet.html
pytest                          # 29 known-answer tests (offline, synthetic data)
```

Market data is downloaded from Yahoo Finance and FRED on first run and cached in `data/cache/`. Offline, the code falls back to a synthetic two-regime market. A banner says when synthetic data is in use.

## What it answers

| Question | Where |
|---|---|
| Returns, volatility, Sharpe/Sortino/Calmar/Omega, skew and kurtosis, net-of-fee and real returns | Overview |
| Is the Sharpe ratio real? Bootstrap CI, Probabilistic and Deflated Sharpe | Overview, Backtest |
| Beating the benchmark? Active return, tracking error, IR, up/down capture, downside beta | Benchmark & attribution |
| Who made the money? Contributions, Brinson-Fachler allocation/selection, single stocks vs SPY | Benchmark & attribution |
| VaR and Expected Shortfall (4 methods); **does the VaR model work?** (Kupiec, Christoffersen) | Risk |
| Which holdings drive risk? Risk contributions, component and incremental VaR | Risk |
| EWMA/GARCH volatility forecasts, vol targeting, gross/net/beta-adjusted exposure, DV01/CS01 | Risk |
| Drawdown depth, duration and recovery time | Drawdowns |
| Does diversification survive a crisis? Calm vs stressed correlations, rolling correlations, HMM regimes | Correlations & regimes |
| Hidden macro bets: equity, rates, credit, inflation, USD, commodity factors; idiosyncratic risk | Factors |
| Max Sharpe vs min vol vs risk parity vs HRP under realistic constraints; Black-Litterman; sensitivity | Optimisation |
| Walk-forward out-of-sample backtest with costs; rebalancing rules; cost sensitivity | Backtest |
| Historical crises and hypothetical shocks (pure and correlated); build-your-own scenario | Stress tests |
| Monte Carlo: normal, Student-t, bootstrap, crash-jump models, each with stated assumptions | Monte Carlo |
| How fast can we get out? Days to liquidate, liquidity-adjusted VaR, risk-limit monitor | Liquidity & limits |
| Hedged vs unhedged returns for a GBP investor | FX |

Every method and assumption is documented in [`docs/methodology.md`](docs/methodology.md), which is also the dashboard's Methodology tab.

## Layout

```
config/portfolio.yaml   holdings, benchmark, constraints, limits, views, costs
src/pa/                 the analytics library (pure functions, no UI)
  data.py               Yahoo/FRED download + cache; synthetic market
  analysis.py           shared context object used by the dashboard and tear sheet
  returns.py  risk.py  var_backtest.py  drawdown.py  correlation.py  factors.py
  optimise.py  backtest.py  attribution.py  stress.py  montecarlo.py
  liquidity.py  limits.py  fx.py  tearsheet.py
app/dashboard.py        Streamlit + Plotly dashboard
tests/                  pytest suite
```

## Customising

Edit `config/portfolio.yaml` to change holdings (any Yahoo ticker), weights, constraints, limits, Black-Litterman views or the benchmark. Weights can also be edited live in the dashboard sidebar.

## Caveats

- The single stocks were picked with hindsight (survivorship bias), so backtests that include them are optimistic.
- ETF prices begin in mid-2007, so the 2008 crisis is covered but earlier crises are not.
- Durations, spreads and factor proxies are approximations suited to analysis, not trading.
