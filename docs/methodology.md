## Methodology and assumptions

### Data
- **Source.** Yahoo Finance daily closes, adjusted for dividends and splits, so all returns are **total returns**. CPI comes from FRED (`CPIAUCSL`). Downloads are cached in `data/cache/`. When the data is offline, a synthetic market is used instead: a two-regime factor model with fat-tailed shocks and crisis correlation jumps. A banner says when it is in use.
- **Calendar.** Rows are SPY trading days. VIX and FX values are forward-filled across holiday gaps of up to 5 days. Nothing is ever back-filled, because that would leak future prices into the past.
- **Asynchronous closes.** EFA and EEM hold assets that close hours before the US. That biases *daily* correlations towards zero, so correlations, factor regressions and stress betas use **weekly** returns by default.
- **Survivorship and hindsight bias.** The six single stocks (Apple, JPMorgan, J&J, Exxon, P&G, Caterpillar) were chosen *today*, knowing they survived and became large. A 2007 investor couldn't have known that, so treat stock-level backtest results as optimistic. The ETF universe has the same issue to a lesser degree.
- **Cash and the risk-free rate.** BIL (1–3 month T-bills) stands in for both.

### Returns and performance
- Portfolio returns use **drifting weights**, rebalanced to target at each month end (configurable).
- Annual return is geometric (CAGR). Volatility is the annualised standard deviation of daily returns. Sharpe and Sortino use excess returns over BIL.
- **Net of fees** deducts the management fee daily. **Real** returns deflate by CPI, spread evenly across each month's trading days. CPI is published with a lag, so this is an after-the-fact measure.
- **Sharpe confidence.**
  - A stationary block bootstrap gives a confidence interval that respects volatility clustering.
  - The Probabilistic Sharpe Ratio (Bailey & López de Prado) adjusts for sample length, skew and kurtosis.
  - The Deflated Sharpe Ratio also corrects for picking the best of N backtested strategies.

### Risk
- **VaR/ES methods.** Historical, normal, Student-t (fitted) and Cornish-Fisher (skew/kurtosis-adjusted). Multi-day VaR uses √time scaling, which understates risk when volatility clusters.
- **VaR backtest.** Rolling, out-of-sample 1-day forecasts (each uses data up to the previous day only).
  - Kupiec test: is the breach frequency right?
  - Christoffersen test: are breaches independent of each other?
  - A conditional-coverage test combines the two.
  - *Filtered historical simulation* rescales past returns by current EWMA volatility. It is usually the only method that passes both tests.
- **Covariance.** Ledoit-Wolf shrinkage over the last 3 years. This stabilises the matrix inverse that optimisers rely on.
- **Risk contributions.** An Euler decomposition, so the contributions sum to total volatility (or total VaR, for component VaR). Incremental VaR re-prices the portfolio after a proposed trade.
- **Volatility models.**
  - EWMA (λ = 0.94) is a forecast that uses data up to t−1.
  - GARCH(1,1) uses Student-t errors. It is fitted on the full sample, so its history is descriptive, not a true forecast.
  - The vol-targeting overlay scales exposure by target ÷ EWMA forecast, capped by the leverage limit.
- **DV01 / CS01.** Position × (spread) duration × 1bp. Durations are approximate config inputs; a production system would take them from holdings data.

### Correlations and regimes
- **Calm vs stressed.** Stress means the *previous* period's VIX is above its 80th percentile, so the regime is known in advance.
- **Two-state hidden Markov model** on portfolio returns, fitted by Baum-Welch.
  - *Smoothed* probabilities use the full sample (hindsight).
  - *Filtered* probabilities use data up to each date only (real time).

### Optimisation
- **Inputs.** Expected returns are excess over cash. The three options are:
  - Black-Litterman: equilibrium returns implied by the current weights, plus the config views. View uncertainty uses the Idzorek-style heuristic Ω = (1−c)/c · PτΣP′.
  - Shrunk historical means.
  - Raw historical means.
- **Constraints:**
  - per-asset caps (5% per single stock)
  - asset-class ranges
  - minimum cash
  - long-only by default (shorting and a gross leverage cap are supported)
- **Methods:**
  - maximum Sharpe
  - minimum volatility
  - risk parity: equal risk contributions across risky assets, cash at its minimum
  - Hierarchical Risk Parity (López de Prado): clusters assets by correlation; applies only the cash constraint
  - equal weight
- **Sensitivity.**
  - Bump each expected return by 1% and re-optimise.
  - Michaud-style resampling: redraw expected returns from their sampling distribution and re-optimise. The spread of weights shows how fragile "optimal" is.

### Backtesting
- **No look-ahead.** At each rebalance date the strategy sees only the trailing lookback window. It trades at that day's close, and the new weights first earn returns the next day. A unit test changes all future returns and checks that past decisions don't change.
- **Costs.** One-way costs in bps on turnover, including the initial build. Band rebalancing trades only when a weight drifts past a threshold.
- Black-Litterman views were written today, so a backtest that uses them carries some hindsight.

### Attribution
- **Brinson-Fachler** by group (Equity, Fixed Income, Real Assets, Cash), monthly, against group proxies (SPY, AGG, DBC, BIL). The 60/40 benchmark holds zero Real Assets and Cash, so for those groups the selection effect appears as *interaction*.
- **Carino linking** makes the monthly effects add up exactly to the compounded active return.
- **Holding-level selection** = weight × (holding return − group proxy return). For the single stocks this answers: did picking these names beat simply owning SPY?

### Stress tests and Monte Carlo
- **Historical stress.** Today's weights, held through each crisis window.
- **Hypothetical shocks.** Market-unit shocks are converted to factor returns using the factor instruments' durations, then mapped to assets through their weekly factor betas. The *correlated* version also moves the other factors by their expected value given the shock, using the stressed-period covariance. Linear betas miss non-linear behaviour (options, convexity, defaults).
- **Monte Carlo.** Each model's assumptions are shown next to its results. Every model holds weights constant and uses the last 10 years of data. The jump model adds crashes beyond anything in history.

### Liquidity
- **Days to liquidate** = position ÷ (participation rate × 60-day average daily traded value). Under stress, ADV is halved and spreads tripled.
- **Exit cost** = half the spread + square-root market impact. Liquidity-adjusted VaR = market VaR + exit cost.
- **Caveat.** ETF on-screen volume understates true liquidity, because authorised participants can create and redeem shares. The underlying bonds (HYG, LQD) can be far less liquid than the ETF in a crisis.
