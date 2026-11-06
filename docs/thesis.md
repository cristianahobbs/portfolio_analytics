# Investment thesis
*Generated from live data to 05 November 2026. Model output for exploration, not investment advice.*

**Performance.** Since June 2007 the portfolio returned **8.3% a year** at **10.0% volatility** (Sharpe 0.71), against 7.6% at 11.8% for the benchmark (Sharpe 0.56). The worst drawdown was -28.7% vs -35.8%, and it captured 91% of the benchmark's up months but only 78% of its down months. Over the last year: +6.8% vs +5.7%.

**Active risk.** Ex-ante tracking error is **5.8%** with a beta of 0.72 to the benchmark. **The biggest sources are SPY, GLD and IEF**: SPY -50% active, 69% of TE; GLD +10% active, 18% of TE; IEF +15% active, 5% of TE (everything else nets to +8%). Most active risk comes from being **underweight SPY**; the 9 single stocks contribute only -4%. By group, Equity accounts for 65%. TSLA and AAPL slightly reduce tracking error: they partly offset the larger active bets (e.g. adding back some of the missing equity exposure). Realised tracking error over the last year was 4.9%.

**Risk drivers.** Equity-type holdings are 43% of capital but **66% of volatility**; the equity factor alone explains 64% of portfolio variance. The 9 single stocks are 18% of capital and 29% of risk. **TSLA** carries the most risk per unit of weight (3.5×), while JNJ and PG diversify most cheaply.

**Diversification.** Average pairwise correlation rises from **0.21 in calm markets to 0.33 when VIX is high**: diversification thins out when it matters. Equity-credit correlation goes from 0.67 to 0.79, so HY credit behaves like equity in a sell-off. Treasuries remain the hedge (SPY-IEF -0.24 calm, -0.20 stressed), but 2022 showed that this hedge fails when inflation drives rates.

**Tail risk.** 1-day 99% VaR is **\$1,739,570** historically vs \$1,308,744 under a normal distribution: the bell curve understates the tail by 33%. Daily returns have excess kurtosis of 9.7. In the VaR backtest at 99%, no method passes both tests, so treat VaR as a lower bound.

**Stress.** Replaying history on today's weights, the worst episode is the **Global Financial Crisis (-18.5%, −\$18.5m)**. The portfolio lost less than the benchmark in 7 of 8 crises. Among hypothetical shocks, **Stagflation** hurts most once correlated moves are included (-10.2%).

**Liquidity.** Liquidity is not a constraint at \$100m: the least liquid holding, DBC, would take 1.9 days to exit at 20% of stressed volume. The real liquidity risk is in HYG and LQD's underlying bonds, which can trade at a discount to NAV in a crisis.

**What the model would change.** The constrained maximum-Sharpe optimiser (Black-Litterman returns, Ledoit-Wolf risk) would **buy SPY +17% and EEM +5%** and **sell EFA -8%, IEF -3% and LQD -2%**, 22% one-way turnover costing about \$46,149. Volatility moves 9.0% → 10.0%, tracking error 5.8% → 4.5%, expected Sharpe 0.31 → 0.32; on the model's own return forecasts the cost is earned back in about a month. In effect it moves the portfolio **closer to the benchmark** (beta 0.72 → 0.88) and gives up some diversification (effective holdings 12 → 8). The sensitivity analysis shows these weights move a lot for small changes in expected returns, so treat them as a direction, not a precise target.

