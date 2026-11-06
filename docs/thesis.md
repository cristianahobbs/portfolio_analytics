# Investment thesis
*Generated from live data to 05 November 2026. Model output for exploration, not investment advice.*

**Performance.** Since June 2007 the portfolio returned **7.2% a year** at **9.6% volatility** (Sharpe 0.62), against 7.6% at 11.8% for the benchmark (Sharpe 0.56). The worst drawdown was -28.9% vs -35.8%, and it captured 85% of the benchmark's up months but only 79% of its down months. Over the last year: +8.5% vs +5.7%.

**Active risk.** Ex-ante tracking error is **5.4%** with a beta of 0.69 to the benchmark. **The biggest sources are SPY, GLD and IEF**: SPY -44% active, 78% of TE; GLD +10% active, 18% of TE; IEF +15% active, 5% of TE (everything else nets to -1%). Most active risk comes from being **underweight SPY**; the 6 single stocks contribute only -5%. By group, Equity accounts for 68%. AAPL and EFA slightly reduce tracking error: they partly offset the larger active bets (e.g. adding back some of the missing equity exposure). Realised tracking error over the last year was 4.9%.

**Risk drivers.** Equity-type holdings are 43% of capital but **62% of volatility**; the equity factor alone explains 63% of portfolio variance. The six single stocks are 12% of capital and 13% of risk. **CAT** carries the most risk per unit of weight (2.3×), while DBC and IEF diversify most cheaply.

**Diversification.** Average pairwise correlation rises from **0.22 in calm markets to 0.32 when VIX is high**: diversification thins out when it matters. Equity-credit correlation goes from 0.67 to 0.79, so HY credit behaves like equity in a sell-off. Treasuries remain the hedge (SPY-IEF -0.24 calm, -0.20 stressed), but 2022 showed that this hedge fails when inflation drives rates.

**Tail risk.** 1-day 99% VaR is **\$1,688,581** historically vs \$1,213,926 under a normal distribution: the bell curve understates the tail by 39%. Daily returns have excess kurtosis of 10.9. In the VaR backtest at 99%, no method passes both tests, so treat VaR as a lower bound.

**Stress.** Replaying history on today's weights, the worst episode is the **Global Financial Crisis (-18.8%, −\$18.8m)**. The portfolio lost less than the benchmark in 7 of 8 crises. Among hypothetical shocks, **Stagflation** hurts most once correlated moves are included (-10.1%).

**Liquidity.** Liquidity is not a constraint at \$100m: the least liquid holding, DBC, would take 1.9 days to exit at 20% of stressed volume. The real liquidity risk is in HYG and LQD's underlying bonds, which can trade at a discount to NAV in a crisis.

**What the model would change.** The constrained maximum-Sharpe optimiser (Black-Litterman returns, Ledoit-Wolf risk) would **buy SPY +18% and EEM +5%** and **sell EFA -9%, IEF -4% and LQD -2%**, 23% one-way turnover costing about \$48,665. Volatility moves 8.2% → 9.5%, tracking error 5.4% → 3.9%, expected Sharpe 0.30 → 0.31; on the model's own return forecasts the cost is earned back in about a month. In effect it moves the portfolio **closer to the benchmark** (beta 0.69 → 0.87) and gives up some diversification (effective holdings 10 → 6). The sensitivity analysis shows these weights move a lot for small changes in expected returns, so treat them as a direction, not a precise target.

