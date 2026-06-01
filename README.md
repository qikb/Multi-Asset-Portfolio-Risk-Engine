# Multi-Asset Portfolio Risk Engine

A production-ready, object-oriented quantitative risk tool built with Python,
`yfinance`, `pandas`, `numpy`, `matplotlib`, and `seaborn`. It evaluates
downside exposure and risk-adjusted returns for multi-asset equity portfolios
on both a **per-asset** and a **whole-portfolio** basis.

## The Objective

Analyse historical equity pricing (e.g. AAPL, JPM, GS) and calculate
institutional-grade risk metrics. Beyond standard volatility tracking, the
engine measures tail risk (VaR/CVaR), drawdown depth and duration, and
diversification benefits via the correlation/covariance structure of the
basket.

## Features

- **OOP architecture** – the logic lives in a reusable
  `PortfolioRiskEngine` class.
- **Vectorised ingestion** – all tickers are fetched in a single
  `yf.download()` call instead of looping per ticker.
- **Per-asset and portfolio metrics** – annualised return, volatility, Sharpe,
  Sortino, maximum drawdown (with duration), historical VaR, and CVaR.
- **Custom weights** – pass arbitrary weights (e.g. 40/30/30); the engine
  computes total portfolio return and covariance-based portfolio volatility.
- **Diversification view** – correlation and covariance matrices.
- **Visualisation** – cumulative returns (assets vs. portfolio) and a drawdown
  underwater chart.
- **Robustness** – type hints, PEP-8 docstrings, and graceful handling of
  delisted/missing tickers (they are dropped and weights re-normalised).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command line

```bash
# Default basket (AAPL, JPM, GS), equal weights
python portfolio_risk_engine.py

# Custom tickers, weights, and a 99% VaR confidence level
python portfolio_risk_engine.py -t AAPL JPM GS -w 0.4 0.3 0.3 -c 0.99

# Skip charts (useful for headless environments)
python portfolio_risk_engine.py --no-plot
```

### As a library

```python
from portfolio_risk_engine import PortfolioRiskEngine

engine = PortfolioRiskEngine(
    tickers=["AAPL", "JPM", "GS"],
    weights=[0.40, 0.30, 0.30],
    risk_free_rate=0.04,
    confidence_level=0.95,
    period="1y",
)
engine.load_data()
engine.print_report()
summary = engine.summary_table()      # tidy DataFrame of every metric
corr = engine.correlation_matrix()    # diversification view
engine.plot_cumulative_returns()
engine.plot_drawdown()
```

## Quantitative Formulas

Daily simple returns are computed as `rₜ = Pₜ / Pₜ₋₁ − 1`. Daily statistics
are annualised with **252** trading days.

- **Annualised return:** `mean(r) × 252`.
- **Annualised volatility:** `std(r) × √252`.
- **Sharpe ratio:** `(Rₐ − R_f) / σₐ`, the excess return per unit of *total*
  volatility, where `R_f` is the risk-free rate.
- **Sortino ratio:** `(Rₐ − R_f) / σ_downside`. The downside deviation
  annualises the standard deviation of only the negative returns, so upside
  volatility is not penalised.
- **Maximum drawdown:** the largest peak-to-trough decline of the cumulative
  return curve, `min(Cₜ / cummax(C) − 1)`. The engine also reports the
  **duration** (longest consecutive run of underwater days).
- **Historical VaR (95%):** the empirical `5%` quantile of the daily return
  distribution — the loss that is not exceeded with 95% confidence on a given
  day. Reported as a negative number.
- **Conditional VaR / Expected Shortfall (95%):** the *average* of all returns
  at or beyond the VaR threshold, i.e. the expected loss *given* that the VaR
  breach occurs. CVaR is a coherent tail-risk measure and is always at least as
  severe as VaR.
- **Correlation matrix:** pairwise Pearson correlations of asset returns,
  quantifying diversification potential.
- **Portfolio return:** `Rₚ = wᵀ R`, the weight-dotted vector of asset returns.
- **Portfolio volatility:** `σₚ = √(wᵀ Σ w)`, where `Σ` is the **annualised
  covariance matrix** and `w` the weight vector. Using the full covariance
  matrix (rather than a weighted average of individual volatilities) correctly
  captures the variance-reduction benefit of diversification — portfolio
  volatility is typically *lower* than any single constituent.

## Sample output

```
[PORTFOLIO] Aggregated Risk Profile:
  Total Return       : 41.30%
  Total Volatility   : 18.20%
  Max Drawdown       : -12.10% (duration 60 days)
  Sharpe Ratio       : 2.04
  Sortino Ratio      : 2.97
  Hist. VaR (95%)    : -1.60%
  CVaR / ES (95%)    : -2.50%
```

> Live figures vary with the market window; the values above are illustrative.
