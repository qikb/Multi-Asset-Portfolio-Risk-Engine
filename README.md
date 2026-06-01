# Multi-Asset Portfolio Risk Engine

This project is a middle-office quantitative risk tool built with Python, `yfinance`, and `NumPy` to evaluate downside exposure and risk-adjusted returns for equity portfolios.

## The Objective
To analyze historical equity pricing (e.g., AAPL, JPM, GS) and calculate institutional-grade risk metrics. Moving beyond standard volatility tracking, this tool calculates Downside Deviation and Maximum Drawdown to provide a highly accurate profile of capital preservation and tail risk.

## How It Works
1. **Data Ingestion:** Pulls the last 252 trading days of daily closing prices via the Yahoo Finance API and calculates daily percentage returns.
2. **Traditional Risk (Sharpe Ratio):** Calculates standard annualized volatility and the Sharpe Ratio against an assumed 4% risk-free rate.
3. **Downside Exposure (Sortino Ratio):** Isolates strictly negative returns to calculate downside deviation, generating the Sortino Ratio to evaluate true risk-adjusted performance without penalizing upside price momentum.
4. **Tail Risk (Max Drawdown):** Computes cumulative returns and rolling maximums to determine the Maximum Drawdown (Max DD), identifying the absolute worst-case peak-to-trough historical loss.

## Sample output
--- Running Advanced Portfolio Risk Analyzer (Sortino & Max DD) ---

[AAPL] Risk Profile:
  Annualized Return : 47.27%
  Volatility (Risk) : 22.08%
  Max Drawdown      : -13.80%
  Sharpe Ratio      : 1.96
  Sortino Ratio     : 3.15

[JPM] Risk Profile:
  Annualized Return : 16.89%
  Volatility (Risk) : 21.43%
  Max Drawdown      : -15.47%
  Sharpe Ratio      : 0.60
  Sortino Ratio     : 0.81

[GS] Risk Profile:
  Annualized Return : 59.25%
  Volatility (Risk) : 27.11%
  Max Drawdown      : -19.42%
  Sharpe Ratio      : 2.04
  Sortino Ratio     : 3.02
