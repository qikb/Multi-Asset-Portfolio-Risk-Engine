import yfinance as yf
import numpy as np
import pandas as pd

tickers = ['AAPL', 'JPM', 'GS']
risk_free_rate = 0.04  

print("--- Running Advanced Portfolio Risk Analyzer (Sortino & Max DD) ---\n")

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        # 1. Calculate Daily Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df = df.dropna()
        
        # 2. Traditional Risk Metrics (Sharpe)
        annual_return = df['Daily_Return'].mean() * 252
        annual_volatility = df['Daily_Return'].std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 3. Downside Risk (Sortino Ratio)
        # Isolate only the negative daily returns
        downside_returns = df.loc[df['Daily_Return'] < 0, 'Daily_Return']
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
        
        # 4. Maximum Drawdown (Worst-case loss from a peak)
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
        df['Rolling_Max'] = df['Cumulative_Return'].cummax()
        df['Drawdown'] = df['Cumulative_Return'] / df['Rolling_Max'] - 1
        max_drawdown = df['Drawdown'].min()
        
        # 5. Output Institutional Metrics
        print(f"[{ticker}] Risk Profile:")
        print(f"  Annualized Return : {annual_return:.2%}")
        print(f"  Volatility (Risk) : {annual_volatility:.2%}")
        print(f"  Max Drawdown      : {max_drawdown:.2%}")
        print(f"  Sharpe Ratio      : {sharpe_ratio:.2f}")
        print(f"  Sortino Ratio     : {sortino_ratio:.2f}\n")
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")