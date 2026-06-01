"""Multi-Asset Portfolio Risk Engine.

A production-ready, object-oriented risk engine that evaluates downside
exposure and risk-adjusted returns for multi-asset equity portfolios.

The engine ingests historical price data through ``yfinance`` (vectorised,
all tickers in a single request), then computes a suite of institutional
risk metrics on both a per-asset and a whole-portfolio basis:

* Annualised return and volatility
* Sharpe and Sortino ratios
* Maximum drawdown (depth and duration)
* Historical Value at Risk (VaR) and Conditional VaR / Expected Shortfall
* Correlation and covariance matrices
* Weighted portfolio return and volatility (covariance-based)

It also provides matplotlib/seaborn visualisations for cumulative returns
and drawdowns.

Example
-------
>>> engine = PortfolioRiskEngine(
...     tickers=["AAPL", "JPM", "GS"],
...     weights=[0.40, 0.30, 0.30],
...     risk_free_rate=0.04,
... )
>>> engine.run()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - yfinance is an optional runtime dep
    yf = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _PLOTTING_AVAILABLE = True
except ImportError:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]
    _PLOTTING_AVAILABLE = False


# Number of trading days used to annualise daily statistics.
TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class AssetMetrics:
    """Container for the risk metrics of a single asset.

    Attributes:
        ticker: The asset's ticker symbol.
        annual_return: Annualised mean return.
        annual_volatility: Annualised standard deviation of returns.
        downside_volatility: Annualised standard deviation of negative returns.
        sharpe_ratio: Excess return per unit of total volatility.
        sortino_ratio: Excess return per unit of downside volatility.
        max_drawdown: Worst peak-to-trough decline (negative number).
        max_drawdown_duration: Length, in trading days, of the longest
            underwater period.
        historical_var: Historical Value at Risk (negative number) for a
            single trading day at the configured confidence level.
        conditional_var: Conditional VaR / Expected Shortfall (negative number).
    """

    ticker: str
    annual_return: float
    annual_volatility: float
    downside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    historical_var: float
    conditional_var: float

    def as_dict(self) -> Dict[str, float]:
        """Return the metrics as a plain dictionary for tabular display.

        Returns:
            A dictionary keyed by human-readable metric name.
        """
        return {
            "Annual Return": self.annual_return,
            "Annual Volatility": self.annual_volatility,
            "Downside Volatility": self.downside_volatility,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown": self.max_drawdown,
            "Max DD Duration (days)": self.max_drawdown_duration,
            "Hist. VaR": self.historical_var,
            "CVaR (ES)": self.conditional_var,
        }


@dataclass
class PortfolioMetrics:
    """Container for whole-portfolio risk metrics.

    Attributes:
        annual_return: Weighted annualised portfolio return.
        annual_volatility: Covariance-based annualised portfolio volatility.
        sharpe_ratio: Portfolio Sharpe ratio.
        sortino_ratio: Portfolio Sortino ratio.
        max_drawdown: Portfolio maximum drawdown (negative number).
        max_drawdown_duration: Longest portfolio underwater period in days.
        historical_var: Portfolio one-day historical VaR (negative number).
        conditional_var: Portfolio one-day CVaR / Expected Shortfall.
    """

    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    historical_var: float
    conditional_var: float


class PortfolioRiskEngine:
    """Compute multi-asset portfolio risk metrics from market data.

    The engine downloads adjusted closing prices for a basket of tickers,
    derives daily returns, and exposes per-asset and portfolio-level risk
    analytics together with optional visualisations.

    Attributes:
        tickers: List of ticker symbols under analysis.
        weights: Portfolio weights aligned to ``tickers`` (sums to 1.0).
        risk_free_rate: Annual risk-free rate used for risk-adjusted ratios.
        confidence_level: Confidence level for VaR / CVaR (e.g. 0.95).
        period: History window passed to ``yfinance`` (e.g. ``"1y"``).
    """

    def __init__(
        self,
        tickers: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        risk_free_rate: float = 0.04,
        confidence_level: float = 0.95,
        period: str = "1y",
    ) -> None:
        """Initialise the engine and validate its configuration.

        Args:
            tickers: Iterable of ticker symbols to analyse.
            weights: Optional portfolio weights aligned with ``tickers``.
                Defaults to an equally weighted portfolio. Weights are
                normalised to sum to 1.0.
            risk_free_rate: Annual risk-free rate as a decimal (0.04 == 4%).
            confidence_level: Confidence level for VaR / CVaR in (0, 1).
            period: Historical look-back window understood by ``yfinance``.

        Raises:
            ValueError: If no tickers are supplied, the weight count does not
                match the ticker count, or the confidence level is invalid.
        """
        cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
        if not cleaned:
            raise ValueError("At least one ticker symbol must be provided.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")

        self.tickers: List[str] = cleaned
        self.risk_free_rate: float = risk_free_rate
        self.confidence_level: float = confidence_level
        self.period: str = period
        self.weights: np.ndarray = self._normalise_weights(weights, len(cleaned))

        # Populated by ``load_data``.
        self.prices: pd.DataFrame = pd.DataFrame()
        self.returns: pd.DataFrame = pd.DataFrame()
        self.portfolio_returns: pd.Series = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_weights(
        weights: Optional[Sequence[float]], n_assets: int
    ) -> np.ndarray:
        """Validate and normalise portfolio weights.

        Args:
            weights: Raw weights or ``None`` for equal weighting.
            n_assets: Number of assets the weights must cover.

        Returns:
            A NumPy array of weights summing to 1.0.

        Raises:
            ValueError: If the weight count is mismatched or the weights sum
                to zero (and therefore cannot be normalised).
        """
        if weights is None:
            return np.repeat(1.0 / n_assets, n_assets)

        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape[0] != n_assets:
            raise ValueError(
                f"Expected {n_assets} weights, received {weight_array.shape[0]}."
            )
        total = weight_array.sum()
        if np.isclose(total, 0.0):
            raise ValueError("Portfolio weights cannot sum to zero.")
        return weight_array / total

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Download price history and compute daily returns.

        All tickers are fetched in a single vectorised ``yf.download`` call.
        Tickers with no usable data (e.g. delisted or mistyped symbols) are
        dropped, and ``self.weights`` is re-normalised over the survivors so
        the portfolio still sums to 1.0.

        Returns:
            The DataFrame of daily simple returns (one column per surviving
            ticker).

        Raises:
            RuntimeError: If ``yfinance`` is not installed or no ticker
                returns usable data.
        """
        if yf is None:
            raise RuntimeError(
                "yfinance is not installed. Install it with `pip install yfinance`."
            )

        raw = yf.download(
            self.tickers,
            period=self.period,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise RuntimeError(
                "No price data returned. Check the tickers and your connection."
            )

        prices = self._extract_close_prices(raw)
        # Drop tickers that are entirely missing (delisted / invalid symbols).
        prices = prices.dropna(axis=1, how="all")
        missing = [t for t in self.tickers if t not in prices.columns]
        if missing:
            print(f"Warning: no data for {missing}; dropping from the portfolio.")

        if prices.empty:
            raise RuntimeError("None of the requested tickers returned data.")

        # Align weights to the surviving, ordered set of tickers.
        surviving = [t for t in self.tickers if t in prices.columns]
        keep_idx = [self.tickers.index(t) for t in surviving]
        self.weights = self._normalise_weights(self.weights[keep_idx], len(surviving))
        self.tickers = surviving
        prices = prices[surviving].dropna(how="any")

        self.prices = prices
        self.returns = prices.pct_change().dropna(how="any")
        self.portfolio_returns = self.returns.to_numpy() @ self.weights
        self.portfolio_returns = pd.Series(
            self.portfolio_returns, index=self.returns.index, name="Portfolio"
        )
        return self.returns

    @staticmethod
    def _extract_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
        """Extract the close-price frame from a ``yf.download`` result.

        ``yfinance`` returns a single-level frame for one ticker and a
        MultiIndex column frame for several. This normalises both shapes into
        a flat ``DataFrame`` of closing prices keyed by ticker.

        Args:
            raw: The raw object returned by ``yf.download``.

        Returns:
            A DataFrame of close prices with one column per ticker.
        """
        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close" if "Close" in raw.columns.get_level_values(0) else None
            if field is not None:
                return raw[field].copy()
            # Fall back to the second level if the layout is transposed.
            return raw.xs("Close", axis=1, level=-1).copy()
        # Single ticker: ``raw`` is already a flat OHLCV frame.
        return raw[["Close"]].copy()

    # ------------------------------------------------------------------
    # Core metric calculations
    # ------------------------------------------------------------------
    @staticmethod
    def _max_drawdown(returns: pd.Series) -> tuple[float, int]:
        """Compute maximum drawdown depth and its longest duration.

        Args:
            returns: Series of periodic simple returns.

        Returns:
            A tuple ``(max_drawdown, duration_in_days)`` where the drawdown is
            a non-positive float and the duration is the length of the longest
            consecutive underwater stretch.
        """
        if returns.empty:
            return 0.0, 0
        cumulative = (1.0 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1.0
        max_dd = float(drawdown.min())

        # Longest run of consecutive underwater (drawdown < 0) days.
        underwater = drawdown < 0
        longest = current = 0
        for is_under in underwater:
            current = current + 1 if is_under else 0
            longest = max(longest, current)
        return max_dd, int(longest)

    def _historical_var_cvar(self, returns: pd.Series) -> tuple[float, float]:
        """Compute one-day historical VaR and Conditional VaR.

        Historical VaR is the empirical quantile of the return distribution at
        ``1 - confidence_level``. CVaR (Expected Shortfall) is the mean of all
        returns at or below that VaR threshold, capturing tail severity.

        Args:
            returns: Series of periodic simple returns.

        Returns:
            A tuple ``(var, cvar)`` expressed as negative decimals (a 5%
            one-day loss is ``-0.05``).
        """
        if returns.empty:
            return 0.0, 0.0
        alpha = 1.0 - self.confidence_level
        var = float(np.quantile(returns.to_numpy(), alpha))
        tail = returns[returns <= var]
        cvar = float(tail.mean()) if not tail.empty else var
        return var, cvar

    def _ratios(
        self, annual_return: float, annual_vol: float, downside_vol: float
    ) -> tuple[float, float]:
        """Compute Sharpe and Sortino ratios with safe division.

        Args:
            annual_return: Annualised return.
            annual_vol: Annualised total volatility.
            downside_vol: Annualised downside volatility.

        Returns:
            A tuple ``(sharpe_ratio, sortino_ratio)``. Returns ``nan`` for a
            ratio whose denominator is zero.
        """
        excess = annual_return - self.risk_free_rate
        sharpe = excess / annual_vol if annual_vol > 0 else float("nan")
        sortino = excess / downside_vol if downside_vol > 0 else float("nan")
        return sharpe, sortino

    def _metrics_from_returns(self, returns: pd.Series) -> Dict[str, float]:
        """Compute the shared metric block for any return series.

        Args:
            returns: Series of periodic simple returns.

        Returns:
            A dictionary of computed metrics shared by assets and the
            portfolio.
        """
        annual_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
        annual_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        downside = returns[returns < 0]
        downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe, sortino = self._ratios(annual_return, annual_vol, downside_vol)
        max_dd, dd_duration = self._max_drawdown(returns)
        var, cvar = self._historical_var_cvar(returns)
        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "downside_volatility": downside_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_drawdown_duration": dd_duration,
            "historical_var": var,
            "conditional_var": cvar,
        }

    def compute_asset_metrics(self) -> Dict[str, AssetMetrics]:
        """Compute the full risk profile for each individual asset.

        Returns:
            A mapping of ticker symbol to its :class:`AssetMetrics`.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called.
        """
        self._require_data()
        results: Dict[str, AssetMetrics] = {}
        for ticker in self.tickers:
            block = self._metrics_from_returns(self.returns[ticker])
            results[ticker] = AssetMetrics(ticker=ticker, **block)
        return results

    def compute_portfolio_metrics(self) -> PortfolioMetrics:
        """Compute whole-portfolio risk metrics.

        Portfolio volatility is computed from the annualised covariance matrix
        using ``sqrt(wᵀ Σ w)`` rather than a naive weighted average of asset
        volatilities, so it correctly reflects diversification.

        Returns:
            The aggregated :class:`PortfolioMetrics`.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called.
        """
        self._require_data()
        block = self._metrics_from_returns(self.portfolio_returns)

        # Covariance-based annualised volatility (captures diversification).
        cov_annual = self.returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
        portfolio_variance = float(self.weights @ cov_annual @ self.weights)
        block["annual_volatility"] = float(np.sqrt(max(portfolio_variance, 0.0)))

        # Recompute ratios against the covariance-based volatility.
        sharpe, sortino = self._ratios(
            block["annual_return"],
            block["annual_volatility"],
            block["downside_volatility"],
        )
        block["sharpe_ratio"] = sharpe
        block["sortino_ratio"] = sortino
        block.pop("downside_volatility")
        return PortfolioMetrics(**block)

    def correlation_matrix(self) -> pd.DataFrame:
        """Return the asset return correlation matrix.

        Returns:
            A DataFrame of pairwise Pearson correlations.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called.
        """
        self._require_data()
        return self.returns.corr()

    def covariance_matrix(self, annualised: bool = True) -> pd.DataFrame:
        """Return the asset return covariance matrix.

        Args:
            annualised: If ``True``, scale the daily covariance by the number
                of trading days per year.

        Returns:
            A DataFrame covariance matrix.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called.
        """
        self._require_data()
        cov = self.returns.cov()
        return cov * TRADING_DAYS_PER_YEAR if annualised else cov

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        """Build a tabular summary of per-asset and portfolio metrics.

        Returns:
            A DataFrame whose columns are the assets plus a ``Portfolio``
            column and whose rows are the individual risk metrics.
        """
        asset_metrics = self.compute_asset_metrics()
        portfolio = self.compute_portfolio_metrics()

        table = {ticker: m.as_dict() for ticker, m in asset_metrics.items()}
        table["Portfolio"] = {
            "Annual Return": portfolio.annual_return,
            "Annual Volatility": portfolio.annual_volatility,
            "Downside Volatility": float("nan"),
            "Sharpe Ratio": portfolio.sharpe_ratio,
            "Sortino Ratio": portfolio.sortino_ratio,
            "Max Drawdown": portfolio.max_drawdown,
            "Max DD Duration (days)": portfolio.max_drawdown_duration,
            "Hist. VaR": portfolio.historical_var,
            "CVaR (ES)": portfolio.conditional_var,
        }
        return pd.DataFrame(table)

    def print_report(self) -> None:
        """Print a formatted, human-readable risk report to stdout."""
        conf_pct = int(round(self.confidence_level * 100))
        weight_str = ", ".join(
            f"{t} {w:.0%}" for t, w in zip(self.tickers, self.weights)
        )
        print("--- Multi-Asset Portfolio Risk Engine ---")
        print(f"Look-back period   : {self.period}")
        print(f"Risk-free rate     : {self.risk_free_rate:.2%}")
        print(f"VaR/CVaR confidence: {conf_pct}%")
        print(f"Portfolio weights  : {weight_str}\n")

        asset_metrics = self.compute_asset_metrics()
        for ticker, m in asset_metrics.items():
            print(f"[{ticker}] Risk Profile:")
            print(f"  Annualized Return  : {m.annual_return:.2%}")
            print(f"  Volatility (Risk)  : {m.annual_volatility:.2%}")
            print(f"  Max Drawdown       : {m.max_drawdown:.2%}"
                  f" (duration {m.max_drawdown_duration} days)")
            print(f"  Sharpe Ratio       : {m.sharpe_ratio:.2f}")
            print(f"  Sortino Ratio      : {m.sortino_ratio:.2f}")
            print(f"  Hist. VaR ({conf_pct}%)    : {m.historical_var:.2%}")
            print(f"  CVaR / ES ({conf_pct}%)    : {m.conditional_var:.2%}\n")

        portfolio = self.compute_portfolio_metrics()
        print("[PORTFOLIO] Aggregated Risk Profile:")
        print(f"  Total Return       : {portfolio.annual_return:.2%}")
        print(f"  Total Volatility   : {portfolio.annual_volatility:.2%}")
        print(f"  Max Drawdown       : {portfolio.max_drawdown:.2%}"
              f" (duration {portfolio.max_drawdown_duration} days)")
        print(f"  Sharpe Ratio       : {portfolio.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio      : {portfolio.sortino_ratio:.2f}")
        print(f"  Hist. VaR ({conf_pct}%)    : {portfolio.historical_var:.2%}")
        print(f"  CVaR / ES ({conf_pct}%)    : {portfolio.conditional_var:.2%}\n")

        print("Correlation Matrix (diversification view):")
        print(self.correlation_matrix().round(3).to_string())
        print()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_cumulative_returns(
        self, show: bool = True, save_path: Optional[str] = None
    ):
        """Plot cumulative returns of each asset versus the portfolio.

        Args:
            show: If ``True``, display the figure interactively.
            save_path: Optional path to write the figure to disk (PNG).

        Returns:
            The matplotlib ``Figure`` instance, or ``None`` if plotting
            libraries are unavailable.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called.
        """
        self._require_data()
        if not _PLOTTING_AVAILABLE:
            print("matplotlib/seaborn not installed; skipping cumulative plot.")
            return None

        sns.set_theme(style="whitegrid")
        growth = (1.0 + self.returns).cumprod()
        portfolio_growth = (1.0 + self.portfolio_returns).cumprod()

        fig, ax = plt.subplots(figsize=(11, 6))
        for ticker in self.tickers:
            ax.plot(growth.index, growth[ticker], linewidth=1.2, alpha=0.8, label=ticker)
        ax.plot(
            portfolio_growth.index,
            portfolio_growth,
            color="black",
            linewidth=2.4,
            label="Portfolio",
        )
        ax.set_title("Cumulative Returns: Assets vs. Portfolio")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_drawdown(
        self, show: bool = True, save_path: Optional[str] = None
    ):
        """Plot the portfolio's historical underwater (drawdown) curve.

        Args:
            show: If ``True``, display the figure interactively.
            save_path: Optional path to write the figure to disk (PNG).

        Returns:
            The matplotlib ``Figure`` instance, or ``None`` if plotting
            libraries are unavailable.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called.
        """
        self._require_data()
        if not _PLOTTING_AVAILABLE:
            print("matplotlib/seaborn not installed; skipping drawdown plot.")
            return None

        sns.set_theme(style="whitegrid")
        cumulative = (1.0 + self.portfolio_returns).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(drawdown.index, drawdown.to_numpy(), 0.0,
                        color="crimson", alpha=0.4)
        ax.plot(drawdown.index, drawdown, color="crimson", linewidth=1.2)
        ax.set_title("Portfolio Drawdown (Underwater Curve)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, plot: bool = True) -> pd.DataFrame:
        """Load data, print the report, and optionally render charts.

        Args:
            plot: If ``True``, render the cumulative-return and drawdown
                charts after printing the report.

        Returns:
            The summary table of metrics.
        """
        self.load_data()
        self.print_report()
        summary = self.summary_table()
        if plot:
            self.plot_cumulative_returns()
            self.plot_drawdown()
        return summary

    def _require_data(self) -> None:
        """Ensure price/return data has been loaded.

        Raises:
            RuntimeError: If :meth:`load_data` has not been called yet.
        """
        if self.returns.empty:
            raise RuntimeError("Data not loaded. Call `load_data()` first.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point for the risk engine.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Asset Portfolio Risk Engine",
    )
    parser.add_argument(
        "-t", "--tickers", nargs="+", default=["AAPL", "JPM", "GS"],
        help="Ticker symbols to analyse (default: AAPL JPM GS).",
    )
    parser.add_argument(
        "-w", "--weights", nargs="+", type=float, default=None,
        help="Portfolio weights aligned to --tickers (default: equal weight).",
    )
    parser.add_argument(
        "-r", "--risk-free-rate", type=float, default=0.04,
        help="Annual risk-free rate as a decimal (default: 0.04).",
    )
    parser.add_argument(
        "-c", "--confidence", type=float, default=0.95,
        help="Confidence level for VaR/CVaR (default: 0.95).",
    )
    parser.add_argument(
        "-p", "--period", default="1y",
        help="History window understood by yfinance (default: 1y).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip rendering the charts.",
    )
    args = parser.parse_args(argv)

    try:
        engine = PortfolioRiskEngine(
            tickers=args.tickers,
            weights=args.weights,
            risk_free_rate=args.risk_free_rate,
            confidence_level=args.confidence,
            period=args.period,
        )
        engine.run(plot=not args.no_plot)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
