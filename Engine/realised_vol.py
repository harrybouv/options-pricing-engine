import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from Engine.market_data import market_data
except ModuleNotFoundError:
    from market_data import market_data


def get_series(data: pd.DataFrame, column: str) -> pd.Series:
    values = data[column]

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    return values.dropna()


def get_history(ticker, start=None, end=None, period=None):
    if period is not None:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    else:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

    if data.empty:
        raise ValueError(f"No historical data found for {ticker}")

    closes = get_series(data, "Close")

    if "Adj Close" in data.columns:
        adj_closes = get_series(data, "Adj Close")
    else:
        adj_closes = closes

    tk = yf.Ticker(ticker)
    info = tk.fast_info

    try:
        S0 = info.get("lastPrice", None)
    except Exception:
        S0 = getattr(info, "lastPrice", None)

    if S0 is None or not np.isfinite(S0):
        S0 = float(closes.iloc[-1])

    return closes, adj_closes, float(S0)


def compute_returns(closes):
    returns = np.log(closes / closes.shift(1))
    returns = returns.dropna()

    return returns


def realised_volatility(returns, window):
    sigma_realised = returns.rolling(window).std() * np.sqrt(252)

    return sigma_realised


def get_realised_vol(ticker: str, period: str = "1y", window: int | None = None) -> float:
    closes, adj_closes, S0 = get_history(ticker, period=period)
    returns = compute_returns(adj_closes)

    if window is not None:
        returns = returns.tail(window)

    sigma = returns.std() * np.sqrt(252)

    return float(sigma)


def plot_rolling_realised_volatility():
    plt.title("Rolling Realised Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()


def ATM_implied_volatility(ticker="AAPL", option_type="calls"):
    data = market_data(ticker, option_type=option_type)
    surface_df = data["surface_df"].copy()

    surface_df["atm_distance"] = (surface_df["moneyness"] - 1).abs()

    atm_df = surface_df.loc[
        surface_df.groupby("expiry")["atm_distance"].idxmin()
    ].copy()

    atm_df = atm_df.sort_values("T")

    return atm_df


def plot_ATM_implied_volatility(atm_df, rv_20=None, rv_50=None):
    plt.figure(figsize=(8, 5))
    plt.plot(atm_df["T"], atm_df["sigma"], marker="o", linestyle="-", label="ATM IV")

    if rv_20 is not None:
        plt.axhline(rv_20, linestyle="--", label=f"20d RV = {rv_20:.3f}")

    if rv_50 is not None:
        plt.axhline(rv_50, linestyle="--", label=f"50d RV = {rv_50:.3f}")

    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    plt.title("AAPL CALLS ATM Implied Volatility Term Structure")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    ticker = "AAPL"
    start = "2025-03-01"
    end = "2026-03-01"
    windows = [10, 20, 30, 40, 50]

    closes, adj_closes, S0 = get_history(ticker, start=start, end=end)
    returns = compute_returns(adj_closes)

    latest_rv = {}

    for window in windows:
        sigma_realised = realised_volatility(returns, window)
        sigma_realised = sigma_realised.dropna()

        latest_rv[window] = float(sigma_realised.iloc[-1])

        print("Latest Realised Volatility for ", window, "day window =", latest_rv[window])

        plt.plot(sigma_realised.index, sigma_realised.values, label=f"{window}d")

    plot_rolling_realised_volatility()

    atm_df = ATM_implied_volatility(ticker=ticker, option_type="calls")
    plot_ATM_implied_volatility(atm_df, rv_20=latest_rv[20], rv_50=latest_rv[50])


if __name__ == "__main__":
    main()