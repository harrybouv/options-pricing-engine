import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from implied_vol import implied_vol
import matplotlib.pyplot as plt
from Black_Scholes import bs_greeks
from market_data import market_data


def get_history(ticker, start, end):

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False
    )

    tk = yf.Ticker(ticker)
    info = tk.fast_info

    S0 = info.get("lastPrice", None)
    closes = data["Close"]
    adj_closes = data["Adj Close"]


    return closes, adj_closes, S0


def compute_returns(closes):

    returns = np.log(closes / closes.shift(1))
    returns = returns.dropna()

    return returns


def realised_volatility(returns, window):

    sigma_realised = returns.rolling(window).std() * np.sqrt(252)

    return sigma_realised


def plot_rolling_realised_volatility():

    plt.title("Rolling Realised Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()


def ATM_implied_volatility():

    data = market_data("AAPL", option_type="calls")
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

    start = "2025-03-01"
    end = "2026-03-01"
    windows = [10, 20, 30, 40, 50]

    closes, adj_closes, S0 = get_history("AAPL", start, end)
    returns = compute_returns(adj_closes)

    latest_rv = {}

    for window in windows:
        sigma_realised = realised_volatility(returns, window)
        sigma_realised = sigma_realised.dropna()
        sigma_realised = sigma_realised.iloc[:, 0]

        latest_rv[window] = float(sigma_realised.iloc[-1])

        print("Latest Realised Volatility for ", window, "day window =", latest_rv[window])

        plt.plot(sigma_realised.index, sigma_realised.values, label=f"{window}d")

    plot_rolling_realised_volatility()

    atm_df = ATM_implied_volatility()
    plot_ATM_implied_volatility(atm_df, rv_20=latest_rv[20], rv_50=latest_rv[50])

if __name__ == "__main__":
    main()