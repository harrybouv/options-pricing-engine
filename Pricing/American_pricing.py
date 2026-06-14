import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Engine.Black_Scholes import bs_price
from Engine.market_data import get_spot_price, get_dividend_yield, get_volatility


def get_data(ticker, K, T, option_type, vol_mode="surface", sigma=None):
    S0 = get_spot_price(ticker)
    q = get_dividend_yield(ticker)

    surface_type = "calls" if option_type == "call" else "puts"

    sigma = get_volatility(
        ticker=ticker,
        K=K,
        T=T,
        vol_mode=vol_mode,
        sigma=sigma,
        option_type=surface_type,
    )

    return {
        "S0": float(S0),
        "sigma": float(sigma),
        "q": float(q),
    }


def binomial_american_option(S0, K, sigma, q, r, T, N, option_type):
    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if N <= 0:
        raise ValueError("N must be positive")

    if T <= 0:
        if option_type == "call":
            payoff = max(S0 - K, 0.0)
        else:
            payoff = max(K - S0, 0.0)

        return {
            "price": payoff,
            "stock_tree": np.array([[S0]]),
            "option_tree": np.array([[payoff]]),
            "u": np.nan,
            "d": np.nan,
            "p_star": np.nan,
            "h": 0.0,
        }

    h = T / N

    u = np.exp(sigma * np.sqrt(h))
    d = np.exp(-sigma * np.sqrt(h))

    p_star = (np.exp((r - q) * h) - d) / (u - d)

    if p_star < 0 or p_star > 1:
        raise ValueError("Risk-neutral probability outside [0, 1]. Increase N or check inputs.")

    stock_tree = np.zeros((N + 1, N + 1))
    option_tree = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[i, j] = S0 * (u**j) * (d ** (i - j))

    for j in range(N + 1):
        S = stock_tree[N, j]

        if option_type == "call":
            option_tree[N, j] = max(S - K, 0.0)
        else:
            option_tree[N, j] = max(K - S, 0.0)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            S = stock_tree[i, j]

            continuation_value = np.exp(-r * h) * (
                p_star * option_tree[i + 1, j + 1]
                + (1 - p_star) * option_tree[i + 1, j]
            )

            if option_type == "call":
                exercise_value = max(S - K, 0.0)
            else:
                exercise_value = max(K - S, 0.0)

            option_tree[i, j] = max(continuation_value, exercise_value)

    return {
        "price": float(option_tree[0, 0]),
        "stock_tree": stock_tree,
        "option_tree": option_tree,
        "u": float(u),
        "d": float(d),
        "p_star": float(p_star),
        "h": float(h),
    }


def price_from_yfinance(ticker, K, r, T, N, option_type, vol_mode="surface", sigma=None):
    market_data = get_data(
        ticker=ticker,
        K=K,
        T=T,
        option_type=option_type,
        vol_mode=vol_mode,
        sigma=sigma,
    )

    result = binomial_american_option(
        S0=market_data["S0"],
        K=K,
        sigma=market_data["sigma"],
        q=market_data["q"],
        r=r,
        T=T,
        N=N,
        option_type=option_type,
    )

    return {
        "ticker": ticker,
        "S0": market_data["S0"],
        "K": K,
        "r": r,
        "T": T,
        "N": N,
        "sigma": market_data["sigma"],
        "q": market_data["q"],
        "option_type": option_type,
        "vol_mode": vol_mode,
        "price": result["price"],
        "u": result["u"],
        "d": result["d"],
        "p_star": result["p_star"],
        "h": result["h"],
    }


def print_result(result):
    _, _, euro_price = bs_price(
        result["S0"],
        result["K"],
        result["r"],
        result["q"],
        result["T"],
        result["sigma"],
        option_type=result["option_type"],
    )

    print("\n=== American Option Price ===")
    print(f"Ticker: {result['ticker']}")
    print(f"Option type: {result['option_type']}")
    print(f"Vol mode: {result['vol_mode']}")
    print(f"S0: {result['S0']:.2f}")
    print(f"K: {result['K']:.2f}")
    print(f"T: {result['T']:.4f} years")
    print(f"N: {result['N']}")
    print(f"r: {result['r']:.4f}")
    print(f"q: {result['q']:.4f}")
    print(f"Sigma: {result['sigma']:.4f}")
    print(f"u: {result['u']:.4f}")
    print(f"d: {result['d']:.4f}")
    print(f"p*: {result['p_star']:.4f}")
    print(f"European comparison price: {euro_price:.4f}")
    print(f"American price: {result['price']:.4f}")


if __name__ == "__main__":
    ticker = input("Ticker: ").upper()
    K = float(input("Strike price K: "))
    r = float(input("Risk-free rate r, e.g. 0.05 for 5%: "))
    T = float(input("Time to expiry T in years, e.g. 0.5 for 6 months: "))
    N = int(input("Number of steps N: "))
    option_type = input("Option type, call or put: ").lower()

    vol_mode = input("Vol mode, surface, realised, or manual: ").lower()

    manual_sigma = None
    if vol_mode == "manual":
        manual_sigma = float(input("Manual volatility sigma, e.g. 0.25: "))

    result = price_from_yfinance(
        ticker=ticker,
        K=K,
        r=r,
        T=T,
        N=N,
        option_type=option_type,
        vol_mode=vol_mode,
        sigma=manual_sigma,
    )

    print_result(result)