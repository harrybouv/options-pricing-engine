import sys
from pathlib import Path
from Engine.market_data import get_spot_price, get_dividend_yield, get_volatility
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Engine.market_data import get_spot_price, get_dividend_yield
from Engine.realised_vol import get_realised_vol


def simulate_gbm_paths(S0, r, q, T, sigma, n_steps, n_paths, seed=42):
    rng = np.random.default_rng(seed)

    dt = T / n_steps

    Z = rng.normal(size=(n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)

    return S0 * np.exp(log_paths)


def price_arithmetic_asian(
    S0,
    K,
    r,
    q,
    T,
    sigma,
    n_steps=252,
    n_paths=100000,
    option_type="call",
    seed=42,
):
    S_paths = simulate_gbm_paths(S0, r, q, T, sigma, n_steps, n_paths, seed)
    avg_price = np.mean(S_paths, axis=1)

    if option_type == "call":
        payoffs = np.maximum(avg_price - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - avg_price, 0.0)
    else:
        raise ValueError("option_type must be either call or put")

    discounted_payoffs = np.exp(-r * T) * payoffs

    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))

    return price, standard_error


def asian_geometric_closed_form(S0, K, r, q, T, sigma, n_steps=252, option_type="call"):
    n = n_steps

    sigma_g = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n**2))

    b_g = 0.5 * (r - q - 0.5 * sigma**2) * ((n + 1) / n) + 0.5 * sigma_g**2

    d1 = (np.log(S0 / K) + (b_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    if option_type == "call":
        price = np.exp(-r * T) * (
            S0 * np.exp(b_g * T) * norm.cdf(d1)
            - K * norm.cdf(d2)
        )
    elif option_type == "put":
        price = np.exp(-r * T) * (
            K * norm.cdf(-d2)
            - S0 * np.exp(b_g * T) * norm.cdf(-d1)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(price)


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

    return S0, sigma, q


def main():

    n_steps = 252
    n_paths = 100000
    ticker = input("Ticker: ").upper()
    K = float(input("Strike price K: "))
    r = float(input("Risk-free rate r, e.g. 0.05 for 5%: "))
    T = float(input("Time to expiry T in years, e.g. 0.5: "))
    option_type = input("Option type, call or put: ").lower()

    vol_mode = input("Vol mode, surface, realised, or manual: ").lower()

    manual_sigma = None
    if vol_mode == "manual":
        manual_sigma = float(input("Manual volatility sigma, e.g. 0.25: "))

    S0, sigma, q = get_data(
        ticker=ticker,
        K=K,
        T=T,
        option_type=option_type,
        vol_mode=vol_mode,
        sigma=manual_sigma,
    )

    arithmetic_price, arithmetic_se = price_arithmetic_asian(
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        sigma=sigma,
        n_steps=n_steps,
        n_paths=n_paths,
        option_type=option_type,
    )

    geometric_price = asian_geometric_closed_form(
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        sigma=sigma,
        n_steps=n_steps,
        option_type=option_type,
    )

    print(f"Ticker:                  {ticker}")
    print(f"Current stock price S0:  {S0:.2f}")
    print(f"Strike K:                {K:.2f}")
    print(f"Volatility sigma:        {sigma:.2%}")
    print(f"Risk-free rate r:        {r:.2%}")
    print(f"Dividend yield q:        {q:.2%}")
    print(f"Maturity T:              {T} year(s)")
    print()

    ci_low = arithmetic_price - 1.96 * arithmetic_se
    ci_high = arithmetic_price + 1.96 * arithmetic_se

    print(f"Arithmetic Asian {option_type}: {arithmetic_price:.4f}")
    print(f"Monte Carlo SE:              {arithmetic_se:.4f}")
    print(f"95% CI:                      [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Geometric Asian {option_type}:  {geometric_price:.4f}")

if __name__ == "__main__":
    main()