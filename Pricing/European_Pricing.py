import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Engine.Black_Scholes import bs_price, bs_greeks
from Engine.market_data import get_spot_price, get_dividend_yield, get_volatility


def get_stock_inputs(ticker, K, T, vol_mode="surface", sigma=None, option_type="call"):
    S = get_spot_price(ticker)
    q = get_dividend_yield(ticker)
    r = 0.04

    surface_type = "calls" if option_type == "call" else "puts"

    sigma = get_volatility(
        ticker=ticker,
        K=K,
        T=T,
        vol_mode=vol_mode,
        sigma=sigma,
        option_type=surface_type,
    )

    return S, sigma, r, q

def price_european_option(S, K, r, q, T, sigma, option_type="call"):
    _, _, price = bs_price(S, K, r, q, T, sigma, option_type=option_type)

    return price


def main():
    print("European Option Pricer")
    print("----------------------")

    ticker = input("Ticker, e.g. AAPL: ").upper()
    option_type = input("Option type, call or put: ").lower()
    K = float(input("Strike price K: "))
    T = float(input("Time to expiry in years, e.g. 0.5: "))

    vol_mode = input("Vol mode, surface, realised, or manual: ").lower()

    manual_sigma = None
    if vol_mode == "manual":
        manual_sigma = float(input("Manual volatility sigma, e.g. 0.25: "))

    S, sigma, r, q = get_stock_inputs(
        ticker=ticker,
        K=K,
        T=T,
        vol_mode=vol_mode,
        sigma=manual_sigma,
        option_type=option_type,
    )
    price = price_european_option(S, K, r, q, T, sigma, option_type=option_type)
    greeks = bs_greeks(S, K, r, q, T, sigma, option_type=option_type)

    print("\nInputs used:")
    print(f"Ticker: {ticker}")
    print(f"Spot price S: {S:.2f}")
    print(f"Strike K: {K:.2f}")
    print(f"Time T: {T:.4f} years")
    print(f"Risk-free rate r: {r:.4f}")
    print(f"Dividend yield q: {q:.4f}")
    print(f"Historical volatility sigma: {sigma:.4f}")

    print(f"\nEuropean {option_type} price: {price:.4f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f}")
    print(f"Rho: {greeks['rho']:.4f}")


if __name__ == "__main__":
    main()