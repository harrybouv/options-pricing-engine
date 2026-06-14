import numpy as np
from scipy.stats import norm


def bs_price(S, K, r, q, T, sigma, option_type="call"):
    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if T <= 0:
        if option_type == "call":
            return np.nan, np.nan, max(S - K, 0.0)
        return np.nan, np.nan, max(K - S, 0.0)

    if sigma <= 0:
        forward = S * np.exp((r - q) * T)
        disc = np.exp(-r * T)

        if option_type == "call":
            price = disc * max(forward - K, 0.0)
        else:
            price = disc * max(K - forward, 0.0)

        return np.nan, np.nan, price

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    return d1, d2, float(price)


def bs_greeks(S, K, r, q, T, sigma, option_type="call"):
    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    d1, d2, price = bs_price(S, K, r, q, T, sigma, option_type=option_type)

    if T <= 0 or sigma <= 0:
        return {
            "price": float(price),
            "delta": np.nan,
            "gamma": np.nan,
            "vega": np.nan,
            "theta": np.nan,
            "rho": np.nan,
            "d1": float(d1) if np.isfinite(d1) else np.nan,
            "d2": float(d2) if np.isfinite(d2) else np.nan,
        }

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    sqrtT = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "d1": float(d1),
        "d2": float(d2),
    }


def main():
    S0 = 152
    K = 160
    r = 0.04
    q = 0.015
    T = 1
    sigma = 0.25

    _, _, price = bs_price(S0, K, r, q, T, sigma, option_type="call")
    greeks = bs_greeks(S0, K, r, q, T, sigma, option_type="call")

    print("Black Scholes Price:", round(price, 5))
    print("delta:", round(greeks["delta"], 5))
    print("gamma:", round(greeks["gamma"], 5))
    print("vega :", round(greeks["vega"], 5))
    print("rho  :", round(greeks["rho"], 5))


if __name__ == "__main__":
    main()