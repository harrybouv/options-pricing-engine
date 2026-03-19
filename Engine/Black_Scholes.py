import numpy as np
from scipy.stats import norm

#parameters - same as GBM - not ingesting data yet (use yahoo finance api)
T = 1 # years
sigma = 0.25 #volatility
r = 0.04 #risk-free rate
q = 0.015 #divedend yield
S0 = 152 # initial price
K = 160 # strike CHANGE  <------


def bs_price(S, K, r, q, T, sigma, option_type="call"):
    option_type = option_type.lower()

    if T <= 0:
        if option_type == "call":
            return 0, 0, max(S - K, 0)
        else:
            return 0, 0, max (K - S, 0)

    if sigma <= 0:
        forward = S * np.exp((r - q) * T)
        disc = np.exp(-r * T)

        if option_type == "call":
            price = disc * max(forward - K, 0.0)
        else:
            price = disc * max(K - forward, 0.0)

        return 0.0, 0.0, price

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    return d1, d2, price

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
            "d1": float(d1),
            "d2": float(d2),
        }

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    sqrtT = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)
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
    _, _, bs = bs_call_price(S0, K, r, q, T, sigma)
    g = bs_call_greeks(S0, K, r, q, T, sigma)

    print("Black Scholes Price:", round(bs, 5))
    print("delta:", round(g["delta"], 5))
    print("gamma:", round(g["gamma"], 5))
    print("vega :", round(g["vega"], 5))
    print("rho  :", round(g["rho"], 5))

if __name__ == '__main__':
    main()