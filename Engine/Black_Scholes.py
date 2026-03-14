import numpy as np
from scipy.stats import norm

#parameters - same as GBM - not ingesting data yet (use yahoo finance api)
T = 1 # years
sigma = 0.25 #volatility
r = 0.04 #risk-free rate
q = 0.015 #divedend yield
S0 = 152 # initial price
K = 160 # strike CHANGE  <------


def bs_call_price(S0, K, r, q, T, sigma):

    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        # deterministic limit
        forward = S0 * np.exp((r - q) * T)
        return np.exp(-r * T) * max(forward - K, 0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2, S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_greeks(S, K, r, q, T, sigma):
    d1, d2, _ = bs_call_price(S, K, r, q, T, sigma)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    sqrtT = np.sqrt(T)

    price = S * disc_q * Nd1 - K * disc_r * Nd2
    delta = disc_q * Nd1
    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT
    theta = -(S * disc_q * pdf_d1 * sigma) / (2 * sqrtT) - r * K * disc_r * Nd2 + q * S * disc_q * Nd1
    rho = K * T * disc_r * Nd2

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