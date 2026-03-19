import numpy as np
from scipy.optimize import brentq
from Black_Scholes import bs_price, bs_greeks


def check_arbitrage(S, K, r, q, T, market_price, option_type="call"):
    option_type = option_type.lower()

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        lower_bound = max(S * disc_q - K * disc_r, 0.0)
        upper_bound = S * disc_q
    elif option_type == "put":
        lower_bound = max(K * disc_r - S * disc_q, 0.0)
        upper_bound = K * disc_r
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return lower_bound <= market_price <= upper_bound


def implied_vol_newton(
    S, K, r, q, T, market_price, option_type="call",
    sigma0=0.2, tol=1e-8, max_iter=100
):
    sigma_est = sigma0

    for _ in range(max_iter):
        _, _, model_price = bs_price(S, K, r, q, T, sigma_est, option_type=option_type)
        greeks = bs_greeks(S, K, r, q, T, sigma_est, option_type=option_type)
        vega = greeks["vega"]

        error = model_price - market_price

        if abs(error) < tol:
            return sigma_est

        if not np.isfinite(vega) or abs(vega) < 1e-12:
            raise ValueError("Vega too small for stable Newton iteration.")

        sigma_est = sigma_est - error / vega

        if sigma_est <= 0 or not np.isfinite(sigma_est):
            raise ValueError("Newton iteration produced invalid volatility.")

    raise RuntimeError("Newton method failed to converge within max_iter.")


def implied_vol_brent(
    S, K, r, q, T, market_price, option_type="call",
    sigma_low=1e-6, sigma_high=5.0
):
    def f(sigma):
        _, _, price = bs_price(S, K, r, q, T, sigma, option_type=option_type)
        return price - market_price

    f_low = f(sigma_low)
    f_high = f(sigma_high)

    if f_low * f_high > 0:
        raise ValueError("Brent bracket does not contain a root.")

    return brentq(f, sigma_low, sigma_high)


def implied_vol(S, K, r, q, T, market_price, option_type="call"):
    if T <= 0:
        raise ValueError("Implied vol undefined for T <= 0.")

    if market_price <= 0:
        raise ValueError("Market price must be positive.")

    if not check_arbitrage(S, K, r, q, T, market_price, option_type=option_type):
        raise ValueError("Market price violates no-arbitrage bounds.")

    try:
        return implied_vol_newton(S, K, r, q, T, market_price, option_type=option_type)
    except (ValueError, RuntimeError):
        return implied_vol_brent(S, K, r, q, T, market_price, option_type=option_type)