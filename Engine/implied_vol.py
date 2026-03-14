from Black_Scholes import bs_call_price, bs_call_greeks
import numpy as np
from scipy.optimize import brentq


#parameters - same as GBM - not ingesting data yet (use yahoo finance api)
T = 1 # years
sigma = 0.25 #volatility
r = 0.04 #risk-free rate
q = 0.015 #divedend yield
S0 = 152 # initial price
K = 160 # strike CHANGE  <------


def check_arbitrage(S0, K, r, q, T, market_price):
    lower_bound = max(S0*np.exp(-q*T) - K*np.exp(-r*T), 0)
    upper_bound = S0* np.exp(-q*T)

    if lower_bound <= market_price <= upper_bound:
        return True
    else:
        return False


def implied_vol_newton(S0, K, r, q, T, market_price, sigma0=0.2, tol=1e-8, max_iter=100):

    for _ in range(max_iter):
        _, _, model_price = bs_call_price(S0, K, r, q, T, sigma)
        greeks = bs_call_greeks(S0, K, r, q, T, sigma)
        vega = greeks["vega"]

        error = model_price - market_price

        if abs(error) < tol:
            return sigma

        if abs(vega) < 1e-12:
            raise ValueError("Vega too small for stable Newton iteration.")

        sigma = sigma - error / vega

        if sigma <= 0:
            raise ValueError("Newton iteration produced non-positive volatility.")

    raise RuntimeError("Newton method failed to converge within max_iter.")


def implied_vol_brent(S0, K, r, q, T, market_price, sigma_low=1e-6, sigma_high=5.0):

    def f(sigma):
        return bs_call_price(S0, K, r, q, T, sigma) - market_price

    f_low = f(sigma_low)
    f_high = f(sigma_high)

    if f_low * f_high > 0:
        raise ValueError("Brent bracket does not contain a root.")

    return brentq(f, sigma_low, sigma_high)


def main():
    _, _, market_price = bs_call_price(S0, K, r, q, T, sigma)
    newton_success = False

    if check_arbitrage(S0, K, r, q, T, market_price):
        try:
            iv = implied_vol_newton(S0, K, r, q, T, market_price)
            print("Implied volatility: ", round(iv,3))
            newton_success = True
            return newton_success
        except ValueError:
            print("Vega too small for stable Newton iteration.")
        except RuntimeError:
            print("Newton method failed to converge.")

    else:
        print("Market price violates no-arbitrage bounds.")

    if not newton_success:
        print("Implied volatility:",round (implied_vol_brent(S0, K, r, q, T, market_price),3))



if __name__ == '__main__':
    main()
