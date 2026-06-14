import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def payoff(S,K, option_type):
    option_type = option_type.lower()

    if option_type == 'call':
        return np.maximum(S - K, 0)
    if option_type == 'put':
        return np.maximum(K - S, 0)

    raise ValueError('Invalid option type')

def simulate_gbm_paths(S0, r, q, T, sigma, n_steps, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    Z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:,0] = S0
    paths[:,1:] = S0 * np.exp(log_paths)

    return paths

def regression_basis(S, K):
    x = S / K
    return np.column_stack([np.ones_like(x), x, x**2])

def longstaff_schwartz_bermudan(
        S0,
        K,
        r,
        q,
        T,
        sigma,
        n_steps=252,
        n_paths=100000,
        n_exercise_dates=12,
        option_type="put",
        seed=42,
):
    option_type = option_type.lower()

    paths = simulate_gbm_paths(S0, r, q, T, sigma, n_steps, n_paths, seed)
    dt = T / n_steps
    times = np.arange(n_steps+1) * dt

    exercise_steps = np.linspace(0, n_steps, n_exercise_dates+1, dtype=int)[1:]
    exercise_steps = np.unique(exercise_steps)

    exercise_index = np.full(n_paths, n_steps)
    cashflow = payoff(paths[:, -1], K, option_type)

    for step in exercise_steps[-2::-1]:
        S_t = paths[:, step]
        immediate = payoff(S_t, K, option_type)
        in_money = immediate > 0

        if in_money.sum() < 3:
            continue

        discounted_future = cashflow * np.exp(-r * (times[exercise_index] - times[step]))

        x = regression_basis(S_t[in_money], K)
        y = discounted_future[in_money]

        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        continuation = x @ coeffs

        exercise_now_itm = immediate[in_money] > continuation

        exercise_now = np.zeros(n_paths, dtype=bool)
        exercise_now[in_money] = exercise_now_itm

        cashflow[exercise_now] = immediate[exercise_now]
        exercise_index[exercise_now] = step

    discounted_cashflows = cashflow * np.exp(-r*times[exercise_index])

    price = float(discounted_cashflows.mean())
    se = float(discounted_cashflows.std(ddof=1) / np.sqrt(n_paths))

    return {
        "price": price,
        "standard_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
        "exercise_steps": exercise_steps,
        "exercise_times": times[exercise_steps],
    }

def get_data(ticker, K, T, option_type, vol_mode='surface', sigma=None):
    try:
        from Engine.market_data import get_spot_price, get_dividend_yield, get_volatility
    except ModuleNotFoundError:
        from market_data import get_spot_price, get_dividend_yield, get_volatility

    S0 = get_spot_price(ticker)
    q = get_dividend_yield(ticker)

    surface_type = 'calls' if option_type == 'call' else 'puts'

    sigma = get_volatility(
        ticker=ticker,
        K=K,
        T=T,
        vol_mode=vol_mode,
        sigma=sigma,
        option_type=surface_type,
    )

    return S0, sigma, q

def price_from_yfinance(
    ticker,
    K,
    r,
    T,
    option_type,
    vol_mode="surface",
    sigma=None,
    n_steps=252,
    n_paths=100000,
    n_exercise_dates=12,
):
    try:
        from Engine.Black_Scholes import bs_price
    except ModuleNotFoundError:
        from Black_Scholes import bs_price

    S0, sigma, q = get_data(ticker, K, T, option_type, vol_mode, sigma)

    result = longstaff_schwartz_bermudan(
        S0=S0,
        K=K,
        r=r,
        q=q,
        T=T,
        sigma=sigma,
        n_steps=n_steps,
        n_paths=n_paths,
        n_exercise_dates=n_exercise_dates,
        option_type=option_type,
    )

    _, _, european_price = bs_price(S0, K, r, q, T, sigma, option_type=option_type)

    result.update({
        "ticker": ticker,
        "S0": S0,
        "K": K,
        "r": r,
        "q": q,
        "T": T,
        "sigma": sigma,
        "option_type": option_type,
        "vol_mode": vol_mode,
        "european_price": european_price,
        "n_steps": n_steps,
        "n_paths": n_paths,
        "n_exercise_dates": n_exercise_dates,
    })

    return result

def print_result(result):
    print("\n=== Bermudan Option Price: Longstaff-Schwartz Monte Carlo ===")
    print(f"Ticker: {result['ticker']}")
    print(f"Option type: {result['option_type']}")
    print(f"Vol mode: {result['vol_mode']}")
    print(f"S0: {result['S0']:.2f}")
    print(f"K: {result['K']:.2f}")
    print(f"T: {result['T']:.4f} years")
    print(f"r: {result['r']:.4f}")
    print(f"q: {result['q']:.4f}")
    print(f"Sigma: {result['sigma']:.4f}")
    print(f"Steps: {result['n_steps']}")
    print(f"Paths: {result['n_paths']}")
    print(f"Exercise dates: {result['n_exercise_dates']}")
    print(f"European comparison price: {result['european_price']:.4f}")
    print(f"Bermudan LSM price: {result['price']:.4f}")
    print(f"Monte Carlo SE: {result['standard_error']:.4f}")
    print(f"95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")


def main():
    ticker = input("Ticker: ").upper()
    option_type = input("Option type, call or put: ").lower()
    K = float(input("Strike price K: "))
    r = float(input("Risk-free rate r, e.g. 0.05: "))
    T = float(input("Time to expiry T in years, e.g. 0.5: "))

    n_steps = int(input("Simulation steps, e.g. 252: "))
    n_paths = int(input("Number of paths, e.g. 100000: "))
    n_exercise_dates = int(input("Number of Bermudan exercise dates, e.g. 12: "))

    vol_mode = input("Vol mode, surface, realised, or manual: ").lower()

    manual_sigma = None
    if vol_mode == "manual":
        manual_sigma = float(input("Manual volatility sigma, e.g. 0.25: "))

    result = price_from_yfinance(
        ticker=ticker,
        K=K,
        r=r,
        T=T,
        option_type=option_type,
        vol_mode=vol_mode,
        sigma=manual_sigma,
        n_steps=n_steps,
        n_paths=n_paths,
        n_exercise_dates=n_exercise_dates,
    )

    print_result(result)


if __name__ == "__main__":
    main()