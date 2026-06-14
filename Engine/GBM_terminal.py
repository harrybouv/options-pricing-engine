import numpy as np
import matplotlib.pyplot as plt

try:
    from Engine.Black_Scholes import bs_price
except ModuleNotFoundError:
    from Black_Scholes import bs_price


T = 1
sigma = 0.25
r = 0.04
q = 0.015
S0 = 152
K = 160


def confidence_interval(price, se):
    return float(price - 1.96 * se), float(price + 1.96 * se)


def terminal_stock_prices(S0, r, q, T, sigma, N, seed=0, antithetic=False):
    rng = np.random.default_rng(seed)

    if antithetic:
        M = N // 2
        Z = rng.standard_normal(M)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(N)

    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    return ST


def discounted_call_payoffs(ST, K, r, T):
    payoffs = np.maximum(ST - K, 0.0)
    discount_prices = np.exp(-r * T) * payoffs

    return discount_prices


def price_from_discounted_payoffs(discount_prices):
    price = float(discount_prices.mean())
    se = float(discount_prices.std(ddof=1) / np.sqrt(len(discount_prices)))
    ci = confidence_interval(price, se)

    return price, se, ci


def apply_terminal_control_variate(ST, discount_prices, S0, r, q, T):
    X = ST
    Y = discount_prices

    meanY = Y.mean()
    meanX = X.mean()

    cov = np.mean((Y - meanY) * (X - meanX))
    varX = np.mean((X - meanX) ** 2)

    if varX <= 0 or not np.isfinite(varX):
        return Y

    b = cov / varX
    EX = S0 * np.exp((r - q) * T)
    Y_cv = Y - b * (X - EX)

    return Y_cv


def gbm_terminal(S0, K, r, q, T, sigma, N, seed=0):
    ST = terminal_stock_prices(S0, r, q, T, sigma, N, seed=seed, antithetic=False)
    discount_prices = discounted_call_payoffs(ST, K, r, T)

    price, SE, ci = price_from_discounted_payoffs(discount_prices)

    Y_cv = apply_terminal_control_variate(ST, discount_prices, S0, r, q, T)
    price_cv, se_cv, ci_cv = price_from_discounted_payoffs(Y_cv)

    return price, SE, ci, price_cv, se_cv, ci_cv


def gbm_terminal_antithetic(S0, K, r, q, T, sigma, N, seed=0):
    ST = terminal_stock_prices(S0, r, q, T, sigma, N, seed=seed, antithetic=True)
    discount_prices = discounted_call_payoffs(ST, K, r, T)

    price, SE, ci = price_from_discounted_payoffs(discount_prices)

    return price, SE, ci


def gbm_terminal_control(S0, K, r, q, T, sigma, N, seed=0):
    ST = terminal_stock_prices(S0, r, q, T, sigma, N, seed=seed, antithetic=False)
    discount_prices = discounted_call_payoffs(ST, K, r, T)

    Y_cv = apply_terminal_control_variate(ST, discount_prices, S0, r, q, T)
    price_cv, se_cv, ci_cv = price_from_discounted_payoffs(Y_cv)

    return price_cv, se_cv, ci_cv


def gbm_terminal_antithetic_control(S0, K, r, q, T, sigma, N, seed=0):
    ST = terminal_stock_prices(S0, r, q, T, sigma, N, seed=seed, antithetic=True)
    discount_prices = discounted_call_payoffs(ST, K, r, T)

    Y_cv = apply_terminal_control_variate(ST, discount_prices, S0, r, q, T)
    price_cv, se_cv, ci_cv = price_from_discounted_payoffs(Y_cv)

    return price_cv, se_cv, ci_cv


def plot_se_curve(S0, K, r, q, T, sigma, seed=0):
    sample_sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    se_plain = []
    se_antithetic = []
    se_control = []
    se_antithetic_control = []

    for N in sample_sizes:
        _, SE, _, _, _, _ = gbm_terminal(S0, K, r, q, T, sigma, N=N, seed=seed)
        _, se_av, _ = gbm_terminal_antithetic(S0, K, r, q, T, sigma, N=N, seed=seed)
        _, se_cv, _ = gbm_terminal_control(S0, K, r, q, T, sigma, N=N, seed=seed)
        _, se_av_cv, _ = gbm_terminal_antithetic_control(S0, K, r, q, T, sigma, N=N, seed=seed)

        se_plain.append(SE)
        se_antithetic.append(se_av)
        se_control.append(se_cv)
        se_antithetic_control.append(se_av_cv)

    plt.figure(figsize=(12, 8))
    plt.plot(sample_sizes, se_plain, marker="o", linewidth=2, label="Plain Monte Carlo")
    plt.plot(sample_sizes, se_antithetic, marker="o", linewidth=2, label="Antithetic")
    plt.plot(sample_sizes, se_control, marker="o", linewidth=2, label="Control Variate")
    plt.plot(sample_sizes, se_antithetic_control, marker="o", linewidth=2, label="Antithetic + Control")
    plt.xscale("log")
    plt.xlabel("Number of simulations")
    plt.ylabel("Standard error")
    plt.title("Monte Carlo Standard Error vs Number of Simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_ci_comparison(S0, K, r, q, T, sigma, N=50000, seed=0):
    price, SE, ci, _, _, _ = gbm_terminal(S0, K, r, q, T, sigma, N=N, seed=seed)
    price_av, se_av, ci_av = gbm_terminal_antithetic(S0, K, r, q, T, sigma, N=N, seed=seed)
    price_control, se_control, ci_control = gbm_terminal_control(S0, K, r, q, T, sigma, N=N, seed=seed)
    price_av_cv, se_av_cv, ci_av_cv = gbm_terminal_antithetic_control(S0, K, r, q, T, sigma, N=N, seed=seed)

    _, _, bs = bs_price(S0, K, r, q, T, sigma)

    labels = ["Plain", "Antithetic", "Control", "Antithetic + Control"]
    prices = [price, price_av, price_control, price_av_cv]
    errors = [1.96 * SE, 1.96 * se_av, 1.96 * se_control, 1.96 * se_av_cv]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 8))
    plt.errorbar(x, prices, yerr=errors, fmt="o", capsize=6, linewidth=2, markersize=8, label="Monte Carlo estimate with 95% CI")
    plt.axhline(bs, linestyle="--", linewidth=2, label="Black Scholes Price")
    plt.xticks(x, labels)
    plt.ylabel("Option price")
    plt.title(f"Option Price Estimates and 95% Confidence Intervals, N = {N}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_variates_comparison():
    price, SE, ci, price_cv, se_cv, ci_cv = gbm_terminal(S0, K, r, q, T, sigma, N=50000, seed=0)
    price_av, se_av, ci_av = gbm_terminal_antithetic(S0, K, r, q, T, sigma, N=50000, seed=0)
    price_control, se_control, ci_control = gbm_terminal_control(S0, K, r, q, T, sigma, N=50000, seed=0)
    price_av_cv, se_av_cv, ci_av_cv = gbm_terminal_antithetic_control(S0, K, r, q, T, sigma, N=50000, seed=0)

    _, _, bs = bs_price(S0, K, r, q, T, sigma)

    print("Black Scholes Price: ", bs)

    print("Plain Monte Carlo Price:", price)
    print("SE:", SE, "CI:", ci)

    print("Antithetic Price:", price_av)
    print("SE:", se_av, "CI:", ci_av)

    print("Control Variate Price:", price_control)
    print("SE:", se_control, "CI:", ci_control)

    print("Antithetic + Control Price:", price_av_cv)
    print("SE:", se_av_cv, "CI:", ci_av_cv)

    print("Percentage Improvement in SE from Plain to Antithetic: ", ((SE - se_av) / SE) * 100)
    print("Percentage Improvement in SE from Plain to Control: ", ((SE - se_control) / SE) * 100)
    print("Percentage Improvement in SE from Plain to Antithetic + Control: ", ((SE - se_av_cv) / SE) * 100)

    plot_se_curve(S0, K, r, q, T, sigma, seed=0)
    plot_ci_comparison(S0, K, r, q, T, sigma, N=50000, seed=0)


def main():
    plot_variates_comparison()


if __name__ == "__main__":
    main()