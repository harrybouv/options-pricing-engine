import numpy as np
from Black_Scholes import bs_call_price

T = 1 # years
sigma = 0.25 #volatility
r = 0.04 #risk-free rate
q = 0.015 #divedend yield
S0 = 152 # initial price
K = 160 # strike

def gbm_terminal(S0, K, r, q, T, sigma, N, seed = 0):

    rng = np.random.default_rng(seed)
    M = N // 2
    Z = rng.standard_normal(M)
    Z = np.concatenate([Z, -Z])  # antithetic

    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discount_prices = np.exp(-r * T) * payoffs

    price = discount_prices.mean()
    SE = discount_prices.std(ddof=1) / np.sqrt(len(discount_prices)) # standard error
    ci = (price - 1.96 * SE, price + 1.96 * SE) # confidence intervals

    X = ST
    Y = discount_prices

    meanY = Y.mean()
    meanX = X.mean()

    cov = np.mean((Y - meanY) * (X - meanX))
    varX = np.mean((X - meanX) ** 2)
    b = cov / varX
    EX = S0 * np.exp((r - q) * T)
    Y_cv = Y - b * (X - EX)

    price_cv = float(Y_cv.mean())
    se_cv = float(Y_cv.std(ddof=1) / np.sqrt(len(Y_cv)))
    ci_cv = (price_cv - 1.96 * se_cv, price_cv + 1.96 * se_cv)

    return price, SE, ci, price_cv, se_cv,ci_cv



def main():

    price, SE, ci, price_cv, se_cv, ci_cv = gbm_terminal(S0, K, r, q, T, sigma, N=50000, seed = 0)
    _,_, bs = bs_call_price(S0, K, r, q, T, sigma)
    ci = (float(price - 1.96 * SE), float(price + 1.96 * SE))
    ci_cv = (float(price_cv - 1.96 * se_cv), float(price_cv + 1.96 * se_cv))

    print("Black Scholes Price: ", bs)
    print("Estimated Option Price:", price)
    print("SE:", SE, "CI:", ci)
    print("Price adjusted through control and antithetic variates: ", price_cv)
    print("Adjusted SE: ", se_cv, "Adjusted CI: ", ci_cv)
    print("Percentage Improvement in SE: ", ((SE-se_cv)/SE)*100)


if __name__ == "__main__":
    main()
