import numpy as np
import matplotlib.pyplot as plt
import Black_Scholes as bs


#parameters - same as GBM - not ingesting data yet (use yahoo finance api)
time_horizon = 365 # duration / days
T = 1 # years
sigma = 0.25 #volatility
r = 0.04 #risk-free rate
q = 0.015 #divedend yield
mu = r - q #drift
S0 = 152 # initial price
K = 160 # strike
delta_t = T/time_horizon
iterations = 10000


# random paths with GBM
def GBM_plot():
    t = np.linspace(0, time_horizon, time_horizon)
    s = np.zeros(time_horizon)
    s[0] = S0

    for i in range(1, time_horizon):
        Z = np.random.normal(0, 1)
        s[i] = s[i-1] * np.exp((mu - (0.5 * sigma **2))*delta_t + (sigma * np.sqrt(delta_t) * Z))

    return t, s


def plot_GBM_paths():

    time_horizon = 365  # duration / days
    T = 1  # years
    sigma = 0.25  # volatility
    r = 0.04  # risk-free rate
    q = 0.015  # divedend yield
    mu = r - q  # drift
    S0 = 152  # initial price
    K = 160  # strike
    delta_t = T / time_horizon
    iterations = 10000

    t = np.arange(time_horizon) * delta_t
    paths = np.zeros((iterations, time_horizon))
    paths[:, 0] = S0

    for i in range(iterations):
        _, s = GBM_plot()
        paths[i, :] = s

    terminal_prices = paths[:, -1] # final path value

    ST = paths[:, -1]
    payoffs = np.maximum(ST - K, 0.0) # payoff at maturity

    discount_factor = np.exp(-r * T)
    N = payoffs.size
    payoff_std = payoffs.std(ddof=1) # standard deviation
    se_payoff = payoff_std / np.sqrt(N) # standard error
    V = discount_factor * payoffs.mean()
    se_V = discount_factor * se_payoff # standard error in discounted price

    ci_low = V - 1.96 * se_V  #95 #confidence intervals
    ci_high = V + 1.96 * se_V
    mean_path = paths.mean(axis=0)
    median_path = np.percentile(paths, 50, axis=0)
    lower = np.percentile(paths, 2.5, axis=0)
    upper = np.percentile(paths, 97.5, axis=0)


    plt.figure(figsize=(12, 8))

    for i in range(min(iterations, 80)):
        plt.plot(t, paths[i, :], linewidth=1)

    plt.fill_between(t, lower, upper, alpha=0.3, label="95% distribution band (2.5–97.5 pct)") # distribution band
    plt.plot(t, median_path, linewidth=3, label="Median path") # median line
    plt.plot(t, mean_path, linestyle='-', linewidth=3, label="Mean path") # mean line
    plt.xlabel("Time (years)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()

    _,_,price = bs.bs_price(S0, K, r, q, T, sigma)
    print("Average Payoff: ", payoffs.mean())
    print("Estimated Option Price:", V)
    print("Black Scholes Price: ", price )
    return t, paths, terminal_prices

if __name__ == "__main__":
    plot_GBM_paths()



