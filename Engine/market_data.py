import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from implied_vol import implied_vol
import matplotlib.pyplot as plt
from Black_Scholes import bs_greeks

def get_spot_price(ticker:str) -> float:

    tk = yf.Ticker(ticker)
    info = tk.fast_info

    S0 = info.get("lastPrice", None)
    if S0 is None:
        S0 = info.get("regularMarketPrice", None)
    if S0 is None or not np.isfinite(S0):
        raise ValueError(f"Could not retrieve valid spot price for {ticker}")

    return float(S0)

def get_expiries(ticker:str) -> list[str]:

    tk = yf.Ticker(ticker)
    expiries = tk.options

    if not expiries:
        raise ValueError(f"No expiries returned for {ticker}")

    return expiries

def time_to_expiry(expiry_str: str, valuation_date: datetime | None = None) -> float:

    if valuation_date is None:
        valuation_date = datetime.now(timezone.utc)

    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    expiry_date = expiry_date.replace(hour=23, minute=59, second=59)

    delta_seconds = (expiry_date - valuation_date).total_seconds()
    T = max(delta_seconds, 0) / (365.25 * 24 * 60 * 60)

    return T

def load_option_chain(ticker: str, expiry: str, option_type: str = "calls") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)

    if option_type == "calls":
        df = chain.calls.copy()
    elif option_type == "puts":
        df = chain.puts.copy()
    else:
        raise ValueError("Option type must be 'calls' or 'puts'")

    if df.empty:
        raise ValueError("Option chain is empty")

    df["ticker"] = ticker.upper()
    df["expiry"] = expiry
    if option_type == "calls":
        df["option_type"] = "call"
    elif option_type == "puts":
        df["option_type"] = "put"

    return df

def get_chain(ticker: str, expiry: str, option_type: str = "calls"):
    S0 = get_spot_price(ticker)
    T = time_to_expiry(expiry)
    raw = load_option_chain(ticker, expiry, option_type=option_type)
    r = 0.04
    q = get_dividend_yield(ticker)

    raw["mid"] = (raw["bid"] + raw["ask"]) / 2

    raw["price_source"] = np.where(
        (raw["bid"] > 0) & (raw["ask"] > 0),
        "mid",
        "last"
    )

    raw["market_price"] = np.where(
        raw["price_source"] == "mid",
        raw["mid"],
        raw["lastPrice"]
    )

    return S0, T, r, q, raw

def get_dividend_yield(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    info = tk.info

    q = info.get("dividendYield", None)

    if q is None or not np.isfinite(q):
        q = 0.0

    return float(q)

def implied_vol_column(raw: pd.DataFrame, S0: float, T: float, r: float, q: float) -> pd.DataFrame:
    sigmas = []

    for _, row in raw.iterrows():
        K = row["strike"]
        market_price = row["market_price"]
        opt_type = row["option_type"]

        try:
            sigma = implied_vol(
                S0,
                K,
                r,
                q,
                T,
                market_price,
                option_type=opt_type
            )
        except Exception:
            sigma = np.nan

        sigmas.append(sigma)

    raw["sigma"] = sigmas
    return raw

def plot_smile_strike(df: pd.DataFrame, S0: float, expiry: str, ticker: str):
    plt.figure(figsize=(8, 5))
    plt.plot(df["strike"], df["sigma"], marker="o", linestyle="-")
    plt.axvline(S0, linestyle="--", label=f"Spot = {S0:.2f}")

    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title(f"{ticker} Implied Volatility Smile ({expiry})")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_smile_moneyness(df: pd.DataFrame, expiry: str, ticker: str):
    plt.figure(figsize=(8, 5))
    plt.plot(df["moneyness"], df["sigma"], marker="o", linestyle="-")

    plt.xlabel("Moneyness (K / S0)")
    plt.ylabel("Implied Volatility")
    plt.title(f"{ticker} Implied Volatility Smile ({expiry})")
    plt.grid(True)
    plt.show()

def plot_multiple_smiles(smile_data: dict[str, pd.DataFrame], ticker: str, S0: float):
    plt.figure(figsize=(10, 6))

    for expiry, df in smile_data.items():
        plt.plot(df["strike"], df["sigma"], marker="o", linestyle="-", label=expiry)

    plt.axvline(S0, linestyle="--", label=f"Spot = {S0:.2f}")

    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title(f"{ticker} Implied Volatility Smiles Across Expiries")
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_smile_data(
    ticker: str,
    expiry: str,
    option_type: str = "calls"
) -> tuple[float, float, float, float, pd.DataFrame]:
    S0, T, r, q, raw = get_chain(ticker, expiry, option_type=option_type)

    raw = raw.dropna(subset=["strike", "market_price"]).copy()
    raw = raw[raw["market_price"] > 0]

    raw = implied_vol_column(raw, S0, T, r, q)

    clean = raw.dropna(subset=["sigma"]).copy()
    clean = greeks_column(clean, S0, T, r, q)
    clean["T"] = T
    clean["moneyness"] = clean["strike"] / S0
    clean["spot"] = S0

    clean = clean[
        (clean["moneyness"] > 0.9) &
        (clean["moneyness"] < 1.1) &
        (clean["market_price"] > 0.05) &
        (clean["sigma"] > 0.01) &
        (clean["sigma"] < 1.0)
        ].copy()

    clean["spread"] = clean["ask"].fillna(0) - clean["bid"].fillna(0)

    clean = clean[
        (clean["bid"].fillna(0) >= 0) &
        (clean["ask"].fillna(0) >= 0)
        ].copy()

    clean = clean[
        (clean["spread"] >= 0)
    ].copy()

    clean = clean.sort_values("moneyness")
    return S0, T, r, q, clean

def greeks_column(raw: pd.DataFrame, S0: float, T: float, r: float, q: float) -> pd.DataFrame:

    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []

    for _, row in raw.iterrows():
        K = row["strike"]
        sigma = row["sigma"]

        try:
            greeks = bs_greeks(S0, K, r, q, T, sigma, option_type=row["option_type"])
            delta = greeks["delta"]
            gamma = greeks["gamma"]
            vega = greeks["vega"]
            theta = greeks["theta"]
            rho = greeks["rho"]
        except Exception:
            delta = gamma = vega = theta = rho = np.nan

        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)
        rhos.append(rho)

    out = raw.copy()
    out["delta"] = deltas
    out["gamma"] = gammas
    out["vega"] = vegas
    out["theta"] = thetas
    out["rho"] = rhos

    return out

def market_data(ticker: str, option_type: str = "calls", start: int = 2, end: int = 8):
    expiries = get_expiries(ticker)[start:end]

    smile_data = {}
    surface_frames = []
    S0_ref = None
    r_ref = None
    q_ref = None

    for expiry in expiries:
        try:
            S0, T, r, q, clean = prepare_smile_data(ticker, expiry, option_type=option_type)

            if S0_ref is None:
                S0_ref = S0
                r_ref = r
                q_ref = q

            smile_data[expiry] = clean
            surface_frames.append(clean)

        except Exception as e:
            print(f"Failed for expiry {expiry}: {e}")

    if not smile_data:
        raise ValueError(f"No valid smile data retrieved for {ticker}")

    surface_df = pd.concat(surface_frames, ignore_index=True)

    return {
        "ticker": ticker.upper(),
        "spot": S0_ref,
        "r": r_ref,
        "q": q_ref,
        "smile_data": smile_data,
        "surface_df": surface_df,
    }

def main():
    ticker = "AAPL"

    for option_type in ["calls", "puts"]:
        print(f"\n===== {option_type.upper()} =====")
        data = market_data(ticker, option_type=option_type)

        for expiry, clean in data["smile_data"].items():
            print(f"\nExpiry: {expiry}")
            print(clean[["contractSymbol", "strike", "option_type", "sigma", "delta", "gamma", "vega", "theta", "rho"]].head(5))

        plot_multiple_smiles(data["smile_data"], f"{data['ticker']} {option_type.upper()}", data["spot"])

if __name__ == "__main__":
    main()