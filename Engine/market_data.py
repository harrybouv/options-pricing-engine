import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

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
    expiry_date = expiry_date.replace(hour = 23, minute = 59, second = 59)

    delta_seconds = (expiry_date - valuation_date).total_seconds()
    T = max(delta_seconds,0)/(365.25*24*60*60)

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
    df["option_type"] = option_type[:-1]

    return df

def get_chain(ticker: str, expiry: str, option_type: str = "calls"):

    S0 = get_spot_price(ticker)
    T = time_to_expiry(expiry)
    raw = load_option_chain(ticker, expiry, option_type=option_type)

    return S0, T, raw

def main():
    ticker = "AAPL"
    expiries = get_expiries(ticker)
    expiry = expiries[0]

    S0, T, raw = get_chain(ticker, expiry, option_type="calls")

    raw["mid"] = (raw["bid"] + raw["ask"]) / 2
    raw["market_price"] = raw["mid"]

    print(f"Ticker: {ticker}")
    print(f"Spot: {S0:.4f}")
    print(f"Expiry: {expiry}")
    print(f"T: {T:.6f} years")
    print(raw[["contractSymbol", "strike", "bid", "ask", "mid", "market_price"]].head(10))


if __name__ == "__main__":
    main()