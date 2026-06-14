import sys
from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from Engine.market_data import market_data
from Engine.Black_Scholes import bs_price, bs_greeks
from Engine.volatility_surface import build_surface_grid

try:
    from Pricing.American_pricing import binomial_american_option
except Exception:
    binomial_american_option = None

try:
    from Pricing.Asian_pricing import price_arithmetic_asian
except Exception:
    price_arithmetic_asian = None

try:
    from Pricing.Bermudan_pricing import longstaff_schwartz_bermudan
except Exception:
    longstaff_schwartz_bermudan = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Options Volatility Workstation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def clean_value(x):
    """Convert numpy/pandas values into JSON-safe Python values."""
    if x is None:
        return None

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x):
            return None
        return float(x)

    if isinstance(x, (np.ndarray,)):
        return x.tolist()

    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()

    if pd.isna(x):
        return None

    return x


def clean_records(df: pd.DataFrame, max_rows: Optional[int] = None):
    """Convert DataFrame to JSON-safe list of dictionaries."""
    if df is None or df.empty:
        return []

    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)

    if max_rows is not None:
        out = out.head(max_rows)

    records = out.to_dict(orient="records")

    return [
        {str(k): clean_value(v) for k, v in row.items()}
        for row in records
    ]


def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def latest_price_stats(hist: pd.DataFrame):
    close = get_series(hist, "Close")

    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2]) if len(close) > 1 else np.nan

    change = latest - previous if np.isfinite(previous) else np.nan
    change_pct = change / previous if np.isfinite(previous) and previous != 0 else np.nan

    returns = np.log(close / close.shift(1)).dropna()

    rv_20 = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else np.nan
    rv_50 = float(returns.tail(50).std() * np.sqrt(252)) if len(returns) >= 50 else np.nan

    return {
        "latest": clean_value(latest),
        "previous": clean_value(previous),
        "change": clean_value(change),
        "change_pct": clean_value(change_pct),
        "rv_20": clean_value(rv_20),
        "rv_50": clean_value(rv_50),
        "high_20": clean_value(float(close.tail(20).max())),
        "low_20": clean_value(float(close.tail(20).min())),
    }


def normalise_history(hist: pd.DataFrame) -> pd.DataFrame:
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist = hist.dropna(how="all").copy()
    hist = hist.reset_index()

    if "Date" not in hist.columns:
        hist = hist.rename(columns={hist.columns[0]: "Date"})

    hist["Date"] = hist["Date"].astype(str)

    wanted = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    existing = [c for c in wanted if c in hist.columns]

    return hist[existing]


def add_contract_fields(surface_df: pd.DataFrame) -> pd.DataFrame:
    df = surface_df.copy()

    if "contractSymbol" not in df.columns:
        df["contractSymbol"] = ""

    if "mid" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
        elif "market_price" in df.columns:
            df["mid"] = df["market_price"]
        else:
            df["mid"] = np.nan

    if "relative_spread" not in df.columns:
        if "bid" in df.columns and "ask" in df.columns:
            mid = (df["bid"] + df["ask"]) / 2
            df["relative_spread"] = np.where(mid > 0, (df["ask"] - df["bid"]) / mid, np.nan)
        else:
            df["relative_spread"] = np.nan

    if "vega" in df.columns:
        df["vega_1pct"] = df["vega"] / 100
    else:
        df["vega_1pct"] = np.nan

    if "theta" in df.columns:
        df["theta_day"] = df["theta"] / 365
    else:
        df["theta_day"] = np.nan

    if "rho" in df.columns:
        df["rho_1pct"] = df["rho"] / 100
    else:
        df["rho_1pct"] = np.nan

    df["contract_id"] = df.apply(
        lambda row: str(row["contractSymbol"])
        if str(row["contractSymbol"]).strip()
        else f"{row.get('expiry')}_{row.get('strike')}_{row.get('option_type')}",
        axis=1,
    )

    return df


def choose_default_contract(df: pd.DataFrame):
    if df.empty:
        return None

    tmp = df.copy()

    if "moneyness" not in tmp.columns or "T" not in tmp.columns:
        return clean_records(tmp.head(1))[0]

    target_T = 45 / 365.25

    tmp["selection_score"] = (
        (tmp["moneyness"] - 1.0).abs() * 2.0
        + (tmp["T"] - target_T).abs()
    )

    row = tmp.loc[tmp["selection_score"].idxmin()]

    return {str(k): clean_value(v) for k, v in row.to_dict().items()}


def surface_quality(df: pd.DataFrame):
    n_contracts = len(df)
    n_expiries = int(df["expiry"].nunique()) if "expiry" in df.columns else 0

    median_spread = (
        float(df["relative_spread"].median())
        if "relative_spread" in df.columns and df["relative_spread"].notna().any()
        else np.nan
    )

    median_oi = (
        float(df["openInterest"].median())
        if "openInterest" in df.columns and df["openInterest"].notna().any()
        else np.nan
    )

    score = 0
    score += int(n_contracts >= 30)
    score += int(n_expiries >= 4)
    score += int(np.isfinite(median_spread) and median_spread < 0.12)
    score += int(np.isfinite(median_oi) and median_oi > 250)

    if score >= 4:
        label = "strong"
    elif score == 3:
        label = "usable"
    elif score == 2:
        label = "thin"
    else:
        label = "weak"

    return {
        "label": label,
        "score": score,
        "contracts": n_contracts,
        "expiries": n_expiries,
        "median_spread": clean_value(median_spread),
        "median_open_interest": clean_value(median_oi),
    }


def make_surface_grid(df: pd.DataFrame):
    try:
        grid = build_surface_grid(
            df,
            value_col="sigma",
            m_min=0.90,
            m_max=1.10,
            n_moneyness=70,
        )
        return clean_records(grid)
    except Exception:
        return []


# ============================================================
# CACHE DATA LOADS
# ============================================================

@lru_cache(maxsize=16)
def cached_market_data(ticker: str, side: str):
    return market_data(ticker.upper(), option_type=side)


@lru_cache(maxsize=16)
def cached_history(ticker: str, period: str):
    hist = yf.download(
        ticker.upper(),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if hist.empty:
        raise ValueError(f"No historical data returned for {ticker}")

    return hist


# ============================================================
# API MODELS
# ============================================================

class PriceRequest(BaseModel):
    model: Literal["European", "American", "Asian", "Bermudan"] = "European"
    S0: float
    K: float
    r: float = 0.05
    q: float = 0.0
    T: float
    sigma: float
    option_type: Literal["call", "put"] = "call"
    binomial_steps: int = 300
    mc_paths: int = 25000
    mc_steps: int = 252
    exercise_dates: int = 12


# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Options API is running",
    }


@app.get("/terminal/{ticker}")
def terminal_snapshot(
    ticker: str,
    side: Literal["calls", "puts"] = Query("calls"),
    period: str = Query("6mo"),
):
    ticker = ticker.upper()

    data = cached_market_data(ticker, side)
    hist = cached_history(ticker, period)

    surface_df = add_contract_fields(data["surface_df"])
    selected = choose_default_contract(surface_df)

    history_df = normalise_history(hist)
    stats = latest_price_stats(hist)
    quality = surface_quality(surface_df)

    option_type = "call" if side == "calls" else "put"

    surface_cols = [
        "contract_id",
        "contractSymbol",
        "ticker",
        "expiry",
        "option_type",
        "strike",
        "T",
        "moneyness",
        "bid",
        "ask",
        "mid",
        "market_price",
        "relative_spread",
        "sigma",
        "delta",
        "gamma",
        "vega",
        "vega_1pct",
        "theta",
        "theta_day",
        "rho",
        "rho_1pct",
        "volume",
        "openInterest",
    ]

    surface_cols = [c for c in surface_cols if c in surface_df.columns]

    return {
        "ticker": ticker,
        "side": side,
        "option_type": option_type,
        "spot": clean_value(data.get("spot")),
        "r": clean_value(data.get("r")),
        "q": clean_value(data.get("q")),
        "price_stats": stats,
        "surface_quality": quality,
        "selected_contract": selected,
        "history": clean_records(history_df),
        "chain": clean_records(surface_df[surface_cols]),
        "surface_grid": make_surface_grid(surface_df),
    }


@app.post("/price")
def price_option(req: PriceRequest):
    d1, d2, european_price = bs_price(
        req.S0,
        req.K,
        req.r,
        req.q,
        req.T,
        req.sigma,
        option_type=req.option_type,
    )

    greeks = bs_greeks(
        req.S0,
        req.K,
        req.r,
        req.q,
        req.T,
        req.sigma,
        option_type=req.option_type,
    )

    price = float(european_price)
    standard_error = None
    ci_low = None
    ci_high = None
    method = "Black-Scholes European closed form"

    if req.model == "American":
        if binomial_american_option is None:
            raise RuntimeError("American pricing module could not be imported.")

        out = binomial_american_option(
            S0=req.S0,
            K=req.K,
            sigma=req.sigma,
            q=req.q,
            r=req.r,
            T=req.T,
            N=req.binomial_steps,
            option_type=req.option_type,
        )

        price = float(out["price"])
        method = f"Binomial American tree, N={req.binomial_steps}"

    elif req.model == "Asian":
        if price_arithmetic_asian is None:
            raise RuntimeError("Asian pricing module could not be imported.")

        price, standard_error = price_arithmetic_asian(
            S0=req.S0,
            K=req.K,
            r=req.r,
            q=req.q,
            T=req.T,
            sigma=req.sigma,
            n_steps=req.mc_steps,
            n_paths=req.mc_paths,
            option_type=req.option_type,
            seed=42,
        )

        ci_low = price - 1.96 * standard_error
        ci_high = price + 1.96 * standard_error
        method = f"Arithmetic Asian Monte Carlo, {req.mc_paths:,} paths"

    elif req.model == "Bermudan":
        if longstaff_schwartz_bermudan is None:
            raise RuntimeError("Bermudan pricing module could not be imported.")

        out = longstaff_schwartz_bermudan(
            S0=req.S0,
            K=req.K,
            r=req.r,
            q=req.q,
            T=req.T,
            sigma=req.sigma,
            n_steps=req.mc_steps,
            n_paths=req.mc_paths,
            n_exercise_dates=req.exercise_dates,
            option_type=req.option_type,
            seed=42,
        )

        price = float(out["price"])
        standard_error = float(out["standard_error"])
        ci_low = float(out["ci_low"])
        ci_high = float(out["ci_high"])
        method = f"Longstaff-Schwartz Bermudan MC, {req.mc_paths:,} paths"

    greeks_clean = {
        "delta": clean_value(greeks.get("delta")),
        "gamma": clean_value(greeks.get("gamma")),
        "vega_raw": clean_value(greeks.get("vega")),
        "vega_1pct": clean_value(greeks.get("vega") / 100),
        "theta_raw": clean_value(greeks.get("theta")),
        "theta_day": clean_value(greeks.get("theta") / 365),
        "rho_raw": clean_value(greeks.get("rho")),
        "rho_1pct": clean_value(greeks.get("rho") / 100),
        "d1": clean_value(greeks.get("d1")),
        "d2": clean_value(greeks.get("d2")),
    }

    return {
        "model": req.model,
        "method": method,
        "price": clean_value(price),
        "european_price": clean_value(european_price),
        "model_premium": clean_value(price - european_price),
        "standard_error": clean_value(standard_error),
        "ci_low": clean_value(ci_low),
        "ci_high": clean_value(ci_high),
        "greeks": greeks_clean,
        "inputs": req.model_dump(),
    }