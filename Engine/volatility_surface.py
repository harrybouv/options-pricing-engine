import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
try:
    from Engine.market_data import market_data
except ModuleNotFoundError:
    from market_data import market_data


def extract_atm_points(surface_df: pd.DataFrame) -> pd.DataFrame:
    atm_rows = []

    for expiry, group in surface_df.groupby("expiry"):
        group = group.copy()
        group["atm_distance"] = np.abs(group["moneyness"] - 1.0)
        idx = group["atm_distance"].idxmin()
        atm_rows.append(group.loc[idx])

    atm_df = pd.DataFrame(atm_rows).sort_values("T").reset_index(drop=True)

    return atm_df


def plot_atm_term_structure(atm_df: pd.DataFrame, ticker: str, option_type: str):
    plt.figure(figsize=(10, 6))
    plt.plot(atm_df["T"], atm_df["sigma"], marker="o", linestyle="-", label="ATM IV")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Implied Volatility")
    plt.title(f"{ticker} {option_type.upper()} ATM Implied Volatility Term Structure")
    plt.legend()
    plt.grid(True)
    plt.show()


def interpolate_smile_to_grid(group: pd.DataFrame, m_grid: np.ndarray, value_col: str = "sigma") -> pd.DataFrame:
    group = group.sort_values("moneyness").copy()
    group = group.dropna(subset=["moneyness", value_col])
    group = group.groupby("moneyness", as_index=False)[value_col].mean()

    x = group["moneyness"].to_numpy(dtype=float)
    y = group[value_col].to_numpy(dtype=float)

    if len(x) < 2:
        return pd.DataFrame()

    mask = (m_grid >= x.min()) & (m_grid <= x.max())

    if mask.sum() < 2:
        return pd.DataFrame()

    y_interp = np.interp(m_grid[mask], x, y)

    out = pd.DataFrame({
        "moneyness": m_grid[mask],
        value_col: y_interp,
    })

    return out


def interpolate_smile_to_grid_pchip(group: pd.DataFrame, m_grid: np.ndarray, value_col: str = "sigma") -> pd.DataFrame:
    group = group.sort_values("moneyness").copy()
    group = group.dropna(subset=["moneyness", value_col])
    group = group.groupby("moneyness", as_index=False)[value_col].mean()

    x = group["moneyness"].to_numpy(dtype=float)
    y = group[value_col].to_numpy(dtype=float)

    if len(x) < 2:
        return pd.DataFrame()

    mask = (m_grid >= x.min()) & (m_grid <= x.max())

    if mask.sum() < 2:
        return pd.DataFrame()

    pchip = PchipInterpolator(x, y)
    y_interp = pchip(m_grid[mask])

    out = pd.DataFrame({
        "moneyness": m_grid[mask],
        value_col: y_interp
    })

    return out

def build_surface_grid(
    surface_df: pd.DataFrame,
    value_col: str = "sigma",
    m_min: float = 0.90,
    m_max: float = 1.10,
    n_moneyness: int = 25,
) -> pd.DataFrame:
    m_grid = np.linspace(m_min, m_max, n_moneyness)
    rows = []

    for expiry, group in surface_df.groupby("expiry"):
        interp_df = interpolate_smile_to_grid(group, m_grid, value_col=value_col)

        if interp_df.empty:
            continue

        T_val = float(group["T"].iloc[0])
        interp_df["expiry"] = expiry
        interp_df["T"] = T_val
        rows.append(interp_df)

    if not rows:
        raise ValueError("No valid interpolated surface could be built")

    grid_df = pd.concat(rows, ignore_index=True)
    grid_df = grid_df.sort_values(["T", "moneyness"]).reset_index(drop=True)

    return grid_df

def build_surface_grid_pchip(
    surface_df: pd.DataFrame,
    value_col: str = "sigma",
    m_min: float = 0.90,
    m_max: float = 1.10,
    n_moneyness: int = 25,
) -> pd.DataFrame:
    m_grid = np.linspace(m_min, m_max, n_moneyness)
    rows = []

    for expiry, group in surface_df.groupby("expiry"):
        interp_df = interpolate_smile_to_grid_pchip(group, m_grid, value_col=value_col)

        if interp_df.empty:
            continue

        T_val = float(group["T"].iloc[0])
        interp_df["expiry"] = expiry
        interp_df["T"] = T_val
        rows.append(interp_df)

    if not rows:
        raise ValueError("No valid interpolated surface could be built")

    grid_df = pd.concat(rows, ignore_index=True)
    grid_df = grid_df.sort_values(["T", "moneyness"]).reset_index(drop=True)

    return grid_df



def get_surface_vol(surface_df: pd.DataFrame, S0: float, K: float, T: float, value_col: str = "sigma") -> float:
    grid_df = build_surface_grid(surface_df, value_col=value_col, m_min=0.90, m_max=1.10, n_moneyness=75)

    target_moneyness = K / S0

    grid_df["distance"] = (
        (grid_df["T"] - T).abs()
        + (grid_df["moneyness"] - target_moneyness).abs()
    )

    row = grid_df.loc[grid_df["distance"].idxmin()]

    return float(row[value_col])


def plot_iv_heatmap(grid_df: pd.DataFrame, ticker: str, option_type: str, value_col: str = "sigma"):
    pivot = grid_df.pivot(index="moneyness", columns="T", values=value_col).sort_index()

    T_vals = pivot.columns.to_numpy(dtype=float)
    m_vals = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(T_vals, m_vals, Z, shading="auto")
    plt.colorbar(mesh, label="Implied Volatility")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Moneyness (K / S0)")
    plt.title(f"{ticker} {option_type.upper()} IV Heatmap")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_iv_contour(grid_df: pd.DataFrame, ticker: str, option_type: str, value_col: str = "sigma"):
    pivot = grid_df.pivot(index="moneyness", columns="T", values=value_col).sort_index()

    T_vals = pivot.columns.to_numpy(dtype=float)
    m_vals = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    T_grid, M_grid = np.meshgrid(T_vals, m_vals)

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(T_grid, M_grid, Z, levels=20)
    plt.colorbar(contour, label="Implied Volatility")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Moneyness (K / S0)")
    plt.title(f"{ticker} {option_type.upper()} IV Contour")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_iv_surface_3d(
    grid_df: pd.DataFrame,
    ticker: str,
    option_type: str,
    value_col: str = "sigma",
    elev: float = 25,
    azim: float = 35,
    cmap: str = "viridis",
):
    pivot = grid_df.pivot(index="moneyness", columns="T", values=value_col).sort_index()

    T_vals = pivot.columns.to_numpy(dtype=float)
    m_vals = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    T_grid, M_grid = np.meshgrid(T_vals, m_vals)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        T_grid,
        M_grid,
        Z,
        cmap=cmap,
        edgecolor="none",
        antialiased=True,
    )

    ax.set_xlabel("Time to Expiry")
    ax.set_ylabel("Moneyness (K / S0)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{ticker} {option_type.upper()} IV Surface")

    ax.view_init(elev=elev, azim=azim)

    zmin = np.nanmin(Z)
    zmax = np.nanmax(Z)
    pad = 0.05 * (zmax - zmin) if zmax > zmin else 0.01
    ax.set_zlim(zmin - pad, zmax + pad)

    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="Implied Volatility")
    plt.show()


def plot_iv_graphs(ticker="AAPL"):
    for option_type in ["calls", "puts"]:
        print(f"\n===== {option_type.upper()} =====")
        data = market_data(ticker, option_type=option_type)
        surface_df = data["surface_df"].copy()

        atm_df = extract_atm_points(surface_df)
        print("\nATM points:")
        print(atm_df[["expiry", "T", "strike", "moneyness", "sigma"]])

        raw_grid_df = build_surface_grid(
            surface_df,
            value_col="sigma",
            m_min=0.90,
            m_max=1.10,
            n_moneyness=75,
        )

        pchip_grid_df = build_surface_grid_pchip(
            surface_df,
            value_col="sigma",
            m_min=0.90,
            m_max=1.10,
            n_moneyness=75,
        )

        print("\nRaw interpolated surface sample:")
        print(raw_grid_df.head())

        print("\nPCHIP interpolated surface sample:")
        print(pchip_grid_df.head())

        plot_atm_term_structure(atm_df, ticker, option_type)

        plot_iv_heatmap(raw_grid_df, ticker, f"{option_type} RAW")
        plot_iv_contour(raw_grid_df, ticker, f"{option_type} RAW")
        plot_iv_surface_3d(raw_grid_df, ticker, f"{option_type} RAW")

        plot_iv_heatmap(pchip_grid_df, ticker, f"{option_type} PCHIP")
        plot_iv_contour(pchip_grid_df, ticker, f"{option_type} PCHIP")
        plot_iv_surface_3d(pchip_grid_df, ticker, f"{option_type} PCHIP")

if __name__ == "__main__":
    plot_iv_graphs("AAPL")