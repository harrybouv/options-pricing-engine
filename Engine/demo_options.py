from market_data import market_data, plot_multiple_smiles
from volatility_surface import plot_iv_graphs
from GBM_plot import plot_GBM_paths
from GBM_terminal import plot_variates_comparison

def main():

    plot_GBM_paths()
    plot_variates_comparison()

    ticker = "AAPL"
    for option_type in ["calls", "puts"]:
        print(f"\n===== {option_type.upper()} =====")
        data = market_data(ticker, option_type=option_type)

        for expiry, clean in data["smile_data"].items():
            print(f"\nExpiry: {expiry}")
            print(clean[["contractSymbol", "strike", "option_type", "sigma", "delta", "gamma", "vega", "theta",
                         "rho"]].head(5))

        plot_multiple_smiles(data["smile_data"], f"{data['ticker']} {option_type.upper()}", data["spot"])

    plot_iv_graphs()

if __name__ ==  "__main__":
    main()