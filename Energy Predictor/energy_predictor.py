"""
Energy Predictor
------------------------------------------------------------------
• Loads hourly data-centre kWh (or PUE) and local grid-carbon-intensity
• Engineers features: IT load %, delta-temp, hour, dow, month, rolling 24 h mean
• Trains Linear Regression baseline, prints metrics, plots results
• Performs a simple what-if: “What if we raise the cooling set-point by ±2 °C?”
"""
import argparse, sys, pathlib, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path(__file__).parent / "data"

def load_energy(site: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "data_center_energy_footprints.csv",
                     parse_dates=["timestamp"])
    df = df.query("site == @site").set_index("timestamp").sort_index()
    return df

def load_carbon(country_code: str) -> pd.Series:
    co2 = pd.read_csv(DATA_DIR / "co2_intensity_eu_2018_2024.csv",
                      parse_dates=["datetime"])
    co2 = co2.query("country == @country_code")\
             .set_index("datetime")["carbon_intensity_gco2_kwh"]\
             .resample("H").interpolate("time")
    return co2

def engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # example engineered features
    df["delta_temp"] = df["outside_temp_c"] - df["cooling_setpoint_c"]
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek
    df["month"] = df.index.month
    df["it_load_pct"] = df["it_load_kw"] / df["it_capacity_kw"]
    df["kwh_rolling24h"] = df["energy_kwh"].rolling(24, min_periods=1).mean()
    df = df.dropna()
    return df

def train_and_eval(df: pd.DataFrame, target="energy_kwh"):
    X = df[["it_load_pct", "delta_temp", "hour", "dow",
            "month", "kwh_rolling24h"]]
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    return model, (mae, r2), (X_te.index, y_te, y_pred)

def what_if_setpoint(model, df, delta_c=(-2, +2)):
    base = df.copy()
    scenarios = {}
    for d in delta_c:
        tmp = base.copy()
        tmp["delta_temp"] = tmp["delta_temp"] - d  # raising set-point ↓ delta
        scenarios[f"setpoint+{d:+d}C"] = model.predict(
            tmp[["it_load_pct", "delta_temp", "hour",
                 "dow", "month", "kwh_rolling24h"]]
        )
    return scenarios

def plot_results(idx, y_true, y_pred, what_if_dict):
    plt.figure(figsize=(10,4))
    plt.plot(idx, y_true, label="Actual", linewidth=1)
    plt.plot(idx, y_pred, label="Predicted", linewidth=1)
    plt.title("Hourly Energy Consumption: Actual vs Predicted")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    for lbl, vals in what_if_dict.items():
        plt.hist(vals - y_pred, bins=40, alpha=0.6, label=lbl)
    plt.title("Energy Δ from changing cooling set-point")
    plt.xlabel("Δ kWh relative to baseline prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()

def cli(argv):
    p = argparse.ArgumentParser(
        description="Train a baseline energy predictor for a data-centre site.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python energy_predictor.py --site SITE_A --years 2022 2023 --plot
        """)
    )
    p.add_argument("--site", required=True, help="Site ID in the dataset")
    p.add_argument("--country", default="DE", help="ISO-2 code for carbon data")
    p.add_argument("--years", nargs="+", type=int, default=[],
                   help="Subset specific years (e.g. 2022 2023)")
    p.add_argument("--plot", action="store_true", help="Show result plots")
    args = p.parse_args(argv)

    # load and merge
    energy = load_energy(args.site)
    if args.years:
        energy = energy[energy.index.year.isin(args.years)]
    co2 = load_carbon(args.country)
    df = energy.join(co2.rename("grid_gco2_kwh"), how="inner")
    df = engineer(df)

    # training
    model, (mae, r2), (idx, y_true, y_pred) = train_and_eval(df)
    print(f"Hold-out MAE  = {mae:,.2f} kWh")
    print(f"Hold-out R²   = {r2: .3f}")

    # other plots
    if args.plot:
        what_if = what_if_setpoint(model, df)
        plot_results(idx, y_true, y_pred, what_if)

if __name__ == "__main__":
    cli(sys.argv[1:])
