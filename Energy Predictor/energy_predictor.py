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

"""
Read dc_hourly_sample.csv, coerce whatever timestamp column exists to timestamp, filter the requested site, index by time.
"""


def load_energy(site: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "dc_hourly_sample.csv")
    # Try to guess the timestamp column ➔ standardise to "timestamp"
    for cand in ("timestamp", "datetime", "date_time", "DateTime", "Date"):
        if cand in df.columns:
            df.rename(columns={cand: "timestamp"}, inplace=True)
            break
    if "timestamp" not in df.columns:
        raise ValueError("No timestamp-like column found in dataset header.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df.query("site == @site").set_index("timestamp").sort_index()


"""
Ensure an hourly CO₂ file exists (calling make_hourly_from_annual if needed) then return a pd.Series of gCO₂ kWh⁻¹, indexed hourly, for the requested country.
"""


def load_carbon(country_code: str, ts_col="datetime"):
    hourly_path = DATA_DIR / "co2_intensity_hourly.csv"
    if not hourly_path.exists():  # first run ➜ create it
        make_hourly_from_annual(
            src_path=DATA_DIR / "annual_carbon_intensity.csv",
            dst_path=hourly_path
        )

    co2 = (pd.read_csv(hourly_path, parse_dates=[ts_col])
           .query("country == @country_code")
           .set_index(ts_col)["carbon_intensity_gco2_kwh"]
           .sort_index())
    return co2


"""
Add domain features: Δ temperature, hour-of-day, day-of-week, month, IT-load %, rolling-24-h mean of kWh. Drops rows with any fresh NaNs.
"""


def engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # example engineered features
    df["delta_temp"] = df["outside_temp_c"] - df["cooling_setpoint_c"]
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["it_load_pct"] = df["it_load_kw"] / df["it_capacity_kw"]
    df["kwh_rolling24h"] = df["energy_kwh"].rolling(24, min_periods=1).mean()
    df = df.dropna()
    return df


"""
Split 80 / 20, fit LinearRegression, return model, metrics (MAE, R²) and the (index, y_true, y_pred) triplet for the test partition.
"""


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
    r2 = r2_score(y_te, y_pred)
    return model, (mae, r2), (X_te.index, y_te, y_pred)


"""
For each ± 2 °C scenario, mutate delta_temp, re-run the model, collect predicted kWh.
"""


def what_if_setpoint(model, df_subset, delta_c=(-2, +2)):
    """Apply what-if only to the supplied subset (test rows)"""
    scenarios = {}
    for d in delta_c:
        tmp = df_subset.copy()
        tmp["delta_temp"] = tmp["delta_temp"] - d
        scenarios[f"setpoint{d:+d}C"] = model.predict(
            tmp[["it_load_pct", "delta_temp", "hour",
                 "dow", "month", "kwh_rolling24h"]]
        )
    return scenarios


"""
One-off helper: explode an annual carbon-intensity table into an hourly CSV by repeating the annual value for every hour of that year.
"""


def make_hourly_from_annual(src_path: pathlib.Path,
                            dst_path: pathlib.Path,
                            country_col="CountryShort",
                            value_col="ValueNumeric"):
    """Convert annual table to an hourly time-series (constant within year)."""
    annual = pd.read_csv(src_path)
    rows = []
    for _, row in annual.iterrows():
        idx = pd.date_range(f'{int(row["Year"])}-01-01',
                            f'{int(row["Year"])}-12-31 23:00',
                            freq="h", tz="UTC")
        rows.append(pd.DataFrame({
            "datetime": idx,
            "country": row[country_col],
            "carbon_intensity_gco2_kwh": row[value_col]
        }))
    pd.concat(rows).to_csv(dst_path, index=False)


"""
Three visuals:
(1) time-sorted line plot of test rows, 
(2) outline histogram of Δ kWh for each scenario (x-axis zoomed to ±0.2 kWh), 
(3) residual plot added later in cli().
"""


def plot_results(idx, y_true, y_pred, what_if_dict):
    # sort by time for a clean line plot
    order = np.argsort(idx)
    idx_sorted = idx[order]
    y_true_sorted = y_true.to_numpy()[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(10, 4))
    plt.plot(idx_sorted, y_true_sorted, label="Actual", linewidth=1)
    plt.plot(idx_sorted, y_pred_sorted, label="Predicted", linewidth=1)
    plt.title("Hourly Energy Consumption: Actual vs Predicted")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # histogram of deltas (same length as y_pred now)
    # ── Better visual: zoomed-in outline histogram ─────────────────────────
    plt.figure(figsize=(6, 3))
    for lbl, vals in what_if_dict.items():
        plt.hist(vals - y_pred,
                 bins=40,
                 histtype="step",  # draw outlines only
                 linewidth=1.4,
                 label=lbl)

    plt.xlim(-0.2, 0.2)  # focus on tiny energy shifts
    plt.axvline(0, color="k", linewidth=0.5)
    plt.title("Δ Energy from set-point ±2 °C (test set)")
    plt.xlabel("kWh difference from baseline")
    plt.legend(frameon=False, fontsize="small")
    plt.tight_layout()
    plt.show()
    # ───────────────────────────────────────────────────────────────────────


"""
Parses --site, --country, --years, --plot; orchestrates the whole flow: 
loads data, joins carbon, engineers features, trains, prints metrics, prints what-if numeric summary, and spawns all plots.
"""


def cli(argv):
    p = argparse.ArgumentParser(
        description="Train a baseline energy predictor for a data-centre site.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Run with something like:
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

    co2 = load_carbon(args.country)  # NEW
    df = energy.join(co2.rename("grid_gco2_kwh"), how="inner")

    # training
    df_eng = engineer(energy)
    model, (mae, r2), (idx_test, y_test, y_pred) = train_and_eval(df_eng)

    print(f"Hold-out MAE  = {mae:,.2f} kWh")
    print(f"Hold-out R²   = {r2: .3f}")

    # full timeline plot
    if args.plot:
        # sort the whole data set chronologically
        df_all = df_eng.sort_index()

        # predict the whole timeline
        X_all = df_all[["it_load_pct", "delta_temp", "hour", "dow",
                        "month", "kwh_rolling24h"]]
        y_all = df_all["energy_kwh"]
        y_hat = model.predict(X_all)

        df_daily = df_all.resample("D").mean(numeric_only=True)  # only numbers
        y_hat_day = (pd.Series(y_hat, index=df_all.index)
                     .resample("D")
                     .mean())

        plt.figure(figsize=(10, 4))
        plt.plot(df_daily.index, df_daily["energy_kwh"],
                 label="Actual (daily mean)")
        plt.plot(y_hat_day.index, y_hat_day,
                 label="Predicted (daily mean)")
        plt.title("Daily Energy Consumption – Actual vs Predicted")
        plt.ylabel("kWh")
        plt.legend(frameon=False, fontsize="small")
        plt.tight_layout()
        plt.show()

        # what-if histogram
        df_test = df_eng.loc[idx_test]
        what_if = what_if_setpoint(model, df_test)

        delta_neg = (what_if["setpoint-2C"] - y_pred).mean()
        delta_pos = (what_if["setpoint+2C"] - y_pred).mean()
        print(f"↘ Lowering the cooling set-point by 2 °C saves "
              f"{abs(delta_neg):.2f} kWh per hour "
              f"(≈{abs(delta_neg) * 100 / y_pred.mean():.2f} % of load).")
        print(f"↗ Raising it by 2 °C adds "
              f"{delta_pos:.2f} kWh per hour "
              f"(≈{delta_pos * 100 / y_pred.mean():.2f} %).")

        plot_results(idx_test, y_test, y_pred, what_if)

        # residual plot
        order = np.argsort(idx_test)
        idx_sorted = idx_test[order]
        residuals = (y_test.to_numpy() - y_pred)[order]  # Actual − Predicted

        plt.figure(figsize=(10, 3))
        plt.plot(idx_sorted, residuals, linewidth=1)
        plt.axhline(0, color="k", linewidth=0.5)
        plt.title("Prediction Residuals – Test Set")
        plt.ylabel("kWh error")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cli(sys.argv[1:])
