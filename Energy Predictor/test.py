import pandas as pd
from rich import print
from rich.console import Console
import numpy as np
import matplotlib.pyplot as plt


def clean_data(df,row_needed, row_names) -> pd.DataFrame:
    df_cleaned = df.dropna(subset=row_needed)

    print(df.head(10)) if __name__ == "__main__" else None

    df_cleaned = df_cleaned[row_needed]

    for i in range(len(row_needed)):
        if row_needed[i]!=row_names[i]:
            df_cleaned = df_cleaned.rename(columns={row_needed[i]: row_names[i]})

    df_cleaned.reset_index(drop=True, inplace=True)

    print(df_cleaned) if __name__ == "__main__" else None
    return df_cleaned


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[red]Error: The specified file was not found.")
        exit()
    return df


if __name__ == "__main__":
    df = clean_data(load_data("Energy Predictor/data/data_center_energy_footprints.csv"), ["Region", "Country", "Unnamed: 5"], ["Region", "Country","Power data"])

    print(df.head(10))

    df['Power data'] = pd.to_numeric(df['Power data'], errors='coerce')

    region_power_sum = df.groupby('Region')['Power data'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.scatter(region_power_sum['Region'], region_power_sum['Power data'], cmap='viridis', alpha=0.7)

    for i, row in region_power_sum.iterrows():
        y_vals = np.linspace(0, row['Power data'], 10)[1:-1]
        x_vals = [row['Region']] * len(y_vals)
        plt.scatter(x_vals, y_vals, cmap='viridis', s=10)

    plt.xlabel('Region')
    plt.ylabel('Power Data')
    plt.title('Total Power Data by Region')

    plt.tight_layout()
    plt.show()