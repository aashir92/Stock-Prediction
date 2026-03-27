"""
Stock next-day close prediction script (AAPL).

This script downloads five years of daily stock data, engineers time-series
features, trains a Random Forest regressor using a strict chronological split,
and evaluates predictions on the test period.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def print_section_header(title: str) -> None:
    """Print a visible section header in console output."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def download_stock_data(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """Download historical stock data using yfinance with error handling."""
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for ticker '{ticker}'. "
                "Check ticker symbol or network connection."
            )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        raise RuntimeError(
            "Failed to download stock data from yfinance. "
            "Please check your internet connection and try again."
        ) from exc


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer next-day target and time-series features."""
    df = df.copy()
    df["Target_Next_Close"] = df["Close"].shift(-1)
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    return df.dropna().copy()


def split_scale_data(
    df: pd.DataFrame, feature_columns: list[str], target_column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Split data chronologically and scale features using train fit only."""
    X = df[feature_columns]
    y = df[target_column].squeeze()

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy().reshape(-1),
        y_test.to_numpy().reshape(-1),
        X_test.index,
    )


def train_and_predict(
    X_train_scaled: np.ndarray, y_train: np.ndarray, X_test_scaled: np.ndarray
) -> np.ndarray:
    """Train Random Forest model and return test predictions."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


def evaluate_predictions(y_test: np.ndarray, predictions: np.ndarray) -> None:
    """Print regression metrics."""
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print_section_header("STEP 5: EVALUATION METRICS")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


def plot_actual_vs_predicted(
    test_dates: pd.Index, y_test: np.ndarray, predictions: np.ndarray
) -> None:
    """Plot actual vs predicted close prices and save to PNG."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label="Actual", linewidth=2)
    plt.plot(test_dates, predictions, label="Predicted", linewidth=2, alpha=0.9)
    plt.title("AAPL Next-Day Closing Price: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "aapl_actual_vs_predicted.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.show()


def main() -> None:
    """Run full next-day stock close prediction pipeline."""
    print_section_header("STEP 1: DATA ACQUISITION")
    try:
        df_raw = download_stock_data(ticker="AAPL", period="5y")
    except RuntimeError as err:
        print(f"Error: {err}")
        sys.exit(1)

    print(f"Downloaded rows: {len(df_raw)}")
    print(f"Date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")

    print_section_header("STEP 2: FEATURE ENGINEERING & TARGET DEFINITION")
    df_features = engineer_features(df_raw)
    print(f"Rows after feature engineering and NaN drop: {len(df_features)}")

    feature_columns = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Close",
        "SMA_10",
        "SMA_50",
        "Daily_Return",
    ]
    target_column = "Target_Next_Close"

    print_section_header("STEP 3: CHRONOLOGICAL SPLIT & SCALING")
    X_train_scaled, X_test_scaled, y_train, y_test, test_dates = split_scale_data(
        df_features,
        feature_columns,
        target_column,
    )
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")

    print_section_header("STEP 4: RANDOM FOREST MODELING")
    predictions = train_and_predict(X_train_scaled, y_train, X_test_scaled)
    print("Model training and prediction completed.")

    evaluate_predictions(y_test, predictions)

    print_section_header("STEP 5: VISUALIZATION")
    plot_actual_vs_predicted(test_dates, y_test, predictions)
    print("Visualization completed.")


if __name__ == "__main__":
    main()
