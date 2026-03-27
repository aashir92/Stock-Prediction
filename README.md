# Stock-Prediction

## Task Objective
Predict next-day AAPL closing price using historical market data with leakage-safe time-series preprocessing.

## Dataset Used
- Source: Yahoo Finance via `yfinance`
- Ticker: `AAPL`
- Range: last 5 years (daily interval)
- Base features: `Open`, `High`, `Low`, `Close`, `Volume`
- Engineered features: `SMA_10`, `SMA_50`, `Daily_Return`
- Target: `Target_Next_Close = Close.shift(-1)`

## Models Applied
- `RandomForestRegressor` (`n_estimators=100`, `random_state=42`)
- Chronological split (80% train, 20% test)
- `StandardScaler` fit only on training data
- Metrics: MAE and RMSE

## Key Results and Findings
- Pipeline correctly avoids temporal leakage by preserving sequence order.
- The model captures some trend behavior but next-day close remains noisy and hard to predict.
- MAE/RMSE provide a baseline for iterative improvement.
