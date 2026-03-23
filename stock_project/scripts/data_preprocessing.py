# =============================================================================
# 02_preprocessing.py
# Stock Market Analysis & Forecasting
# Step 2: Data Preprocessing & Feature Engineering
# =============================================================================
# pip install pandas numpy
# =============================================================================

import os
import numpy as np
import pandas as pd

print("Loading raw data...")
df = pd.read_csv("data/sp500_raw.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Rows loaded  : {len(df)}")
print(f"Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Missing (raw): {df['price'].isna().sum()}")


print("\n--- Step 1: Handling Missing Values ---")
df["price"] = df["price"].fillna(method="ffill")
df = df.dropna(subset=["price"]).reset_index(drop=True)
print(f"Missing after fill: {df['price'].isna().sum()}")
print(f"Rows after cleaning: {len(df)}")


print("\n--- Step 2: Time Features ---")
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")
df["quarter"] = df["date"].dt.quarter
df["week"]    = df["date"].dt.isocalendar().week.astype(int)
df["weekday"] = df["date"].dt.day_name()
df["decade"]  = (df["year"] // 10) * 10


print("--- Step 3: Daily Returns ---")
df["daily_return"]     = df["price"].pct_change() * 100          # % return
df["log_return"]       = np.log(df["price"] / df["price"].shift(1)) * 100
df["price_change"]     = df["price"].diff()
df["direction"]        = np.where(df["daily_return"] >= 0, "Up", "Down")

print("--- Step 4: Moving Averages ---")
df["ma_7"]   = df["price"].rolling(window=7,   min_periods=1).mean()
df["ma_30"]  = df["price"].rolling(window=30,  min_periods=1).mean()
df["ma_90"]  = df["price"].rolling(window=90,  min_periods=1).mean()
df["ma_200"] = df["price"].rolling(window=200, min_periods=1).mean()


df["golden_cross"] = (df["ma_30"] > df["ma_200"]).astype(int)

print("--- Step 5: Volatility ---")
df["volatility_7"]  = df["daily_return"].rolling(window=7).std()
df["volatility_30"] = df["daily_return"].rolling(window=30).std()
df["volatility_90"] = df["daily_return"].rolling(window=90).std()


print("--- Step 6: Bollinger Bands ---")
df["bb_mid"]   = df["price"].rolling(window=20).mean()
df["bb_std"]   = df["price"].rolling(window=20).std()
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
df["bb_width"] = df["bb_upper"] - df["bb_lower"]

df["cumulative_return"] = (df["price"] / df["price"].iloc[0] - 1) * 100


returns = df["daily_return"].dropna()
print("\n--- Summary Statistics ---")
print(f"Mean daily return   : {returns.mean():.4f}%")
print(f"Median daily return : {returns.median():.4f}%")
print(f"Std dev of returns  : {returns.std():.4f}%")
print(f"Best single day     : {returns.max():.4f}%")
print(f"Worst single day    : {returns.min():.4f}%")
print(f"Positive days       : {(returns > 0).sum()}")
print(f"Negative days       : {(returns < 0).sum()}")
print(f"\nFull DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")


df.to_csv("data/sp500_processed.csv", index=False)
print("\nProcessed data saved to: data/sp500_processed.csv")
print("Preprocessing complete. Run 03_eda.py next.")