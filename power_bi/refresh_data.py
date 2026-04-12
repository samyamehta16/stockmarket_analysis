# =============================================================================
# powerbi/refresh_data.py
# Refresh all CSVs that Power BI reads from — run this to update dashboards
# Schedule this with Windows Task Scheduler for automatic daily refresh
# =============================================================================
# pip install pandas numpy prophet scikit-learn pandas-datareader python-dotenv
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", None)

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Starting Power BI data refresh...")

# =============================================================================
# 1. FETCH LIVE DATA
# =============================================================================
print("Fetching live S&P 500 from FRED...")
raw = pdr.DataReader("SP500", "fred",
                     start="1990-01-01",
                     end=datetime.today().strftime("%Y-%m-%d"),
                     api_key=FRED_API_KEY)
raw.columns = ["price"]
raw.index.name = "date"
raw = raw.reset_index()
raw["date"]  = pd.to_datetime(raw["date"])
raw["price"] = pd.to_numeric(raw["price"], errors="coerce").fillna(method="ffill")
raw = raw.dropna().sort_values("date").reset_index(drop=True)
print(f"  Fetched {len(raw)} rows up to {raw['date'].max().date()}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
df = raw.copy()
df["daily_return"]  = df["price"].pct_change() * 100
df["log_return"]    = np.log(df["price"] / df["price"].shift(1)) * 100
df["ma_7"]          = df["price"].rolling(7,   min_periods=1).mean()
df["ma_30"]         = df["price"].rolling(30,  min_periods=1).mean()
df["ma_90"]         = df["price"].rolling(90,  min_periods=1).mean()
df["ma_200"]        = df["price"].rolling(200, min_periods=1).mean()
df["volatility_7"]  = df["daily_return"].rolling(7).std()
df["volatility_30"] = df["daily_return"].rolling(30).std()
df["volatility_90"] = df["daily_return"].rolling(90).std()
df["bb_mid"]        = df["price"].rolling(20).mean()
df["bb_std"]        = df["price"].rolling(20).std()
df["bb_upper"]      = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"]      = df["bb_mid"] - 2 * df["bb_std"]
df["bb_width"]      = df["bb_upper"] - df["bb_lower"]
df["golden_cross"]  = (df["ma_30"] > df["ma_200"]).astype(int)
df["year"]          = df["date"].dt.year
df["month"]         = df["date"].dt.month
df["month_name"]    = df["date"].dt.strftime("%b")
df["quarter"]       = df["date"].dt.quarter
df["direction"]     = np.where(df["daily_return"] >= 0, "Up", "Down")
df["cumulative_return"] = (df["price"] / df["price"].iloc[0] - 1) * 100

# Market mood based on MA crossover
latest = df.iloc[-1]
if latest["ma_30"] > latest["ma_200"] * 1.02:
    df["market_mood"] = "Bullish"
elif latest["ma_30"] < latest["ma_200"] * 0.98:
    df["market_mood"] = "Bearish"
else:
    df["market_mood"] = "Neutral"

# =============================================================================
# 3. HYBRID MODEL
# =============================================================================
print("Running Hybrid Model...")

FEATURE_COLS = [
    "ma_7","ma_30","ma_90","ma_200",
    "volatility_7","volatility_30","volatility_90",
    "bb_upper","bb_lower","bb_mid","bb_width",
    "daily_return","log_return","golden_cross",
]

df_model = df.dropna(subset=FEATURE_COLS + ["price"]).reset_index(drop=True)
n       = len(df_model)
n_test  = max(60, int(n * 0.20))
n_train = n - n_test

train_df = df_model.iloc[:n_train].copy()

# Prophet
prophet_train = train_df[["date","price"]].rename(columns={"date":"ds","price":"y"})
prophet_full  = df_model[["date","price"]].rename(columns={"date":"ds","price":"y"})

model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05, interval_width=0.95)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
model.fit(prophet_train)

pred_full = model.predict(prophet_full[["ds"]])
df_model["prophet_pred"]  = pred_full["yhat"].values
df_model["prophet_lower"] = pred_full["yhat_lower"].values
df_model["prophet_upper"] = pred_full["yhat_upper"].values
df_model["residual"]      = df_model["price"] - df_model["prophet_pred"]

train_df = df_model.iloc[:n_train].copy()
test_df  = df_model.iloc[n_train:].copy()

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=6,
                           min_samples_leaf=10, max_features="sqrt",
                           random_state=42, n_jobs=-1)
rf.fit(train_df[FEATURE_COLS].values, train_df["residual"].values)

df_model["rf_correction"] = rf.predict(df_model[FEATURE_COLS].values)
df_model["hybrid_pred"]   = df_model["prophet_pred"] + df_model["rf_correction"]
df_model["hybrid_error"]  = df_model["price"] - df_model["hybrid_pred"]

# Metrics
test_df   = df_model.iloc[n_train:].copy()
actual    = test_df["price"].values
predicted = test_df["hybrid_pred"].values
rmse = float(np.sqrt(np.mean((actual - predicted)**2)))
mae  = float(np.mean(np.abs(actual - predicted)))
mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
r2   = float(1 - np.sum((actual-predicted)**2) / np.sum((actual-actual.mean())**2))

# Future forecast (30 business days)
last_date    = df_model["date"].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="B")
future_input = pd.DataFrame({"ds": future_dates})
future_pred  = model.predict(future_input)
last_feat    = df_model[FEATURE_COLS].iloc[-1].values.reshape(1,-1)
future_rf    = rf.predict(np.repeat(last_feat, 30, axis=0))

future_df = pd.DataFrame({
    "date"    : future_dates,
    "forecast": future_pred["yhat"].values + future_rf,
    "lower_95": future_pred["yhat_lower"].values,
    "upper_95": future_pred["yhat_upper"].values,
    "type"    : "Forecast"
})

# Feature importance
feat_imp = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

# Monthly seasonality
month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_avg = (df.dropna(subset=["daily_return"])
               .groupby("month_name")["daily_return"].mean()
               .reindex(month_order).reset_index())
monthly_avg.columns = ["month", "avg_return"]

# Annual returns
annual_returns = (df.dropna(subset=["daily_return"])
                  .groupby("year")["daily_return"].sum()
                  .reset_index())
annual_returns.columns = ["year", "annual_return"]
annual_returns["positive"] = annual_returns["annual_return"] > 0

# Metrics summary
metrics_df = pd.DataFrame([{
    "model"      : "Hybrid (Prophet + RF)",
    "RMSE"       : round(rmse, 2),
    "MAE"        : round(mae, 2),
    "MAPE_pct"   : round(mape, 2),
    "R2"         : round(r2, 4),
    "train_rows" : n_train,
    "test_rows"  : n_test,
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "data_end_date": str(last_date.date()),
}])

# KPI summary (single row — Power BI card visuals read this)
kpis = pd.DataFrame([{
    "current_price"  : round(float(df["price"].iloc[-1]), 2),
    "price_change"   : round(float(df["price"].iloc[-1] - df["price"].iloc[-2]), 2),
    "pct_change"     : round(float((df["price"].iloc[-1]/df["price"].iloc[-2]-1)*100), 4),
    "all_time_high"  : round(float(df["price"].max()), 2),
    "ytd_return_pct" : round(float(df[df["year"]==datetime.today().year]["daily_return"].sum()), 2),
    "volatility_30d" : round(float(df["volatility_30"].iloc[-1]), 4),
    "market_mood"    : df["market_mood"].iloc[-1],
    "ma_30"          : round(float(df["ma_30"].iloc[-1]), 2),
    "ma_200"         : round(float(df["ma_200"].iloc[-1]), 2),
    "last_updated"   : datetime.now().strftime("%Y-%m-%d %H:%M"),
}])

# =============================================================================
# 4. SAVE ALL CSVs FOR POWER BI
# =============================================================================
files = {
    "sp500_prices.csv"          : df[["date","price","ma_7","ma_30","ma_90","ma_200",
                                       "daily_return","volatility_30","bb_upper",
                                       "bb_lower","bb_mid","direction","cumulative_return",
                                       "year","month","month_name","quarter"]],
    "forecast_results.csv"      : df_model[["date","price","prophet_pred","hybrid_pred",
                                             "prophet_lower","prophet_upper","hybrid_error"]],
    "future_forecast.csv"       : future_df,
    "feature_importance.csv"    : feat_imp,
    "monthly_seasonality.csv"   : monthly_avg,
    "annual_returns.csv"        : annual_returns,
    "model_metrics.csv"         : metrics_df,
    "kpis.csv"                  : kpis,
}

for filename, data in files.items():
    path = os.path.join(OUTPUT_DIR, filename)
    data.to_csv(path, index=False)
    print(f"  Saved: powerbi/data/{filename}  ({len(data)} rows)")

print(f"\n Power BI data refresh complete!")
print(f"   RMSE: {rmse:,.2f}  |  MAPE: {mape:.2f}%  |  R²: {r2:.4f}")
print(f"   Data up to: {last_date.date()}")
print(f"   Output folder: {OUTPUT_DIR}")