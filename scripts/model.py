import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

os.makedirs("plots", exist_ok=True)
os.makedirs("data",  exist_ok=True)

NAVY  = "#1A2B4A"
TEAL  = "#0E9F8E"
GOLD  = "#F0B429"
RED   = "#EF4444"
MUTED = "#8A9BB5"
BG    = "#F4F6FA"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : "#FFFFFF",
    "axes.edgecolor"   : "#E2E8F0",
    "axes.labelcolor"  : NAVY,
    "xtick.color"      : MUTED,
    "ytick.color"      : MUTED,
    "grid.color"       : "#E2E8F0",
    "grid.linewidth"   : 0.5,
    "axes.titlesize"   : 14,
    "axes.titleweight" : "bold",
    "axes.titlecolor"  : NAVY,
})

def save(name):
    plt.savefig(f"plots/{name}", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: plots/{name}")

#loading processed data first
df = pd.read_csv("data/sp500_processed.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df.dropna(subset=["price"])

print(f"Rows: {len(df)} | {df['date'].min().date()} → {df['date'].max().date()}")

#feature columns (engineered indicators from preprocessing)
FEATURE_COLS = [
    "ma_7", "ma_30", "ma_90", "ma_200",
    "volatility_7", "volatility_30", "volatility_90",
    "bb_upper", "bb_lower", "bb_mid", "bb_width",
    "daily_return", "log_return",
    "golden_cross",
]

#dropping rows where any feature is NaN 
df_model = df.dropna(subset=FEATURE_COLS + ["price"]).reset_index(drop=True)
print(f"Rows after dropping NaN features: {len(df_model)}")


#training and testing split (80-20)
n        = len(df_model)
n_test   = max(60, int(n * 0.20))
n_train  = n - n_test

train_df = df_model.iloc[:n_train].copy()
test_df  = df_model.iloc[n_train:].copy()

print(f"\nTrain: {n_train} rows  |  Test: {n_test} rows")
print(f"Train period: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
print(f"Test  period: {test_df['date'].min().date()}  → {test_df['date'].max().date()}")


#Step1 — Prophet: Trend + Seasonality
print("\n1. Training Prophet model")

#Prophet requires columns named 'ds' (date) and 'y' (target)
prophet_train = train_df[["date", "price"]].rename(columns={"date": "ds", "price": "y"})
prophet_full  = df_model[["date", "price"]].rename(columns={"date": "ds", "price": "y"})

prophet = Prophet(
    yearly_seasonality  = True,
    weekly_seasonality  = True,
    daily_seasonality   = False,   # daily is too noisy for stock prices
    seasonality_mode    = "multiplicative",   # better for growing series
    changepoint_prior_scale = 0.05,           # controls trend flexibility
    seasonality_prior_scale = 10.0,
    interval_width      = 0.95,
)

#Adding monthly seasonality
prophet.add_seasonality(name="monthly", period=30.5, fourier_order=5)

prophet.fit(prophet_train)
print("Prophet training complete.")

#Prophet's in-sample predictions for the full dataset
prophet_pred_full = prophet.predict(prophet_full[["ds"]])

#Merging back
df_model["prophet_pred"] = prophet_pred_full["yhat"].values
df_model["prophet_lower"] = prophet_pred_full["yhat_lower"].values
df_model["prophet_upper"] = prophet_pred_full["yhat_upper"].values

#Residuals (what Prophet couldn't explain)
df_model["residual"] = df_model["price"] - df_model["prophet_pred"]

train_df = df_model.iloc[:n_train].copy()
test_df  = df_model.iloc[n_train:].copy()

#2. Random Forest: To learn the Residuals
print("\n2. Training Random Forest on residuals")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["residual"].values

X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["residual"].values

rf = RandomForestRegressor(
    n_estimators    = 300,
    max_depth       = 6,       #shallow to avoid overfitting
    min_samples_leaf= 10,
    max_features    = "sqrt",  #standard for RF — randomness for robustness
    random_state    = 42,
    n_jobs          = -1
)
rf.fit(X_train, y_train)
print("  Random Forest training complete.")

#RF predicts residuals
train_df["rf_residual_pred"] = rf.predict(X_train)
test_df["rf_residual_pred"]  = rf.predict(X_test)

#3. Hybrid forecast = Prophet + RF Residual Correction
print("\n3. Computing Hybrid Forecast")

train_df["hybrid_pred"] = train_df["prophet_pred"] + train_df["rf_residual_pred"]
test_df["hybrid_pred"]  = test_df["prophet_pred"]  + test_df["rf_residual_pred"]

#4. Evaluation Metrics
print("\n4. Evaluation Metrics")

def evaluate(actual, predicted, label):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot
    print(f"\n  [{label}]")
    print(f"    RMSE  : {rmse:,.2f}")
    print(f"    MAE   : {mae:,.2f}")
    print(f"    MAPE  : {mape:.2f}%")
    print(f"    R²    : {r2:.4f}")
    return {"model": label, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

actual_test = test_df["price"].values

metrics_prophet = evaluate(actual_test, test_df["prophet_pred"].values, "Prophet Only")
metrics_hybrid  = evaluate(actual_test, test_df["hybrid_pred"].values,  "Hybrid (Prophet + RF)")

print(f"\n  Improvement from RF correction:")
print(f"    RMSE reduced by : {metrics_prophet['RMSE'] - metrics_hybrid['RMSE']:,.2f}")
print(f"    MAPE reduced by : {metrics_prophet['MAPE'] - metrics_hybrid['MAPE']:.2f}%")

#Plot 1: Predicted vs Actual
print("\n1. Plotting results")

fig, ax = plt.subplots(figsize=(15, 6))

#Training fit
ax.plot(train_df["date"], train_df["price"],
        color="#CBD5E1", linewidth=0.5, label="Training Data")

#Test period
ax.plot(test_df["date"], test_df["price"],
        color=NAVY, linewidth=1.2, label="Actual (Test)")
ax.plot(test_df["date"], test_df["prophet_pred"],
        color=MUTED, linewidth=1.0, linestyle=":", label="Prophet Only")
ax.plot(test_df["date"], test_df["hybrid_pred"],
        color=GOLD, linewidth=1.8, linestyle="--", label="Hybrid Forecast")

#Confidence band from Prophet
ax.fill_between(test_df["date"],
                test_df["prophet_lower"], test_df["prophet_upper"],
                color=TEAL, alpha=0.10, label="Prophet 95% CI")

#Metrics annotation
metrics_txt = (
    f"Hybrid Model — Test Set\n"
    f"RMSE: {metrics_hybrid['RMSE']:,.1f}   "
    f"MAE: {metrics_hybrid['MAE']:,.1f}   "
    f"MAPE: {metrics_hybrid['MAPE']:.2f}%   "
    f"R²: {metrics_hybrid['R2']:.4f}"
)
ax.annotate(metrics_txt,
            xy=(0.01, 0.97), xycoords="axes fraction",
            fontsize=9, color=NAVY, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#E2E8F0", alpha=0.9))

ax.set_title("Hybrid Model (Prophet + Random Forest) — Forecast vs Actual")
ax.set_xlabel("Date")
ax.set_ylabel("S&P 500 Index Value")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
save("08_hybrid_forecast_vs_actual.png")

#Plot 2 — Prophet Components (trend, weekly, yearly)
fig = prophet.plot_components(prophet.predict(prophet_train[["ds"]]))
fig.patch.set_facecolor(BG)
plt.suptitle("Prophet Model Components", fontsize=14, fontweight="bold",
             color=NAVY, y=1.01)
plt.tight_layout()
save("09_prophet_components.png")

#Plot 3 — Feature Importance (which indicators matter most)
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors = [TEAL if v >= importances.median() else MUTED for v in importances.values]
ax.barh(importances.index, importances.values, color=colors, edgecolor="none")
ax.set_title("Random Forest — Feature Importance\n(Which technical indicators correct Prophet's errors)")
ax.set_xlabel("Importance Score")
ax.grid(True, axis="x", linestyle="--", alpha=0.5)
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8, color=NAVY)
plt.tight_layout()
save("10_feature_importance.png")

#Plot 4 — Residuals: Prophet vs Hybrid
prophet_residuals = test_df["price"].values - test_df["prophet_pred"].values
hybrid_residuals  = test_df["price"].values - test_df["hybrid_pred"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, resid, label, color in zip(
    axes,
    [prophet_residuals, hybrid_residuals],
    ["Prophet Only Residuals", "Hybrid Model Residuals"],
    [MUTED, TEAL]
):
    ax.plot(test_df["date"], resid, color=color, linewidth=0.8, alpha=0.8)
    ax.axhline(0, color=RED, linewidth=1, linestyle="--")
    ax.fill_between(test_df["date"], resid, 0,
                    where=(resid > 0), color=TEAL, alpha=0.15)
    ax.fill_between(test_df["date"], resid, 0,
                    where=(resid < 0), color=RED,  alpha=0.15)
    ax.set_title(f"{label}\nStd: {np.std(resid):,.1f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#FFFFFF")

plt.suptitle("Residual Comparison: Prophet vs Hybrid", fontsize=14,
             fontweight="bold", color=NAVY)
plt.tight_layout()
save("11_residual_comparison.png")

#Step 5 — FUTURE 30-DAY FORECAST
print("\n5. Future 30-Day Forecast")

#Generate future dates
last_date    = df_model["date"].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                             periods=30, freq="B")   # business days only

future_prophet_input = pd.DataFrame({"ds": future_dates})
future_prophet_pred  = prophet.predict(future_prophet_input)

#For RF features: carry forwarding last known values
last_features = df_model[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
future_rf_correction = rf.predict(
    np.repeat(last_features, len(future_dates), axis=0)
)

future_df = pd.DataFrame({
    "date"           : future_dates,
    "prophet_forecast": future_prophet_pred["yhat"].values,
    "rf_correction"  : future_rf_correction,
    "hybrid_forecast": future_prophet_pred["yhat"].values + future_rf_correction,
    "lower_95"       : future_prophet_pred["yhat_lower"].values,
    "upper_95"       : future_prophet_pred["yhat_upper"].values,
})

print("\n30-Day Forward Forecast (Business Days):")
print(future_df[["date", "hybrid_forecast", "lower_95", "upper_95"]]
      .round(2).to_string(index=False))

#Plotting future forecast
recent = df_model[df_model["date"] >= df_model["date"].max() - pd.Timedelta(days=180)]

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(recent["date"], recent["price"],
        color=NAVY, linewidth=1.2, label="Historical (last 6 months)")
ax.fill_between(future_df["date"],
                future_df["lower_95"], future_df["upper_95"],
                color=TEAL, alpha=0.15, label="95% CI (Prophet)")
ax.plot(future_df["date"], future_df["hybrid_forecast"],
        color=GOLD, linewidth=2.0, linestyle="--", label="30-Day Hybrid Forecast")
ax.scatter(future_df["date"], future_df["hybrid_forecast"],
           color=GOLD, s=25, zorder=5)

ax.axvline(last_date, color=RED, linewidth=1.0,
           linestyle=":", alpha=0.7, label="Forecast start")

ax.set_title("S&P 500 — 30-Day Future Forecast (Hybrid Model)")
ax.set_xlabel("Date")
ax.set_ylabel("Index Value")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
save("12_future_forecast_30day.png")

#saving results
test_df[["date", "price", "prophet_pred", "hybrid_pred"]].to_csv(
    "data/test_forecast_results.csv", index=False)
future_df.to_csv("data/future_forecast_30day.csv", index=False)

#Metrics summary
metrics_df = pd.DataFrame([metrics_prophet, metrics_hybrid])
metrics_df.to_csv("data/model_metrics.csv", index=False)

#Final summary 
print("\n")
print("=" * 56)
print("HYBRID MODEL — FINAL SUMMARY")
print("=" * 56)
print(f"  Architecture   : Prophet + Random Forest (Residual)")
print(f"  Features used  : {len(FEATURE_COLS)} technical indicators")
print(f"  RF Trees       : 300  |  Max Depth: 6")
print(f"  Train period   : {train_df['date'].min().date()} → {train_df['date'].max().date()}")
print(f"  Test period    : {test_df['date'].min().date()}  → {test_df['date'].max().date()}")
print()
print(f"  Prophet Only:")
print(f"    RMSE: {metrics_prophet['RMSE']:,.2f}  |  MAPE: {metrics_prophet['MAPE']:.2f}%  |  R²: {metrics_prophet['R2']:.4f}")
print()
print(f"  Hybrid (Prophet + RF):")
print(f"    RMSE: {metrics_hybrid['RMSE']:,.2f}  |  MAPE: {metrics_hybrid['MAPE']:.2f}%  |  R²: {metrics_hybrid['R2']:.4f}")
print()
print(f"  RMSE improvement : {metrics_prophet['RMSE'] - metrics_hybrid['RMSE']:,.2f}")
print(f"  MAPE improvement : {metrics_prophet['MAPE'] - metrics_hybrid['MAPE']:.2f}%")
print("=" * 56)