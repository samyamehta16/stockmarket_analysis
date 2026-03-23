# =============================================================================
# Stock Market Analysis & Forecasting
# Script: EDA Visualizations (Step 3)
# Description: Generates 7 analytical charts from processed S&P 500 data
# Usage: python eda_visualizations.py
# Requires: data/sp500_processed.csv (run earlier fetch/processing scripts first)
# =============================================================================

# --- Install dependencies (run once in terminal) ---
# pip install pandas matplotlib seaborn scipy numpy

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

# Create output folder for plots
os.makedirs("plots", exist_ok=True)


# =============================================================================
# CONFIGURATION — Colors & Chart Style
# =============================================================================

NAVY  = "#1A2B4A"   # titles, axes
TEAL  = "#0E9F8E"   # primary line color
GOLD  = "#F0B429"   # secondary / highlight
RED   = "#EF4444"   # negative values / crisis zones
MUTED = "#8A9BB5"   # tick labels
BG    = "#F4F6FA"   # figure background
WHITE = "#FFFFFF"   # axes background

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : WHITE,
    "axes.edgecolor"    : "#E2E8F0",
    "axes.labelcolor"   : NAVY,
    "xtick.color"       : MUTED,
    "ytick.color"       : MUTED,
    "grid.color"        : "#E2E8F0",
    "grid.linewidth"    : 0.5,
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.titlecolor"   : NAVY,
    "axes.labelsize"    : 11,
    "legend.framealpha" : 0.9,
})


# =============================================================================
# HELPER — Save chart to /plots folder
# =============================================================================

def save(filename):
    plt.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved → plots/{filename}")


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading processed data...")
df = pd.read_csv("data/sp500_processed.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Rows: {len(df)}  |  Range: {df['date'].min().date()} → {df['date'].max().date()}")


# =============================================================================
# PLOT 1 — Long-Term Price Trend with Moving Averages
# =============================================================================

print("\nPlot 1: Price trend with moving averages...")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(df["date"], df["price"],  color="#CBD5E1", linewidth=0.6, label="Daily Price",  alpha=0.9)
ax.plot(df["date"], df["ma_30"],  color=TEAL,      linewidth=1.2, label="30-Day MA")
ax.plot(df["date"], df["ma_200"], color=GOLD,      linewidth=1.8, label="200-Day MA")

ax.set_title("S&P 500 — Historical Price Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Index Value")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
save("01_price_trend.png")


# =============================================================================
# PLOT 2 — Daily Returns Distribution
# =============================================================================

print("Plot 2: Daily returns distribution...")

returns    = df["daily_return"].dropna()
mu, sigma  = returns.mean(), returns.std()

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(returns, bins=120, density=True, color=TEAL, alpha=0.65, edgecolor="none")

# Overlay normal distribution curve for comparison
x = np.linspace(returns.min(), returns.max(), 500)
ax.plot(x, stats.norm.pdf(x, mu, sigma),
        color=GOLD, linewidth=2, linestyle="--", label="Normal Distribution")
ax.axvline(mu, color=RED, linewidth=1.5, label=f"Mean: {mu:.3f}%")

ax.set_title("Distribution of Daily Returns")
ax.set_xlabel("Daily Return (%)")
ax.set_ylabel("Density")
ax.set_xlim(-8, 8)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
save("02_return_distribution.png")


# =============================================================================
# PLOT 3 — Rolling Volatility with Crisis Period Annotations
# =============================================================================

print("Plot 3: Rolling volatility with crisis annotations...")

vol = df.dropna(subset=["volatility_30"])

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(vol["date"], vol["volatility_30"], color=TEAL, linewidth=0.6, alpha=0.7)

# Smoothed trend line (90-day window)
smoothed = uniform_filter1d(vol["volatility_30"].values, size=90)
ax.plot(vol["date"], smoothed, color=GOLD, linewidth=1.8, label="Trend (smoothed)")

# Annotate major market crisis periods
crises = [
    ("2001-09-01", "2002-06-01", "Dot-com /\n9-11"),
    ("2008-09-01", "2009-06-01", "2008\nCrisis"),
    ("2020-02-01", "2020-06-01", "COVID-19"),
]
for start, end, label in crises:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               alpha=0.12, color=RED, zorder=0)
    ax.text(pd.Timestamp(start), vol["volatility_30"].max() * 0.92,
            label, fontsize=8, color=RED, fontweight="bold")

ax.set_title("30-Day Rolling Volatility — S&P 500")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (Std Dev of Daily Returns %)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
save("03_volatility.png")


# =============================================================================
# PLOT 4 — Monthly Seasonality (Average Daily Return by Month)
# =============================================================================

print("Plot 4: Monthly seasonality...")

month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

monthly = (
    df.dropna(subset=["daily_return"])
      .groupby("month_name")["daily_return"]
      .mean()
      .reindex(month_order)
)

colors = [TEAL if v >= 0 else RED for v in monthly.values]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(monthly.index, monthly.values, color=colors, width=0.65, edgecolor="none")
ax.axhline(0, color=NAVY, linewidth=0.8)

# Add value labels on each bar
for bar, val in zip(bars, monthly.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + (0.003 if val >= 0 else -0.005),
        f"{val:.3f}%",
        ha="center",
        va="bottom" if val >= 0 else "top",
        fontsize=9, color=NAVY, fontweight="bold"
    )

ax.set_title("Average Daily Return by Month — Seasonality Analysis")
ax.set_xlabel("Month")
ax.set_ylabel("Average Daily Return (%)")
ax.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
save("04_seasonality.png")


# =============================================================================
# PLOT 5 — Annual Cumulative Returns
# =============================================================================

print("Plot 5: Annual cumulative returns...")

annual = (
    df.dropna(subset=["daily_return"])
      .groupby("year")["daily_return"]
      .sum()
      .reset_index()
)
annual.columns = ["year", "annual_return"]

colors = [TEAL if v >= 0 else RED for v in annual["annual_return"]]

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(annual["year"].astype(str), annual["annual_return"],
       color=colors, width=0.75, edgecolor="none")
ax.axhline(0, color=NAVY, linewidth=0.8)

ax.set_title("Annual Cumulative Returns — S&P 500")
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Return (%)")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
save("05_annual_returns.png")


# =============================================================================
# PLOT 6 — Bollinger Bands (Last 2 Years)
# =============================================================================

print("Plot 6: Bollinger Bands...")

recent = df[
    df["date"] >= df["date"].max() - pd.Timedelta(days=730)
].dropna(subset=["bb_upper"])

fig, ax = plt.subplots(figsize=(13, 5))

ax.fill_between(recent["date"], recent["bb_lower"], recent["bb_upper"],
                color=TEAL, alpha=0.12, label="Bollinger Band (±2 SD)")
ax.plot(recent["date"], recent["price"],    color=NAVY, linewidth=1.0, label="Price")
ax.plot(recent["date"], recent["bb_mid"],   color=TEAL, linewidth=1.0,
        linestyle="--", label="20-Day MA (mid)")
ax.plot(recent["date"], recent["bb_upper"], color=GOLD, linewidth=0.8, label="Upper Band")
ax.plot(recent["date"], recent["bb_lower"], color=GOLD, linewidth=0.8, label="Lower Band")

ax.set_title("Bollinger Bands — Last 2 Years")
ax.set_xlabel("Date")
ax.set_ylabel("Index Value")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
save("06_bollinger_bands.png")


# =============================================================================
# PLOT 7 — Feature Correlation Heatmap
# =============================================================================

print("Plot 7: Correlation heatmap...")

corr_cols = ["price", "daily_return", "volatility_30", "ma_30", "ma_200", "bb_width"]
corr      = df[corr_cols].dropna().corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    linewidths=0.5,
    ax=ax,
    center=0,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 10}
)
ax.set_title("Feature Correlation Heatmap")

plt.tight_layout()
save("07_correlation_heatmap.png")


# =============================================================================
# EDA SUMMARY — Printed to terminal
# =============================================================================

print("\n" + "=" * 50)
print("           EDA SUMMARY")
print("=" * 50)
print(f"Total trading days   : {len(df)}")
print(f"Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Lowest index value   : {df['price'].min():,.2f}")
print(f"Highest index value  : {df['price'].max():,.2f}")
print(f"Overall growth       : {(df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100:.2f}%")
print(f"Positive return days : {(df['daily_return'] > 0).sum()}")
print(f"Negative return days : {(df['daily_return'] < 0).sum()}")
print("=" * 50)
print("\nAll 7 plots saved to → ./plots/")
print("Next step: run 04_arima_modeling.py")