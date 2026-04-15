# Stock Market Analysis Dashboard — Power BI Setup Guide


## Overview

This project presents a dynamic Stock Market Analysis Dashboard built using Power BI. It visualizes S&P 500 data, forecasting results, and key performance indicators using data generated via Python scripts.

The dashboard is divided into:

* Market Overview (General Users)
* Analyst View (Advanced Insights)


## Folder Structure

```
your_project/
│── powerbi/
│   ├── data/
│   │   ├── sp500_prices.csv
│   │   ├── forecast_results.csv
│   │   ├── future_forecast.csv
│   │   ├── feature_importance.csv
│   │   ├── monthly_seasonality.csv
│   │   ├── annual_returns.csv
│   │   ├── model_metrics.csv
│   │   ├── kpis.csv
│   ├── refresh_data.py
│── POWERBI_SETUP.md
```


## Step 1 — Generate the Data Files

Run the following command to generate all required CSV files:

```
cd your_project
python powerbi/refresh_data.py
```

This will generate 8 CSV files in the `powerbi/data/` directory.

---

## Step 2 — Connect Power BI to the CSVs

1. Open **Power BI Desktop**
2. Click **Home → Get Data → Text/CSV**
3. Navigate to your `powerbi/data/` folder
4. Import all 8 CSV files one by one
5. For each file, click **Load**

Files to import:

* `sp500_prices.csv`
* `forecast_results.csv`
* `future_forecast.csv`
* `feature_importance.csv`
* `monthly_seasonality.csv`
* `annual_returns.csv`
* `model_metrics.csv`
* `kpis.csv`

---

## Step 3 — Build Dashboard 1 (General User)

### Page Name: "Market Overview"

| Visual           | Type       | Data                | Fields                         |
| ---------------- | ---------- | ------------------- | ------------------------------ |
| Current Price    | Card       | kpis                | current_price                  |
| Price Change     | Card       | kpis                | pct_change                     |
| Market Mood      | Card       | kpis                | market_mood                    |
| All-Time High    | Card       | kpis                | all_time_high                  |
| Price Trend      | Line Chart | sp500_prices        | date (X), price (Y), ma_30 (Y) |
| Volatility Gauge | Gauge      | kpis                | volatility_30d                 |
| Seasonality      | Bar Chart  | monthly_seasonality | month (X), avg_return (Y)      |
| Annual Returns   | Bar Chart  | annual_returns      | year (X), annual_return (Y)    |
| Period Slicer    | Slicer     | sp500_prices        | date                           |

### Tips:

* Use **Conditional Formatting** for Market Mood:

  * Bullish = Green
  * Bearish = Red
  * Neutral = Yellow
* Apply conditional colors for Annual Returns:

  * Positive = Green/Teal
  * Negative = Red

---

## Step 4 — Build Dashboard 2 (Analyst View)

### Page Name: "Analyst View"

| Visual             | Type       | Data               | Fields                                             |
| ------------------ | ---------- | ------------------ | -------------------------------------------------- |
| RMSE               | Card       | model_metrics      | RMSE                                               |
| MAE                | Card       | model_metrics      | MAE                                                |
| MAPE               | Card       | model_metrics      | MAPE_pct                                           |
| R² Score           | Card       | model_metrics      | R2                                                 |
| Forecast vs Actual | Line Chart | forecast_results   | date (X), price + hybrid_pred + prophet_pred (Y)   |
| Future Forecast    | Line Chart | future_forecast    | date (X), forecast + lower_95 + upper_95 (Y)       |
| Feature Importance | Bar Chart  | feature_importance | feature (Y), importance (X)                        |
| Bollinger Bands    | Line Chart | sp500_prices       | date (X), price + bb_upper + bb_lower + bb_mid (Y) |
| Volatility         | Area Chart | sp500_prices       | date (X), volatility_30 (Y)                        |
| Last Updated       | Card       | model_metrics      | last_updated                                       |

### Tips:

* Use distinct colors for Forecast vs Actual:

  * Actual = Navy
  * Hybrid = Gold
  * Prophet = Grey
* Use area shading between lower and upper bounds for forecast confidence
* Sort Feature Importance in descending order
* Highlight top features using color emphasis
* Add a **Date Slicer** connected to `sp500_prices[date]`

---

## Step 5 — Make Data Dynamic (Auto-Refresh)

### Option A: Manual Refresh

Run:

```
python powerbi/refresh_data.py
```

Then click **Refresh** in Power BI.

---

### Option B: Scheduled Refresh (Windows Task Scheduler)

1. Open Task Scheduler
2. Click **Create Basic Task**
3. Name: `SP500 Power BI Refresh`
4. Trigger: Daily
5. Action: Start a program
6. Program: `python`
7. Arguments: `D:\your_project\powerbi\refresh_data.py`

This automates daily data updates.

---

### Option C: Power BI Python Script (Advanced)

1. Go to **File → Options → Python scripting**
2. Set your Python environment
3. Add a Python visual and use:

```python
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime

df = pdr.DataReader("SP500", "fred",
                    start="2020-01-01",
                    end=datetime.today().strftime("%Y-%m-%d"))
df.columns = ["price"]
df = df.reset_index()
dataset = df
```

---

## Step 6 — Final Touches

* Set page layout to **16:9**
* Use bookmarks for navigation
* Add a dashboard title
* Export as PDF for submission

---

## Color Reference

| Element    | Hex     |
| ---------- | ------- |
| Primary    | #1A2B4A |
| Accent     | #0E9F8E |
| Forecast   | #F0B429 |
| Positive   | #22C55E |
| Negative   | #EF4444 |
| Labels     | #8A9BB5 |
| Background | #F4F6FA |

---

## Conclusion

This setup provides a fully dynamic and refreshable Power BI dashboard for stock market analysis. By integrating Python-based data generation with Power BI visualizations, the system ensures up-to-date insights for both general users and analysts.
