# Stock Market Analysis & Forecasting

A comprehensive project for analyzing and forecasting S&P 500 stock market data using Python.

## Overview

This project collects historical S&P 500 data from FRED (Federal Reserve Economic Data) and performs preprocessing to prepare it for analysis and modeling.

## Steps

### 1. Data Collection

- Fetches S&P 500 index data from FRED API
- Saves raw data to `data/sp500_raw.csv`
- Run: `python scripts/data_collection.py`

<img width="971" height="839" alt="image" src="https://github.com/user-attachments/assets/72bc4074-4491-4188-9cff-f9239a8b5765" />

### 2. Preprocessing of Data

- Handles missing values
- Adds time-based features (year, month, quarter, etc.)
- Calculates daily returns, moving averages, volatility, Bollinger Bands
- Saves processed data to `data/sp500_processed.csv`
- Run: `python scripts/data_preprocessing.py`

<img width="1485" height="474" alt="image" src="https://github.com/user-attachments/assets/861a7270-65ed-492e-bc82-feff6d9651c0" />

<img width="1470" height="861" alt="image" src="https://github.com/user-attachments/assets/d87eb61f-76d0-49b6-9178-6883ffe8151f" />

---

## 3. Modeling & Forecasting

- Implements a hybrid forecasting model using:
  - **Facebook Prophet** for trend and seasonality
  - **Random Forest Regressor** for residual correction

- Combines both to generate improved predictions
- Outputs:
  - Forecasted values
  - Confidence intervals
  - Feature importance

- Run: `python scripts/model.py`

<img width="1619" height="445" alt="image" src="https://github.com/user-attachments/assets/de825e8b-f637-401f-8f81-d16e5b2fad45" />

---

## 4. Streamlit Dashboard

An interactive web app for real-time analysis and forecasting.

### Features:

- Select custom date ranges
- Choose forecast horizon
- Visualize:
  - Price trends
  - Moving averages
  - Bollinger Bands
  - Forecast results
  - Feature importance

- Displays model performance metrics (RMSE, MAE, MAPE, R²)

<img width="2538" height="1308" alt="image" src="https://github.com/user-attachments/assets/60fb3129-901d-4d08-954a-33cdc1b5268d" />
<img width="2522" height="1317" alt="image" src="https://github.com/user-attachments/assets/d9573a16-2d3e-46e3-8ecd-ff774aac450a" />

### Run the app:

```bash
streamlit run app.py
```

---

## 5. Power BI Dashboard

- Interactive business intelligence dashboard built using Power BI
- Visual insights include:
  - Historical price trends
  - Volatility analysis
  - Moving averages comparison
  - KPI summaries

- User View:
  <img width="1879" height="1057" alt="image" src="https://github.com/user-attachments/assets/11b850c6-26d0-449b-b88d-430c994d7bda" />

- Analyst View:
  <img width="1902" height="1053" alt="image" src="https://github.com/user-attachments/assets/6a14e69f-cdcb-4140-a602-53bac26f51d0" />

---

## 6. Docker Support

Containerized setup for easy deployment.

### Build Docker Image:

```bash
docker build -t sp500-forecast .
```

### Run Container:

```bash
docker run -p 8501:8501 sp500-forecast
```

### Access App:

```
http://localhost:8501
```

---

## Requirements

- Python 3.8+
- pandas
- pandas-datareader
- python-dotenv
- numpy
- scikit-learn
- prophet
- streamlit
- plotly

---

## Setup

1. Clone the repository
2. Create virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate:

   ```bash
   .venv\Scripts\activate   # Windows
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Set FRED API key in `.env` file:

   ```
   FRED_API_KEY=your_key_here
   ```

6. Run data collection and preprocessing scripts

---

## Data

- Raw data: `data/sp500_raw.csv`
- Processed data: `data/sp500_processed.csv`

---

## Contributors

- [samyamehta16](https://github.com/samyamehta16)
- [palakg29](https://github.com/palakg29)

---

## License

MIT License
