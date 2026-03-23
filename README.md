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


## Requirements

- Python 3.8+
- pandas
- pandas-datareader
- python-dotenv
- numpy

## Setup

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt` (if available) or manually install packages
5. Set FRED API key in `.env` file: `FRED_API_KEY=your_key_here`
6. Run data collection and preprocessing scripts

## Data

- Raw data: `data/sp500_raw.csv`
- Processed data: `data/sp500_processed.csv`

## Contributors

- [samyamehta16](https://github.com/samyamehta16)
- Palak Goyal

## License

MIT License
