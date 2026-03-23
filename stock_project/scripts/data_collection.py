#very basic library imports for data collection
import os
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime
from dotenv import load_dotenv

#loading my fred api key 
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

if not FRED_API_KEY:
    print('=' * 60)
    print('WARNING: FRED_API_KEY not set, some issue')
print('\nFetching S&P 500 data from FRED...')

START_DATE = '1950-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

try:
    sp500 = pdr.DataReader(
        'SP500',
        'fred',
        start=START_DATE,
        end=END_DATE,
        api_key=FRED_API_KEY if FRED_API_KEY else None
    )
except Exception as e:
    print(f'Error fetching data: {e}')
    raise

sp500.columns = ['price']
sp500.index.name = 'date'
sp500 = sp500.reset_index()
sp500['date'] = pd.to_datetime(sp500['date'])
sp500 = sp500.sort_values('date').reset_index(drop=True)

#basic description and stats printed
print(f'\nShape        : {sp500.shape}')
print(f"Date range   : {sp500['date'].min().date()} → {sp500['date'].max().date()}")
print(f'Missing values: {sp500['price'].isna().sum()}')
print('\nFirst 5 rows:')
print(sp500.head())
print('\nLast 5 rows:')
print(sp500.tail())

#making a data directory and saving the raw data as a csv file
os.makedirs('data', exist_ok=True)
sp500.to_csv('data/sp500_raw.csv', index=False)
print('\nRaw data saved to: data/sp500_raw.csv')
