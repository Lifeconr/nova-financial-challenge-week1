import pandas as pd
import numpy as np
import talib
import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the stock symbols (focused dataset)
stocks = ['AAPL', 'AMZN', 'GOOG', 'META']
data_dir = 'Data/yfinance_data/'

# Load stock data into a dictionary of DataFrames
stock_data = {}
for stock in stocks:
    file_path = os.path.join(data_dir, f'{stock}_historical_data.csv')
    df = pd.read_csv(file_path)
    # Ensure required columns exist
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if all(col in df.columns for col in required_cols):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        stock_data[stock] = df
    else:
        print(f"Missing required columns in {stock}_historical_data.csv")

print("Data loaded successfully for selected stocks.")