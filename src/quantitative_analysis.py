import pandas as pd
import numpy as np
import ta
import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_stock_data(data_dir):
    stock_data = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('_historical_data.csv'):
            stock = file_name.replace('_historical_data.csv', '')
            df = pd.read_csv(os.path.join(data_dir, file_name))
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in {file_name}. Found columns: {df.columns.tolist()}")
            
            # Convert date and numeric columns
            df['Date'] = pd.to_datetime(df['Date'])
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            stock_data[stock] = df
    return stock_data


def calculate_technical_indicators(stock_data):
    """
    Calculate technical indicators (SMA, RSI, MACD) for each stock.
    
    Args:
        stock_data (dict): Dictionary of stock DataFrames.
    
    Returns:
        dict: Updated stock_data with new indicator columns.
    """
    if not stock_data:
        raise ValueError("stock_data is empty. Load stock data before calculating indicators.")
    for stock, df in stock_data.items():
        # Simple Moving Average (SMA) - 50-day
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Relative Strength Index (RSI) - 14-day
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Moving Average Convergence Divergence (MACD)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        stock_data[stock] = df
    print("Technical indicators calculated for selected stocks.")
    return stock_data

def calculate_financial_metrics(stock_data):
    financial_metrics = {}
    
    if not stock_data:
        raise ValueError("stock_data is empty. Load stock data before calculating metrics.")

    for stock, df in stock_data.items():
        daily_returns = df['Close'].pct_change().dropna()
        
        if not daily_returns.empty:
            annualized_return = (1 + daily_returns.mean()) ** 252 - 1
            annualized_volatility = daily_returns.std() * (252 ** 0.5)

            financial_metrics[stock] = {
                'Annualized Return': annualized_return,
                'Annualized Volatility': annualized_volatility
            }

    return pd.DataFrame(financial_metrics).T


def visualize_data(stock_data, output_dir='notebooks/figures/'):
    """
    Generate visualizations for stock prices, SMA, RSI, and MACD.
    
    Args:
        stock_data (dict): Dictionary of stock DataFrames.
        output_dir (str): Directory to save visualization images.
    """
    if not stock_data:
        raise ValueError("stock_data is empty. Load stock data before visualizing.")
    os.makedirs(output_dir, exist_ok=True)
    for stock, df in stock_data.items():
        # Plot 1: Stock Price with SMA
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')
        plt.plot(df.index, df['SMA_50'], label='50-day SMA', color='orange')
        plt.title(f'{stock} Stock Price with 50-day SMA')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'{stock}_price_sma.png'))
        plt.close()
        
        # Plot 2: RSI
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought')
        plt.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold')
        plt.title(f'{stock} RSI (14-day)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'{stock}_rsi.png'))
        plt.close()
        
        # Plot 3: MACD
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['MACD'], label='MACD', color='blue')
        plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange')
        plt.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='grey', alpha=0.3)
        plt.title(f'{stock} MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'{stock}_macd.png'))
        plt.close()
    print(f"Visualizations for {', '.join(stock_data.keys())} saved in {output_dir}")

if __name__ == "__main__":
    # Define the stock symbols (focused dataset)
    stocks = ['AAPL', 'AMZN', 'GOOG', 'META']
    
    # Execute the workflow
    stock_data = load_stock_data(stocks)
    stock_data = calculate_technical_indicators(stock_data)
    metrics_df = calculate_financial_metrics(stock_data)
    visualize_data(stock_data)