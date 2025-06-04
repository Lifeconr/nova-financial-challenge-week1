import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(stocks, news_path='Data/corrected_analyst_ratings.csv', stock_dir='Data/yfinance_data/'):
    """
    Load news and stock data.

    Args:
        stocks (list): List of stock symbols.
        news_path (str): Path to the news dataset CSV.
        stock_dir (str): Directory containing stock data CSVs.

    Returns:
        tuple: (news_df, stock_data)
    """
    try:
        news_df = pd.read_csv(news_path)
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df['date_only'] = news_df['date'].dt.date
    except FileNotFoundError:
        raise FileNotFoundError(f"News file {news_path} not found.")

    stock_data = {}
    for stock in stocks:
        file_path = os.path.join(stock_dir, f'{stock}_historical_data.csv')
        try:
            df = pd.read_csv(file_path)

            # Handle column renaming if needed
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not set(expected_columns).issubset(df.columns):
                print(f"{stock} file columns: {list(df.columns)}")
                
                # Try renaming based on pattern
                if len(df.columns) >= 6:
                    print(f"Renaming columns for {stock}")
                    df.columns = ['Unnamed'] + expected_columns  # Adjust if there’s an extra column like index
                    df = df.drop(columns=['Unnamed'], errors='ignore')
                else:
                    print(f"Warning: Could not rename columns for {stock} due to insufficient columns.")
                    continue

            # Convert to proper types
            for col in expected_columns[1:]:  # Skip 'Date'
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=expected_columns[1:])

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.index.name = 'Date'
            stock_data[stock] = df

        except Exception as e:
            print(f"Error processing {stock}: {e}")


    if not stock_data:
        raise ValueError("No valid stock data loaded. Check stock CSVs for formatting issues.")

    return news_df, stock_data

def perform_sentiment_analysis(news_df):
    """
    Perform sentiment analysis on news headlines.

    Args:
        news_df (pd.DataFrame): News DataFrame with 'headline' and 'date_only' columns.

    Returns:
        pd.DataFrame: Daily average sentiment scores.
    """
    if 'headline' not in news_df.columns:
        raise KeyError("Column 'headline' not found in news DataFrame.")

    news_df['sentiment'] = news_df['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    daily_sentiment = news_df.groupby('date_only')['sentiment'].mean().reset_index()
    daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only']).dt.normalize()

    return daily_sentiment

def calculate_correlations(stock_data, daily_sentiment, stocks, output_dir='notebooks/figures/'):
    """
    Calculate correlations between stock returns and news sentiment, and generate scatter plots.

    Args:
        stock_data (dict): Dictionary of stock DataFrames.
        daily_sentiment (pd.DataFrame): DataFrame with daily sentiment scores.
        stocks (list): List of stock symbols.
        output_dir (str): Directory to save scatter plots.

    Returns:
        dict: Correlations for each stock.
    """
    os.makedirs(output_dir, exist_ok=True)
    correlations = {}

    for stock in stocks:
        if stock not in stock_data:
            print(f"Skipping {stock}: No valid stock data loaded.")
            correlations[stock] = np.nan
            continue

        df = stock_data[stock].copy()

        if 'Close' not in df.columns:
            print(f"Skipping {stock}: 'Close' column missing.")
            correlations[stock] = np.nan
            continue

        df['Daily Return'] = df['Close'].pct_change()
        df = df.reset_index()
        df['date_only'] = pd.to_datetime(df['Date']).dt.normalize()
        # Ensure daily_sentiment['date_only'] is also datetime64[ns]
        daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only']).dt.normalize()

        merged_df = pd.merge(df[['date_only', 'Daily Return']], daily_sentiment, on='date_only', how='inner')

        merged_df = pd.merge(df[['date_only', 'Daily Return']], daily_sentiment, on='date_only', how='inner')

        if not merged_df.empty:
            correlation = merged_df['Daily Return'].corr(merged_df['sentiment'])
            correlations[stock] = correlation
            print(f"Correlation between {stock} daily returns and news sentiment: {correlation:.2f}")

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='sentiment', y='Daily Return', data=merged_df)
            plt.title(f'{stock} Daily Returns vs. News Sentiment (Correlation: {correlation:.2f})')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Daily Return')
            plt.grid()
            plt.savefig(os.path.join(output_dir, f'{stock}_sentiment_return_scatter.png'))
            plt.close()
        else:
            print(f"No overlapping data for {stock}. Skipping correlation.")
            correlations[stock] = np.nan

    return correlations

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    news_path = os.path.join(project_root, '..', 'Data', 'corrected_analyst_ratings.csv')
    stock_dir = os.path.join(project_root, '..', 'Data', 'yfinance_data')
    stocks = ['AAPL', 'AMZN', 'GOOG', 'META']

    news_df, stock_data = load_data(stocks, news_path=news_path, stock_dir=stock_dir)
    daily_sentiment = perform_sentiment_analysis(news_df)
    correlations = calculate_correlations(stock_data, daily_sentiment, stocks)
