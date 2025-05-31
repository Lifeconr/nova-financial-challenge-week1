# Nova Financial Challenge (Week 1)

## Overview

This repository contains the code and analysis for the Nova Financial Challenge (Week 1), The project focuses on financial data analysis, including Exploratory Data Analysis (EDA), quantitative analysis using technical indicators, and correlation studies between news sentiment and stock movements. The dataset includes financial news articles and stock price data for seven companies (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA).

## Repository Structure
nova-financial-challenge-week1/
├── Data/
│ ├── raw_analyst_ratings.csv # Original financial news dataset (1,407,328 rows, 311.40 MB) - Excluded from Git due to size limits
│ ├── corrected_analyst_ratings.csv # Preprocessed news dataset (55,987 rows)
│ ├── invalid_dates.csv # Rows with unparseable dates (273.67 MB) - Excluded from Git due to size limits
│ └── yfinance_data/ # Stock price data for seven companies (e.g., AAPL_historical_data.csv)
├── src/
│ ├── preprocess_dates.py # Script to standardize dates in the news dataset
│ ├── financial_eda.py # Class for performing EDA (descriptive statistics, text analysis, etc.)
├── notebooks/
│ └── eda_sentiment_analysis.ipynb # Notebook for EDA and planned sentiment analysis
├── .gitignore # Excludes the Data/ directory from version control
└── README.md # Project overview and instructions (this file)
Install Dependencies: Install Python 3.11 and required libraries:

bash

Copy
pip install pandas matplotlib seaborn scikit-learn nltk textblob
For Task 2, install additional libraries:

bash

Copy
pip install TA-Lib yfinance
Provide Datasets: The Data/ directory is not tracked in Git due to GitHub’s 100 MB file size limit. You must manually place the datasets in the Data/ directory. These include:

raw_analyst_ratings.csv (311.40 MB)
invalid_dates.csv (273.67 MB)
corrected_analyst_ratings.csv (smaller, can be regenerated)
Alternatively, regenerate corrected_analyst_ratings.csv by running preprocess_dates.py if the raw data is available.