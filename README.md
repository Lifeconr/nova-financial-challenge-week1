# Nova Financial Challenge (Week 1)

## Overview

This repository contains the code and analysis for the Nova Financial Challenge (Week 1), The project focuses on financial data analysis, including Exploratory Data Analysis (EDA), quantitative analysis using technical indicators, and correlation studies between news sentiment and stock movements. The dataset includes financial news articles and stock price data for seven companies (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA).


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
