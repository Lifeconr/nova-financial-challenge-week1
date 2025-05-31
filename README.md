Here's the corrected version of your README with proper Markdown headers (`#`, `##`, etc.) and formatting that will render correctly when viewed on GitHub or any Markdown viewer:

```markdown
# Nova Financial Challenge (Week 1)

## 🧾 Overview

This repository contains the code and analysis for **Nova Financial Challenge – Week 1**. The project focuses on financial data analysis, including:

- Exploratory Data Analysis (EDA)
- Quantitative analysis using technical indicators
- Correlation studies between news sentiment and stock movements

The dataset includes financial news articles and stock price data for seven companies: **AAPL**, **AMZN**, **GOOG**, **META**, **MSFT**, **NVDA**, and **TSLA**.

---

---

## ⚙️ Installation

Install the required libraries for EDA and text analysis:

```bash
pip install pandas matplotlib seaborn scikit-learn nltk textblob
````

For Task 2 (technical analysis and stock data):

```bash
pip install TA-Lib yfinance
```

Download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 📦 Data Setup

The `Data/` directory is **not tracked** in Git due to GitHub’s 100 MB file size limit. You must manually place the datasets in the `Data/` directory.

### Required datasets:

* `raw_analyst_ratings.csv` (311.40 MB)
* `invalid_dates.csv` (273.67 MB)
* `corrected_analyst_ratings.csv` (can be regenerated)

To regenerate `corrected_analyst_ratings.csv`, run:

```bash
python src/preprocess_dates.py
```

---

Let me know if you'd like to add a license, contribution guide, or any badges.

```
```
