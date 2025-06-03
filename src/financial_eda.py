import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from wordcloud import WordCloud

class FinancialEDA:
    """
    A class to perform Exploratory Data Analysis on financial news and stock data.

    Attributes:
        news_df (pd.DataFrame): The financial news dataset.
        stock_data (dict): Dictionary of stock price dataframes.
    """

    def __init__(self, news_path, stock_dir):
        """
        Initialize the FinancialEDA class with news and stock data.

        Args:
            news_path (str): Path to the corrected news dataset CSV file.
            stock_dir (str): Directory containing stock price CSV files.
        """
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4') 
        # Load news and stock data
        self.news_df = pd.read_csv(news_path)
        self.stock_data = {}
        stocks = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
        for stock in stocks:
            file_path = os.path.join(stock_dir, f"{stock}_historical_data.csv")
            if os.path.exists(file_path):
                self.stock_data[stock] = pd.read_csv(file_path)
                self.stock_data[stock]['Date'] = pd.to_datetime(self.stock_data[stock]['Date'])

        # Ensure dates are in datetime format
        if 'date' in self.news_df.columns:
            self.news_df['date'] = pd.to_datetime(self.news_df['date'])
            self.news_df['date_only'] = self.news_df['date'].dt.date
        else:
            raise ValueError("Column 'date' not found in the dataset. Check the corrected file.")

    def descriptive_statistics(self):
        """
        Compute descriptive statistics for headline lengths and publisher activity.

        Returns:
            dict: Statistics including headline length summary and publisher counts.
        """
        if self.news_df.empty:
            raise ValueError("News DataFrame is empty. Check the input dataset.")
        self.news_df['headline_length'] = self.news_df['headline'].str.len()
        length_stats = self.news_df['headline_length'].describe()
        publisher_counts = self.news_df['publisher'].value_counts()
        
        # Visualize Headline Length
        plt.figure(figsize=(8, 5))
        sns.histplot(self.news_df['headline_length'], bins=30, kde=True)
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length (Characters)')
        plt.ylabel('Frequency')
        plt.show()
        
        # Visualize Top Publishers
        plt.figure(figsize=(10, 6))
        publisher_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Publishers by Article Count')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()
        
        return {'length_stats': length_stats, 'publisher_counts': publisher_counts}

    def text_analysis(self):
        """
        Perform text analysis including topic modeling and keyword extraction.

        Returns:
            dict: LDA topics and word cloud data.
        """
        if self.news_df.empty:
            raise ValueError("News DataFrame is empty. Check the input dataset.")
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        def preprocess_text(text):
            """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
            tokens = word_tokenize(text.lower())
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
            return tokens
        
        self.news_df['tokens'] = self.news_df['headline'].apply(preprocess_text)
        
        # Word Cloud
        all_tokens = [token for tokens in self.news_df['tokens'] for token in tokens]
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Headline Keywords')
        plt.show()
        
        # Topic Modeling with LDA
        dictionary = corpora.Dictionary(self.news_df['tokens'])
        corpus = [dictionary.doc2bow(tokens) for tokens in self.news_df['tokens']]
        lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
        topics = {f'Topic {idx}': topic for idx, topic in lda_model.print_topics(-1)}
        return {'topics': topics}

    def time_series_analysis(self):
        """
        Analyze publication frequency and hourly patterns over time.

        Returns:
            dict: Daily and hourly publication counts.
        """
        if self.news_df.empty:
            raise ValueError("News DataFrame is empty. Check the input dataset.")
        daily_counts = self.news_df.groupby('date_only').size()
        self.news_df['hour'] = self.news_df['date'].dt.hour
        hourly_counts = self.news_df.groupby('hour').size()
        
        # Plot Daily Frequency
        plt.figure(figsize=(12, 6))
        daily_counts.plot()
        plt.title('Daily Article Publication Frequency')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot Hourly Frequency
        plt.figure(figsize=(10, 5))
        hourly_counts.plot(kind='bar')
        plt.title('Article Publication Frequency by Hour (UTC)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        plt.show()
        
        return {'daily_counts': daily_counts, 'hourly_counts': hourly_counts}

    def publisher_analysis(self):
        """
        Analyze publisher contributions and news types.

        Returns:
            pd.DataFrame: Publisher vs. news type distribution.
        """
        if self.news_df.empty:
            raise ValueError("News DataFrame is empty. Check the input dataset.")
        def categorize_headline(headline):
            """Categorize headlines into types based on keywords."""
            headline = headline.lower()
            if 'earnings' in headline:
                return 'Earnings'
            elif 'price target' in headline:
                return 'Price Target'
            elif 'fda' in headline or 'approval' in headline:
                return 'Regulatory'
            return 'Other'
        
        self.news_df['news_type'] = self.news_df['headline'].apply(categorize_headline)
        publisher_news_type = self.news_df.groupby(['publisher', 'news_type']).size().unstack().fillna(0)
        
        # Visualize Top Publishers by News Type
        top_publishers = self.news_df['publisher'].value_counts().head(5).index
        top_publisher_data = publisher_news_type.loc[top_publishers]
        plt.figure(figsize=(10, 6))
        top_publisher_data.plot(kind='bar', stacked=True)
        plt.title('News Types by Top Publishers')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()
        
        # Extract Domains
        def extract_domain(publisher):
            """Extract domain from publisher email if applicable."""
            if '@' in str(publisher):
                return publisher.split('@')[1]
            return None
        self.news_df['publisher_domain'] = self.news_df['publisher'].apply(extract_domain)
        domain_counts = self.news_df['publisher_domain'].value_counts()
        
        return {'publisher_news_type': publisher_news_type, 'domain_counts': domain_counts}