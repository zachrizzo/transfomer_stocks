import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
from textblob import TextBlob
import os
import dotenv

dotenv.load_dotenv()
alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')

@st.cache_resource
def load_data(stockSymbol, start_date, end_date):
    stock_data = yf.download(stockSymbol, start=start_date, end=end_date)
    if stock_data.empty:
        st.error(f"No data found for symbol {stockSymbol}. Please check the symbol and try again.")
        return pd.DataFrame(), None

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['Normalized_Close'] = close_scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    if 'Volume' in stock_data.columns:
        volume_scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Normalized_Volume'] = volume_scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))

    return stock_data, close_scaler

def preprocess_data(stock_data, news_data, selected_indicators, close_scaler, seq_length):
    features = []

    # Always include normalized close price
    features.append(stock_data['Normalized_Close'].values.reshape(-1, 1))

    # Process each selected indicator
    for indicator in selected_indicators:
        if indicator['key'] == 'volume':
            if 'Normalized_Volume' in stock_data.columns:
                features.append(stock_data['Normalized_Volume'].values.reshape(-1, 1))
        elif indicator['key'] == 'news':
            if not news_data.empty:
                # Ensure news_data's index is sorted and monotonic
                news_data = news_data.sort_index()
                # Use stock_data.index.normalize() to get dates as DatetimeIndex
                stock_dates = stock_data.index.normalize()
                # Reindex news_data to stock_dates
                news_data = news_data.reindex(stock_dates, method='ffill').fillna(0)
                sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
                normalized_sentiment = sentiment_scaler.fit_transform(news_data['sentiment'].values.reshape(-1, 1))
                features.append(normalized_sentiment)
            else:
                st.warning("News data is empty.")
        else:
            # Get the columns for this indicator
            cols = indicator.get('columns', [])
            if cols:
                for col in cols:
                    if col in stock_data.columns:
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        normalized_col = scaler.fit_transform(stock_data[col].values.reshape(-1, 1))
                        features.append(normalized_col)
                    else:
                        st.warning(f"No column '{col}' found for indicator {indicator['name']}.")
            else:
                st.warning(f"No columns specified for indicator {indicator['name']}.")

    # Combine all features
    if features:
        data_combined = np.hstack(features)
        X, _ = create_sequences(data_combined, seq_length)

        # Set y to be the next 'Normalized_Close' value
        y = stock_data['Normalized_Close'].values[seq_length:].reshape(-1, 1)

        # Remove sequences with any NaNs
        valid_indices = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]

        return X, y
    else:
        # Return empty arrays if no features are selected
        return np.array([]), np.array([])

@st.cache_resource
def fetch_news_batch(stockSymbol, start_date, end_date):
    url = f"https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&symbols={stockSymbol}&sort=desc&limit=50"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_api_key_id,
        "APCA-API-SECRET-KEY": alpaca_api_secret_key
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while fetching news: {http_err}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as err:
        st.error(f"Error fetching news: {err}")
        return pd.DataFrame()
    except ValueError:
        st.error("Error parsing the JSON response for news data.")
        return pd.DataFrame()

    articles = []
    for item in news_data.get('news', []):
        headline = item.get('headline', '')
        link = item.get('url', '')
        published_at = item.get('created_at', '')
        sentiment = TextBlob(headline).sentiment.polarity
        articles.append({
            'title': headline,
            'link': link,
            'publishedAt': published_at,
            'sentiment': sentiment
        })
    news_df = pd.DataFrame(articles)
    if not news_df.empty:
        # Convert 'publishedAt' to datetime with UTC, then make it timezone-naive
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'], utc=True).dt.tz_convert(None)
    return news_df

@st.cache_resource
def fetch_all_news(stockSymbol, _stock_data_dates):
    all_news = pd.DataFrame()
    stock_data_dates = pd.to_datetime(_stock_data_dates).date
    unique_dates = sorted(set(stock_data_dates))  # Ensure dates are unique and sorted

    for i in range(0, len(unique_dates), 50):
        start_date = unique_dates[i]
        end_date = unique_dates[min(i + 49, len(unique_dates) - 1)]
        batch_news = fetch_news_batch(stockSymbol, start_date, end_date)
        all_news = pd.concat([all_news, batch_news], ignore_index=True)

    # Ensure the index is a DatetimeIndex and remove any duplicates
    if not all_news.empty:
        # After making 'publishedAt' timezone-naive, normalize the datetime
        all_news.index = all_news['publishedAt'].dt.normalize()
        all_news = all_news[~all_news.index.duplicated(keep='first')]
        all_news = all_news.sort_index()

    return all_news

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Normalize data function
def normalize(data, mean, std):
    return (data - mean) / std

# Denormalize data function
def denormalize(data, mean, std):
    return data * std + mean
