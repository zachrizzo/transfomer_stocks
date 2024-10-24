# data_utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
from textblob import TextBlob#import the environment variables
import os
import dotenv
dotenv.load_dotenv()

alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')



@st.cache_resource
def load_data(stockSymbol, start_date, end_date):
    stock_data = yf.download(stockSymbol, start=start_date, end=end_date)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['Normalized_Close'] = close_scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    if 'Volume' in stock_data.columns:
        volume_scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Normalized_Volume'] = volume_scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))

    return stock_data, close_scaler

def preprocess_data(stock_data, news_data, list_of_indicators, close_scaler, seq_length):
    features = []

    # Always include normalized close price
    features.append(stock_data['Normalized_Close'].values.reshape(-1, 1))

    # Process each selected indicator
    for indicator in list_of_indicators:
        if indicator['selected']:
            if indicator['key'] == 'volume':
                if 'Normalized_Volume' in stock_data.columns:
                    features.append(stock_data['Normalized_Volume'].values.reshape(-1, 1))
            elif indicator['key'] == 'news':
                if not news_data.empty:
                    news_data = news_data.reindex(stock_data.index, fill_value=0)
                    sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
                    normalized_sentiment = sentiment_scaler.fit_transform(news_data['sentiment'].values.reshape(-1, 1))
                    features.append(normalized_sentiment)
            else:
                if indicator['name'] in stock_data.columns:
                    indicator_scaler = MinMaxScaler(feature_range=(0, 1))
                    normalized_indicator = indicator_scaler.fit_transform(stock_data[indicator['name']].values.reshape(-1, 1))
                    features.append(normalized_indicator)

    # Combine all features
    if features:
        data_combined = np.hstack(features)
        X, y = create_sequences(data_combined, seq_length)
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
        st.error(f"HTTP error occurred: {http_err}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as err:
        st.error(f"Error fetching news: {err}")
        return pd.DataFrame()
    except ValueError:
        st.error("Error parsing the JSON response")
        return pd.DataFrame()

    articles = []
    for item in news_data.get('news', []):
        articles.append({
            'title': item['headline'],
            'link': item['url'],
            'publishedAt': item['created_at'],
            'sentiment': TextBlob(item['headline']).sentiment.polarity
        })
    news_df = pd.DataFrame(articles)
    if not news_df.empty:
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
    return news_df

@st.cache_resource
def fetch_all_news(stockSymbol, stock_data_dates):
    all_news = pd.DataFrame()
    stock_data_dates = pd.to_datetime(stock_data_dates).date
    unique_dates = sorted(set(stock_data_dates))  # Ensure dates are unique and sorted

    for i in range(0, len(unique_dates), 50):
        start_date = unique_dates[i]
        end_date = unique_dates[min(i + 49, len(unique_dates) - 1)]
        batch_news = fetch_news_batch(stockSymbol, start_date, end_date)
        all_news = pd.concat([all_news, batch_news])

    # Ensure the index is a DatetimeIndex and remove any duplicates
    all_news.index = pd.to_datetime(all_news['publishedAt']).dt.date
    all_news = all_news[~all_news.index.duplicated(keep='first')]

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


