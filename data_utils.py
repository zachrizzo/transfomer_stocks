"""
Data Utilities Module for Stock Prediction Application.

This module provides functions for loading, preprocessing, and managing
stock price and news data for the stock price prediction model.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests
from textblob import TextBlob
import os
import logging
import dotenv
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime, date, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')

if not alpaca_api_key_id or not alpaca_api_secret_key:
    logger.warning("Alpaca API keys not found in environment variables. News data fetching will not work.")

@st.cache_resource
def load_data(stockSymbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> Tuple[pd.DataFrame, Optional[MinMaxScaler]]:
    """
    Load stock data from Yahoo Finance and normalize it.

    Args:
        stockSymbol: The ticker symbol for the stock
        start_date: The start date for the data range
        end_date: The end date for the data range

    Returns:
        A tuple containing the stock data DataFrame and the close price scaler
    """
    # Ensure end_date is not in the future
    today = datetime.now().date()
    if isinstance(end_date, str):
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        if end_date_obj > today:
            end_date = today.strftime('%Y-%m-%d')
    elif isinstance(end_date, date) and end_date > today:
        end_date = today

    logger.info(f"Loading data for {stockSymbol} from {start_date} to {end_date}")

    try:
        # Try to download with progress=False to avoid potential issues
        stock_data = yf.download(stockSymbol, start=start_date, end=end_date, progress=False)

        if stock_data.empty:
            # Try with a shorter date range as a fallback
            fallback_start = datetime.now().date() - timedelta(days=365)
            logger.warning(f"No data found for {stockSymbol} from {start_date} to {end_date}. Trying with shorter range: {fallback_start} to {today}")
            st.warning(f"No data found for {stockSymbol} from {start_date} to {end_date}. Trying with shorter range: {fallback_start} to {today}")

            stock_data = yf.download(stockSymbol, start=fallback_start, end=today, progress=False)

            if stock_data.empty:
                st.error(f"No data found for symbol {stockSymbol}. Please check the symbol and try again.")
                return pd.DataFrame(), None

        # Normalize Close prices
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Normalized_Close'] = close_scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        # Normalize Volume if available
        if 'Volume' in stock_data.columns:
            volume_scaler = MinMaxScaler(feature_range=(0, 1))
            stock_data['Normalized_Volume'] = volume_scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))

        logger.info(f"Successfully loaded data: {len(stock_data)} rows")
        return stock_data, close_scaler

    except Exception as e:
        st.error(f"Error loading data for {stockSymbol}: {str(e)}")
        logger.error(f"Error loading data: {str(e)}", exc_info=True)

        # Try with a different API method as a last resort
        try:
            logger.info(f"Attempting to fetch {stockSymbol} data using Ticker object")
            ticker = yf.Ticker(stockSymbol)
            stock_data = ticker.history(start=start_date, end=end_date)

            if not stock_data.empty:
                # Normalize Close prices
                close_scaler = MinMaxScaler(feature_range=(0, 1))
                stock_data['Normalized_Close'] = close_scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

                # Normalize Volume if available
                if 'Volume' in stock_data.columns:
                    volume_scaler = MinMaxScaler(feature_range=(0, 1))
                    stock_data['Normalized_Volume'] = volume_scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))

                logger.info(f"Successfully loaded data using Ticker object: {len(stock_data)} rows")
                return stock_data, close_scaler
        except Exception as ticker_error:
            logger.error(f"Error using Ticker object: {str(ticker_error)}", exc_info=True)

        return pd.DataFrame(), None

def preprocess_data(
    stock_data: pd.DataFrame,
    news_data: pd.DataFrame,
    selected_indicators: List[Dict[str, Any]],
    close_scaler: MinMaxScaler,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess stock and news data for model training.

    Args:
        stock_data: DataFrame containing stock price data
        news_data: DataFrame containing news data
        selected_indicators: List of selected technical indicators
        close_scaler: Scaler used to normalize close prices
        seq_length: Length of sequence for time series data

    Returns:
        A tuple containing input features (X) and target values (y) as numpy arrays
    """
    features = []

    # Always include normalized close price
    features.append(stock_data['Normalized_Close'].values.reshape(-1, 1))

    # Process each selected indicator
    for indicator in selected_indicators:
        if indicator['key'] == 'volume':
            if 'Normalized_Volume' in stock_data.columns:
                features.append(stock_data['Normalized_Volume'].values.reshape(-1, 1))
            else:
                logger.warning("Volume data requested but not available in stock data")

        elif indicator['key'] == 'news':
            if not news_data.empty:
                try:
                    # Ensure news_data's index is sorted and monotonic
                    news_data = news_data.sort_index()

                    # Use stock_data.index.normalize() to get dates as DatetimeIndex
                    stock_dates = stock_data.index.normalize()

                    # Reindex news_data to stock_dates
                    news_data = news_data.reindex(stock_dates, method='ffill').fillna(0)

                    sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
                    normalized_sentiment = sentiment_scaler.fit_transform(news_data['sentiment'].values.reshape(-1, 1))
                    features.append(normalized_sentiment)

                except Exception as e:
                    logger.error(f"Error processing news data: {str(e)}", exc_info=True)
                    st.warning(f"Error processing news data: {str(e)}. News data will be ignored.")
            else:
                logger.warning("News data requested but the dataset is empty")
                st.warning("News data is empty. The model will proceed without news features.")
        else:
            # Get the columns for this indicator
            cols = indicator.get('columns', [])
            if cols:
                for col in cols:
                    if col in stock_data.columns:
                        try:
                            scaler = MinMaxScaler(feature_range=(0, 1))
                            normalized_col = scaler.fit_transform(stock_data[col].values.reshape(-1, 1))
                            features.append(normalized_col)
                        except Exception as e:
                            logger.error(f"Error normalizing column {col}: {str(e)}", exc_info=True)
                            st.warning(f"Error processing indicator {indicator['name']} column {col}. This feature will be ignored.")
                    else:
                        logger.warning(f"No column '{col}' found for indicator {indicator['name']}")
                        st.warning(f"No column '{col}' found for indicator {indicator['name']}.")
            else:
                logger.warning(f"No columns specified for indicator {indicator['name']}")
                st.warning(f"No columns specified for indicator {indicator['name']}.")

    # Combine all features
    if features:
        try:
            data_combined = np.hstack(features)
            X, _ = create_sequences(data_combined, seq_length)

            # Set y to be the next 'Normalized_Close' value
            y = stock_data['Normalized_Close'].values[seq_length:].reshape(-1, 1)

            # Remove sequences with any NaNs
            valid_indices = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
            X = X[valid_indices]
            y = y[valid_indices]

            logger.info(f"Preprocessed data: X shape {X.shape}, y shape {y.shape}")

            if X.size == 0 or y.size == 0:
                logger.warning("No valid data after preprocessing")
                st.warning("No valid data after preprocessing. Try different indicators or date ranges.")

            return X, y

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}", exc_info=True)
            st.error(f"Error in data preprocessing: {str(e)}")
            return np.array([]), np.array([])
    else:
        # Return empty arrays if no features are selected
        logger.warning("No features selected for preprocessing")
        return np.array([]), np.array([])

@st.cache_resource
def fetch_news_batch(stockSymbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> pd.DataFrame:
    """
    Fetch news data for a stock symbol within a date range from Alpaca API.

    Args:
        stockSymbol: The ticker symbol for the stock
        start_date: Start date for news articles
        end_date: End date for news articles

    Returns:
        DataFrame containing news articles with sentiment analysis
    """
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
        logger.info(f"Successfully fetched news data: {len(news_data.get('news', []))} articles")

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching news: {http_err}", exc_info=True)
        st.error(f"HTTP error occurred while fetching news: {http_err}")
        return pd.DataFrame()

    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching news: {err}", exc_info=True)
        st.error(f"Error fetching news: {err}")
        return pd.DataFrame()

    except ValueError:
        logger.error("Error parsing the JSON response for news data", exc_info=True)
        st.error("Error parsing the JSON response for news data.")
        return pd.DataFrame()

    articles = []
    for item in news_data.get('news', []):
        headline = item.get('headline', '')
        link = item.get('url', '')
        published_at = item.get('created_at', '')

        # Perform sentiment analysis on headline
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
def fetch_all_news(stockSymbol: str, stock_data_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch all news articles for a stock across the given dates.

    Args:
        stockSymbol: The ticker symbol for the stock
        stock_data_dates: DatetimeIndex containing dates to fetch news for

    Returns:
        DataFrame containing all news articles with sentiment analysis
    """
    all_news = pd.DataFrame()

    try:
        stock_data_dates = pd.to_datetime(stock_data_dates).date
        unique_dates = sorted(set(stock_data_dates))  # Ensure dates are unique and sorted

        logger.info(f"Fetching news for {stockSymbol} across {len(unique_dates)} dates")

        # Fetch news in batches to avoid hitting API limits
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

            logger.info(f"Successfully fetched {len(all_news)} news articles")
        else:
            logger.warning(f"No news articles found for {stockSymbol}")

    except Exception as e:
        logger.error(f"Error fetching all news: {str(e)}", exc_info=True)
        st.error(f"Error fetching news data: {str(e)}")

    return all_news

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and target values for time series prediction.

    Args:
        data: Input data as a numpy array
        seq_length: Length of each sequence

    Returns:
        A tuple containing sequences (X) and corresponding next values (y)
    """
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Normalize data using mean and standard deviation.

    Args:
        data: Input data to normalize
        mean: Mean value for normalization
        std: Standard deviation for normalization

    Returns:
        Normalized data
    """
    return (data - mean) / std

def denormalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Denormalize data using mean and standard deviation.

    Args:
        data: Normalized data
        mean: Mean value used in normalization
        std: Standard deviation used in normalization

    Returns:
        Denormalized data
    """
    return data * std + mean
