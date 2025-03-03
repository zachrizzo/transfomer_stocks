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
import time
import socket
import pandas_datareader as pdr
from yahooquery import Ticker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')
fmp_api_key = os.getenv('FMP_NEWS_API_KEY')

if not alpaca_api_key_id or not alpaca_api_secret_key:
    logger.warning("Alpaca API keys not found in environment variables. News data fetching will not work.")

# Add a function to check internet connectivity
def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if there is an internet connection by trying to connect to Google's DNS.

    Args:
        host: The host to connect to (default is Google's DNS)
        port: The port to connect to (default is 53, DNS port)
        timeout: Connection timeout in seconds

    Returns:
        True if connection successful, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        logger.error(f"Internet connection check failed: {ex}")
        return False

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
    # Check internet connection first
    if not check_internet_connection():
        st.error("No internet connection detected. Please check your network and try again.")
        return pd.DataFrame(), None

    # Ensure dates are properly formatted and validated
    today = datetime.now().date()

    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid start date format: {start_date}. Using 1 year ago.")
            start_date_obj = today - timedelta(days=365)
    else:
        start_date_obj = start_date

    if isinstance(end_date, str):
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Invalid end date format: {end_date}. Using today.")
            end_date_obj = today
    else:
        end_date_obj = end_date

    # Ensure end_date is not in the future
    if end_date_obj > today:
        logger.warning(f"End date {end_date_obj} is in the future. Using today's date instead.")
        end_date_obj = today
        end_date = today.strftime('%Y-%m-%d')

    # Ensure start_date is not in the future and not after end_date
    if start_date_obj > today:
        logger.warning(f"Start date {start_date_obj} is in the future. Using 1 year before today instead.")
        start_date_obj = today - timedelta(days=365)
        start_date = start_date_obj.strftime('%Y-%m-%d')

    if start_date_obj > end_date_obj:
        logger.warning(f"Start date {start_date_obj} is after end date {end_date_obj}. Swapping dates.")
        start_date_obj, end_date_obj = end_date_obj, start_date_obj
        start_date = start_date_obj.strftime('%Y-%m-%d')
        end_date = end_date_obj.strftime('%Y-%m-%d')

    logger.info(f"Loading data for {stockSymbol} from {start_date} to {end_date}")

    # Normalize stock symbol - remove any trailing whitespace or special characters
    stockSymbol = stockSymbol.strip().upper()

    # Set up retries
    max_retries = 3
    retry_delay = 2  # seconds

    # Try different methods to fetch data
    methods = [
        # Method 1: Financial Modeling Prep API (most reliable)
        lambda: fetch_from_fmp(stockSymbol, start_date, end_date),

        # Method 2: Use yf.download with default parameters
        lambda: yf.download(stockSymbol, start=start_date, end=end_date, progress=False),

        # Method 3: Use yf.Ticker.history
        lambda: yf.Ticker(stockSymbol).history(start=start_date, end=end_date),

        # Method 4: Use yf.download with different parameters
        lambda: yf.download(stockSymbol, start=start_date, end=end_date, progress=False, threads=False),

        # Method 5: Try with a shorter date range
        lambda: yf.download(stockSymbol, start=(datetime.now().date() - timedelta(days=365)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), progress=False),

        # Method 6: Try pandas-datareader
        lambda: fetch_from_pandas_datareader(stockSymbol, start_date, end_date),

        # Method 7: Try yahooquery
        lambda: fetch_from_yahooquery(stockSymbol, start_date, end_date)
    ]

    for method_idx, method in enumerate(methods):
        for attempt in range(max_retries):
            try:
                logger.info(f"Trying method {method_idx+1}, attempt {attempt+1} for {stockSymbol}")
                stock_data = method()

                if stock_data is None or stock_data.empty:
                    logger.warning(f"Method {method_idx+1}, attempt {attempt+1} returned empty data for {stockSymbol}")
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                else:
                    # Check if we have enough data points
                    if len(stock_data) < 5:  # Arbitrary threshold, adjust as needed
                        logger.warning(f"Method {method_idx+1} returned only {len(stock_data)} data points for {stockSymbol}, which may not be enough")
                        if attempt < max_retries - 1:
                            logger.warning(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue

                    # Data successfully loaded, process it
                    # Normalize Close prices
                    close_scaler = MinMaxScaler(feature_range=(0, 1))
                    stock_data['Normalized_Close'] = close_scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

                    # Normalize Volume if available
                    if 'Volume' in stock_data.columns:
                        volume_scaler = MinMaxScaler(feature_range=(0, 1))
                        stock_data['Normalized_Volume'] = volume_scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))

                    logger.info(f"Successfully loaded data using method {method_idx+1}: {len(stock_data)} rows")
                    return stock_data, close_scaler

            except Exception as e:
                logger.error(f"Error in method {method_idx+1}, attempt {attempt+1} for {stockSymbol}: {str(e)}")

                if attempt < max_retries - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

    # If we've reached here, all methods and retries have failed
    logger.error(f"Failed to fetch data for {stockSymbol} after trying all methods")
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

# Add fetch_stock_data function for compatibility with live_trading.py
def fetch_stock_data(stockSymbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> pd.DataFrame:
    """
    Fetch stock data for live trading. This is a wrapper around load_data for compatibility.

    Args:
        stockSymbol: The ticker symbol for the stock
        start_date: The start date for the data range
        end_date: The end date for the data range

    Returns:
        DataFrame containing stock data
    """
    try:
        logger.info(f"Fetching stock data for {stockSymbol} from {start_date} to {end_date}")
        stock_data, _ = load_data(stockSymbol, start_date, end_date)

        if stock_data.empty:
            logger.error(f"Failed to fetch data for {stockSymbol}. Please check the symbol and date range.")
            return pd.DataFrame()

        logger.info(f"Successfully fetched {len(stock_data)} rows of data for {stockSymbol}")
        return stock_data
    except Exception as e:
        logger.error(f"Error fetching stock data for {stockSymbol}: {str(e)}")
        return pd.DataFrame()

# Add functions to fetch data from additional sources
def fetch_from_pandas_datareader(symbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> pd.DataFrame:
    """
    Fetch stock data using pandas-datareader as another fallback option.

    Args:
        symbol: The ticker symbol for the stock
        start_date: The start date for the data range
        end_date: The end date for the data range

    Returns:
        DataFrame containing stock data
    """
    try:
        logger.info(f"Fetching data from pandas-datareader for {symbol}")
        df = pdr.data.get_data_yahoo(symbol, start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found in pandas-datareader for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Error fetching data from pandas-datareader: {str(e)}")
        return pd.DataFrame()

def fetch_from_yahooquery(symbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> pd.DataFrame:
    """
    Fetch stock data using yahooquery as another fallback option.

    Args:
        symbol: The ticker symbol for the stock
        start_date: The start date for the data range
        end_date: The end date for the data range

    Returns:
        DataFrame containing stock data
    """
    try:
        # Convert dates to string format if they're date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')

        logger.info(f"Fetching data from yahooquery for {symbol}")
        ticker = Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        # If df is multi-index with symbol as first level, get just the data for our symbol
        if isinstance(df.index, pd.MultiIndex):
            if symbol in df.index.get_level_values(0):
                df = df.loc[symbol]
            else:
                logger.warning(f"Symbol {symbol} not found in yahooquery results")
                return pd.DataFrame()

        if df.empty:
            logger.warning(f"No data found in yahooquery for {symbol}")

        return df

    except Exception as e:
        logger.error(f"Error fetching data from yahooquery: {str(e)}")
        return pd.DataFrame()

def fetch_from_fmp(symbol: str, start_date: Union[str, date], end_date: Union[str, date]) -> pd.DataFrame:
    """
    Fetch stock data from Financial Modeling Prep API.

    Args:
        symbol: The ticker symbol for the stock
        start_date: The start date for the data range
        end_date: The end date for the data range

    Returns:
        DataFrame containing stock data
    """
    if not fmp_api_key:
        logger.warning("FMP API key not found in environment variables")
        return pd.DataFrame()

    try:
        # Convert dates to string format if they're date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')

        # Fetch historical daily data
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={fmp_api_key}"
        logger.info(f"Fetching data from Financial Modeling Prep for {symbol}")

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            logger.error(f"FMP API error: {data['Error Message']}")
            return pd.DataFrame()

        if "historical" not in data:
            logger.error(f"Unexpected FMP API response format: {data.keys()}")
            return pd.DataFrame()

        # Convert to DataFrame
        historical_data = data["historical"]
        df = pd.DataFrame(historical_data)

        # Rename columns to match Yahoo Finance format
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjClose': 'Adj Close',
            'volume': 'Volume'
        })

        # Convert date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        # Sort by date (oldest to newest)
        df = df.sort_index()

        if df.empty:
            logger.warning(f"No data found in FMP for {symbol} between {start_date} and {end_date}")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from FMP: {str(e)}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"Error parsing FMP data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error with FMP: {str(e)}")
        return pd.DataFrame()
