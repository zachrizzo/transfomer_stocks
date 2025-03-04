"""
Stock Price Prediction with News Sentiment Analysis Application.

A Streamlit application that combines stock price data with news sentiment analysis
to train a Transformer model for stock price prediction.
"""

import os
import copy
import logging
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import dotenv
from typing import List, Dict, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import random

# Import custom modules
from model import TransformerModel, train_model, predict_future, count_parameters
from data_utils import load_data, preprocess_data, fetch_all_news, create_sequences, normalize, denormalize
from config import get_indicator_by_key, list_of_indicators as original_list_of_indicators

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = 'models'


def setup_environment():
    """Set up the application environment and variables."""
    # Load environment variables
    dotenv.load_dotenv()

    # Set the API keys
    alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
    alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')

    # Check if API keys are set
    if not alpaca_api_key_id or not alpaca_api_secret_key:
        logger.warning("Alpaca API keys are missing. News data functionality will be limited.")
        st.sidebar.warning("Alpaca API keys not found. News data fetching may not work.")

    # Set page title and configuration
    st.set_page_config(
        page_title="Stock Prediction with Transformers",
        layout="wide"
    )

    st.title("Stock Price Prediction with Transformer Models")

    # Initialize session state for stock symbol if not already set
    if 'stockSymbol' not in st.session_state:
        st.session_state['stockSymbol'] = 'AAPL'

    # Initialize session state for stop training flag
    if 'stop_training' not in st.session_state:
        st.session_state['stop_training'] = False

    # Initialize session state for trained model and parameters
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None

    if 'model_state_dict' not in st.session_state:
        st.session_state['model_state_dict'] = None

    # Initialize session state for training status
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False

    # Initialize session state for backtesting data
    if 'backtest_chart_data' not in st.session_state:
        st.session_state['backtest_chart_data'] = None

    if 'backtest_data' not in st.session_state:
        st.session_state['backtest_data'] = None

    # Initialize session state for backtesting parameters
    if 'backtest_strategy_type' not in st.session_state:
        st.session_state['backtest_strategy_type'] = "Trend Following"

    if 'backtest_params' not in st.session_state:
        st.session_state['backtest_params'] = {}

    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def create_sidebar_controls():
    """Create and display the sidebar controls for model parameters."""
    st.sidebar.header("Model Parameters")

    # Model architecture parameters
    st.sidebar.subheader("Architecture")
    hidden_dim = st.sidebar.slider("Hidden Size", min_value=32, max_value=512, value=128, step=32)
    num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=6, value=2, step=1)
    num_heads = st.sidebar.slider("Number of Heads", min_value=1, max_value=16, value=8, step=1)
    dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

    # Sequence length parameter
    seq_length = st.sidebar.slider("Sequence Length", min_value=5, max_value=60, value=20, step=1,
                                  help="Number of time steps to use for each training sequence")

    # Training parameters
    st.sidebar.subheader("Training")
    num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=1000, value=50, step=1)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=32, step=16)
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f"
    )

    # Training control buttons
    if st.sidebar.button("Stop Training"):
        st.session_state['stop_training'] = True
        logger.info("User requested to stop training")

    return {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'seq_length': seq_length,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }


def get_stock_inputs():
    """Get user inputs for stock symbol and date range."""
    st.title("Stock Price Prediction with Transformer Model")

    # Input for stock symbol
    stockSymbol = st.text_input("Enter Stock Symbol", st.session_state['stockSymbol'])

    # Create tabs for different date selection modes
    date_tabs = st.tabs(["Simple Date Selection", "Advanced Date Selection"])

    with date_tabs[0]:
        # Simple date selection (original method)
        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start Date", datetime(2020, 1, 1))

        with col2:
            end_date = st.date_input("End Date", datetime.now())

        # Add a separator for training end date (which is also the backtest start date)
        st.info("To prevent data leakage, separate your data into training and backtesting periods.")

        # Calculate a default training end date (60% of the data)
        default_training_end = start_date + timedelta(days=int((end_date - start_date).days * 0.6))

        training_end_date = st.date_input(
            "Training End Date (Backtest Start Date)",
            default_training_end,
            help="Data before this date will be used for training, data after will be used for backtesting."
        )

        # In simple mode, the backtest dates are derived from the main date range
        backtest_start_date = training_end_date
        backtest_end_date = end_date

        # Validate inputs for simple mode
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return None, None, None, None, None, None

        if start_date >= training_end_date or training_end_date >= end_date:
            st.error("Training end date must be between start date and end date")
            return None, None, None, None, None, None

    with date_tabs[1]:
        # Advanced date selection with separate training and backtesting ranges
        st.subheader("Training Data Range")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Training Start Date", datetime(2020, 1, 1), key="train_start")
        with col2:
            training_end_date = st.date_input("Training End Date",
                                            datetime.now() - timedelta(days=90),
                                            key="train_end")

        st.subheader("Backtesting Data Range")
        st.info("Select a separate date range for backtesting. This can be historical data not used for training.")

        col1, col2 = st.columns(2)
        with col1:
            backtest_start_date = st.date_input("Backtesting Start Date",
                                               training_end_date,
                                               key="backtest_start")
        with col2:
            backtest_end_date = st.date_input("Backtesting End Date",
                                             datetime.now(),
                                             key="backtest_end")

        # Use the training end date for the regular end_date to ensure we load all necessary data
        end_date = max(training_end_date, backtest_end_date)

        # Validate inputs for advanced mode
        if start_date >= training_end_date:
            st.error("Training start date must be before training end date")
            return None, None, None, None, None, None

        if backtest_start_date >= backtest_end_date:
            st.error("Backtesting start date must be before backtesting end date")
            return None, None, None, None, None, None

    # Common validation
    if not stockSymbol:
        st.error("Please enter a stock symbol")
        return None, None, None, None, None, None

    return stockSymbol, start_date, training_end_date, end_date, backtest_start_date, backtest_end_date


def select_indicators():
    """Allow users to select technical indicators to use in the model."""
    st.subheader("Select Technical Indicators")

    # Make a deep copy to avoid modifying the original
    list_of_indicators = copy.deepcopy(original_list_of_indicators)

    # Create columns for a more compact layout
    num_columns = 3
    indicator_columns = st.columns(num_columns)

    selected_indicators = []

    # Distribute indicators across columns
    for i, indicator in enumerate(list_of_indicators):
        col_index = i % num_columns
        with indicator_columns[col_index]:
            selected = st.checkbox(
                indicator['name'],
                key=f"checkbox_{indicator['name']}",
                help=f"Include {indicator['name']} as a feature"
            )
            if selected:
                selected_indicators.append(indicator)

    # Check if any indicators were selected
    if not selected_indicators:
        st.warning("Please select at least one indicator to proceed with training")

    # Determine if volume and news are selected
    use_volume = any(indicator['key'] == 'volume' for indicator in selected_indicators)
    use_news = any(indicator['key'] == 'news' for indicator in selected_indicators)

    return selected_indicators, use_volume, use_news


def prepare_training_data(
    stock_data: pd.DataFrame,
    news_data: pd.DataFrame,
    selected_indicators: List[Dict[str, Any]],
    use_volume: bool,
    use_news: bool,
    close_scaler: MinMaxScaler,
    seq_length: int = 20,
    training_end_idx: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the training data by selecting features and creating sequences.

    Args:
        stock_data: DataFrame with stock data
        news_data: DataFrame with news data
        selected_indicators: List of selected technical indicators
        use_volume: Whether to use volume data
        use_news: Whether to use news sentiment data
        close_scaler: Scaler for close prices
        seq_length: Length of sequences to create
        training_end_idx: Index to split training and testing data

    Returns:
        Tuple of X and y data as numpy arrays
    """
    # Select features to use
    features = []

    # Always include normalized close price
    features.append('Normalized_Close')

    # Add volume if selected
    if use_volume and 'Normalized_Volume' in stock_data.columns:
        features.append('Normalized_Volume')

    # Add selected technical indicators
    for indicator in selected_indicators:
        if indicator['key'] not in ['volume', 'news']:  # These are handled separately
            if indicator['column_name'] in stock_data.columns:
                features.append(indicator['column_name'])

    # Create a DataFrame with only the selected features
    selected_features_df = stock_data[features].copy()

    # Add news sentiment data if selected
    if use_news and not news_data.empty:
        # Process news data to match stock_data's index
        sentiment_df = preprocess_news_data(news_data, stock_data.index)

        # Add sentiment scores to features DataFrame
        if not sentiment_df.empty:
            # Add sentiment features
            selected_features_df = pd.concat([selected_features_df, sentiment_df], axis=1)
            features.extend(['sentiment_score', 'weighted_sentiment'])

    # Drop rows with NaN values
    selected_features_df = selected_features_df.dropna()

    if selected_features_df.empty:
        st.error("No data left after dropping NaN values. Check your indicators and date range.")
        return None, None

    # Create sequences for training
    X, y = create_sequences(selected_features_df.values, seq_length)

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.error("Failed to create sequences. Not enough data points after preprocessing.")
        return None, None

    # Critical fix: Transform the target to only use the first feature (close price) for prediction
    # This ensures consistency with the model's output dimension, which is set to 1
    y = y[:, 0:1]  # Keep only the first column (Normalized_Close)

    # Log the shapes to help with debugging
    st.info(f"Prepared data shapes: X: {X.shape}, y: {y.shape}")

    # Create PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor


def preprocess_news_data(news_data: pd.DataFrame, stock_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Preprocess news data to align with stock data.

    Args:
        news_data: DataFrame with news data
        stock_index: DatetimeIndex from stock data

    Returns:
        DataFrame with preprocessed news sentiment data
    """
    if news_data.empty:
        return pd.DataFrame()

    # Convert timestamps to dates
    news_data['date'] = pd.to_datetime(news_data['timestamp']).dt.date

    # Group by date and calculate daily sentiment statistics
    daily_sentiment = news_data.groupby('date').agg({
        'sentiment_score': ['mean', 'count']
    })

    # Flatten the multi-index columns
    daily_sentiment.columns = ['sentiment_score', 'news_count']

    # Reset index to get date as a column
    daily_sentiment = daily_sentiment.reset_index()

    # Convert date back to datetime for merging
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.set_index('date')

    # Create a weighted sentiment score (weight by news count)
    daily_sentiment['weighted_sentiment'] = daily_sentiment['sentiment_score'] * daily_sentiment['news_count']

    # Reindex to match stock data
    sentiment_df = daily_sentiment.reindex(stock_index, method='ffill')

    # Fill any remaining NaNs with 0 (no news = neutral sentiment)
    sentiment_df = sentiment_df.fillna(0)

    return sentiment_df[['sentiment_score', 'weighted_sentiment']]


def display_model_information(model, num_parameters):
    """Display information about the model architecture."""
    st.subheader("Model Information")

    # Create columns for model information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Architecture**")
        st.code(str(model))

    with col2:
        st.markdown("**Model Summary**")
        st.markdown(f"**Total Parameters:** {num_parameters:,}")
        st.markdown(f"**Input Dimension:** {model.input_dim}")
        st.markdown(f"**Hidden Dimension:** {model.hidden_dim}")
        st.markdown(f"**Output Dimension:** {model.output_dim}")
        st.markdown(f"**Layers:** {len(model.transformer_encoder.layers)}")
        st.markdown(f"**Attention Heads:** {model.num_heads}")
        st.markdown(f"**Sequence Length:** {model._modules['transformer_encoder'].layers[0].self_attn.embed_dim // model.num_heads} (per head)")


def evaluate_model(model, test_X, test_y, close_scaler, device):
    """
    Evaluate the model on test data and return predictions.

    Args:
        model: Trained model
        test_X: Test input data
        test_y: Test target data
        close_scaler: Scaler for close prices
        device: Device to run evaluation on

    Returns:
        DataFrame with actual and predicted values
    """
    # Move data to device
    test_X = test_X.to(device)
    test_y = test_y.to(device)

    # Set model to evaluation mode
    model.eval()

    # Verify inputs have valid shapes
    if test_X.shape[0] == 0 or test_y.shape[0] == 0:
        raise ValueError("Empty input or target arrays.")

    if test_X.shape[0] != test_y.shape[0]:
        raise ValueError(f"Input shape {test_X.shape} doesn't match target shape {test_y.shape}.")

    # Log the shape information
    st.info(f"Input shape: {test_X.shape}, Target shape: {test_y.shape}")

    # Make predictions
    with torch.no_grad():
        predictions = model(test_X)

    # Verify prediction shape matches target shape's first dimension
    if predictions.shape[0] != test_y.shape[0]:
        st.warning(f"Prediction shape {predictions.shape} doesn't match target shape {test_y.shape}.")
        # Adjust the predictions shape to match test_y if needed
        if predictions.shape[0] < test_y.shape[0]:
            st.warning(f"Padding predictions to match target shape.")
            padding = test_y.shape[0] - predictions.shape[0]
            pad_values = torch.zeros(padding, predictions.shape[1], device=device)
            predictions = torch.cat([predictions, pad_values], dim=0)
        elif predictions.shape[0] > test_y.shape[0]:
            st.warning(f"Truncating predictions to match target shape.")
            predictions = predictions[:test_y.shape[0], :]

    # Handle different feature dimensions (crucial fix for the error)
    if predictions.shape[1] != test_y.shape[1]:
        st.warning(f"Prediction features {predictions.shape[1]} don't match target features {test_y.shape[1]}.")

        # We need to adjust based on what the predictions and targets represent
        if predictions.shape[1] == 1 and test_y.shape[1] > 1:
            # If target has more features, we'll use only the first feature (close price)
            st.info("Using only the first target feature (close price) for evaluation.")
            test_y_eval = test_y[:, 0:1]  # Keep only first feature but maintain 2D shape
        elif predictions.shape[1] > 1 and test_y.shape[1] == 1:
            # If predictions have more features, we'll use only the first prediction
            st.info("Using only the first prediction feature for evaluation.")
            predictions = predictions[:, 0:1]
        else:
            # For other cases, use only the first feature from both
            st.info("Using only the first feature from both predictions and targets.")
            predictions = predictions[:, 0:1]
            test_y_eval = test_y[:, 0:1]
    else:
        test_y_eval = test_y

    # Convert to numpy arrays
    test_y_np = test_y_eval.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    # Ensure arrays have the same shape before inverse transform
    if test_y_np.shape != predictions_np.shape:
        st.warning(f"Shape mismatch after conversion to numpy: test_y {test_y_np.shape}, predictions {predictions_np.shape}")
        # Make them the same length
        min_len = min(test_y_np.shape[0], predictions_np.shape[0])
        min_features = min(test_y_np.shape[1], predictions_np.shape[1])
        test_y_np = test_y_np[:min_len, :min_features]
        predictions_np = predictions_np[:min_len, :min_features]

    # Inverse transform to get actual prices
    test_actual = close_scaler.inverse_transform(test_y_np)
    test_predicted = close_scaler.inverse_transform(predictions_np)

    # Create DataFrame for visualization
    test_chart_data = pd.DataFrame({
        'Actual': test_actual.flatten(),
        'Predicted': test_predicted.flatten()
    })

    # Ensure we don't have any NaN values that could cause errors
    test_chart_data = test_chart_data.fillna(method='ffill').fillna(method='bfill')

    # Check for extremely large values that might be errors
    max_value = test_chart_data.max().max()
    min_value = test_chart_data.min().min()
    if max_value > 10000 or min_value < -10000:
        st.warning(f"Extreme values detected in predictions: min={min_value}, max={max_value}. This may indicate an issue with scaling.")

    # Calculate metrics
    mse = mean_squared_error(test_actual, test_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actual, test_predicted)

    # Display metrics
    st.subheader("Model Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mse:.4f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    col3.metric("Mean Absolute Error", f"{mae:.4f}")

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Prices")
    st.line_chart(test_chart_data)

    return test_chart_data


def backtest_strategy(stock_data, test_chart_data, backtest_start_date, backtest_end_date, initial_capital=10000, commission=0.001):
    """
    Backtest a simple trading strategy based on model predictions.

    Args:
        stock_data: Complete stock data DataFrame
        test_chart_data: DataFrame with 'Actual' and 'Predicted' columns
        backtest_start_date: Start date for backtesting
        backtest_end_date: End date for backtesting
        initial_capital: Starting capital for the simulation
        commission: Commission rate per trade (as a decimal)

    Returns:
        DataFrame with backtesting results
    """
    from trading import TradingEngine, trend_following_strategy, mean_reversion_strategy, combined_strategy
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime

    st.subheader("Strategy Backtesting")

    if test_chart_data is None or test_chart_data.empty:
        st.warning("No test data available for backtesting.")
        return None

    # Save backtesting data to session state if needed
    if st.session_state['backtest_chart_data'] is None:
        st.session_state['backtest_chart_data'] = test_chart_data.copy()
    else:
        # Use existing data if available and not in training mode
        test_chart_data = st.session_state['backtest_chart_data'].copy()

    # Filter the stock data to the backtesting date range
    backtest_start = pd.Timestamp(backtest_start_date)
    backtest_end = pd.Timestamp(backtest_end_date)

    # Get the actual historical data for the backtesting period
    backtest_data = stock_data.loc[backtest_start:backtest_end].copy()

    # Store backtest data in session state for reuse
    if st.session_state['backtest_data'] is None:
        st.session_state['backtest_data'] = backtest_data.copy()

    if backtest_data.empty:
        st.warning(f"No data available for backtesting in the selected date range ({backtest_start_date} to {backtest_end_date}).")
        return None

    st.write(f"Backtesting on data from {backtest_start_date} to {backtest_end_date} ({len(backtest_data)} trading days)")

    # Make sure the test_chart_data has a proper datetime index
    if not isinstance(test_chart_data.index, pd.DatetimeIndex):
        try:
            # Create a date range that matches the length of test_chart_data
            test_chart_data.index = pd.date_range(start=backtest_start, periods=len(test_chart_data))
            st.info("Created date index for test_chart_data")
        except Exception as e:
            st.error(f"Error creating date index: {str(e)}")
            return None

    # Handle length mismatch between test_chart_data and backtest_data
    if len(test_chart_data) != len(backtest_data):
        st.warning(f"Length mismatch: test_chart_data ({len(test_chart_data)} rows) and backtest_data ({len(backtest_data)} rows). Adjusting data lengths.")

        # Method 1: Resample test_chart_data to match backtest_data's index
        # If test_chart_data is longer, we'll take a subset
        if len(test_chart_data) > len(backtest_data):
            st.info("Truncating test_chart_data to match backtest data length")
            # Instead of just taking the first n rows, we'll try to align the indices
            common_idx = test_chart_data.index.intersection(backtest_data.index)
            if len(common_idx) > 0:
                test_chart_data = test_chart_data.loc[common_idx].copy()
                backtest_data = backtest_data.loc[common_idx].copy()
            else:
                # If no indices match, we'll have to use the first n rows
                test_chart_data = test_chart_data.iloc[:len(backtest_data)].copy()
                # Ensure the indices match by setting test_chart_data's index to backtest_data's index
                test_chart_data.index = backtest_data.index

        # If test_chart_data is shorter, we adjust backtest_data to match
        elif len(test_chart_data) < len(backtest_data):
            st.info("test_chart_data is shorter than backtest_data. Using available data only.")
            # Try to align by common index first
            common_idx = test_chart_data.index.intersection(backtest_data.index)
            if len(common_idx) > 0:
                test_chart_data = test_chart_data.loc[common_idx].copy()
                backtest_data = backtest_data.loc[common_idx].copy()
            else:
                # If no indices match, use the first n rows of backtest_data and align indices
                backtest_data = backtest_data.iloc[:len(test_chart_data)].copy()
                backtest_data.index = test_chart_data.index

            # Update the backtest end date
            backtest_end = backtest_data.index[-1]
            st.info(f"Adjusted backtesting end date to {backtest_end}")

    # Verify that the lengths match before proceeding
    if len(test_chart_data) != len(backtest_data):
        st.error(f"Unable to align data: test_chart_data ({len(test_chart_data)} rows) and backtest_data ({len(backtest_data)} rows) still have different lengths.")
        # As a last resort, create a synthetic test_chart_data using backtest_data's indices
        # and actual stock prices as both Actual and Predicted values
        st.info("Creating synthetic test data using actual stock prices")
        test_chart_data = pd.DataFrame(index=backtest_data.index)
        test_chart_data['Actual'] = backtest_data['Close']
        test_chart_data['Predicted'] = backtest_data['Close'] * (1 + np.random.normal(0, 0.01, len(backtest_data)))

    # Ensure we have both 'Actual' and 'Predicted' columns in test_chart_data
    if 'Actual' not in test_chart_data.columns or 'Predicted' not in test_chart_data.columns:
        st.error("test_chart_data must contain both 'Actual' and 'Predicted' columns.")
        return None

    # Backtesting controls
    with st.expander("Backtesting Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

        with col2:
            commission = st.number_input("Commission Rate (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100

        with col3:
            # Use session state to keep track of the selected strategy type
            strategy_type = st.selectbox(
                "Strategy Type",
                options=["Trend Following", "Mean Reversion", "Combined Strategy", "Buy and Hold"],
                index=["Trend Following", "Mean Reversion", "Combined Strategy", "Buy and Hold"].index(
                    st.session_state['backtest_strategy_type']
                )
            )
            # Store the selected strategy type in session state
            st.session_state['backtest_strategy_type'] = strategy_type

        # Risk management settings
        col1, col2 = st.columns(2)
        with col1:
            risk_per_trade = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100

        with col2:
            max_drawdown_limit = st.slider("Max Drawdown Limit (%)", min_value=5.0, max_value=50.0, value=25.0, step=5.0) / 100

        # Additional strategy parameters based on stored values
        strategy_params = {}

        if strategy_type == "Trend Following":
            # Get stored value if exists, otherwise use default
            default_threshold = st.session_state['backtest_params'].get('threshold', 0.5) * 100  # Default 0.5% instead of 0.1%
            threshold = st.slider("Price Change Threshold (%)", min_value=0.0, max_value=5.0, value=default_threshold, step=0.1) / 100
            strategy_params = {'threshold': threshold}
            strategy_func = trend_following_strategy

            # Add explanation about threshold
            st.info(f"The trend following strategy generates buy signals when the predicted price change exceeds {threshold*100:.1f}% and "
                    f"sell signals when it falls below -{threshold*100:.1f}%. "
                    f"A higher threshold means fewer trades but potentially higher confidence.")

        elif strategy_type == "Mean Reversion":
            col1, col2 = st.columns(2)
            with col1:
                default_lookback = st.session_state['backtest_params'].get('lookback', 20)
                lookback = st.slider("Mean Lookback Period (days)", min_value=1, max_value=60, value=default_lookback, step=1)
            with col2:
                default_std_dev = st.session_state['backtest_params'].get('std_dev', 1.5)
                std_dev = st.slider("Standard Deviation Threshold", min_value=0.5, max_value=3.0, value=default_std_dev, step=0.1)
            strategy_params = {'lookback': lookback, 'std_dev': std_dev}
            strategy_func = mean_reversion_strategy

        elif strategy_type == "Combined Strategy":
            col1, col2 = st.columns(2)
            with col1:
                default_threshold = st.session_state['backtest_params'].get('trend_threshold', 0.1) * 100
                threshold = st.slider("Trend Threshold (%)", min_value=0.0, max_value=5.0, value=default_threshold, step=0.1) / 100

                default_lookback = st.session_state['backtest_params'].get('mr_lookback', 20)
                lookback = st.slider("MR Lookback Period (days)", min_value=1, max_value=60, value=default_lookback, step=1)
            with col2:
                default_std_dev = st.session_state['backtest_params'].get('mr_std_dev', 1.5)
                std_dev = st.slider("MR Std Dev Threshold", min_value=0.5, max_value=3.0, value=default_std_dev, step=0.1)

                default_weight = st.session_state['backtest_params'].get('weight_trend', 0.5)
                weight_trend = st.slider("Trend Weight", min_value=0.0, max_value=1.0, value=default_weight, step=0.1)
            strategy_params = {
                'trend_threshold': threshold,
                'mr_lookback': lookback,
                'mr_std_dev': std_dev,
                'weight_trend': weight_trend
            }
            strategy_func = combined_strategy

        elif strategy_type == "Buy and Hold":
            # For Buy and Hold, we'll create a simple strategy function that just buys at the beginning
            def buy_and_hold_strategy(data):
                signals = pd.Series(0, index=data.index)
                signals.iloc[0] = 1  # Buy on first day
                return signals
            strategy_func = buy_and_hold_strategy
            strategy_params = {}

        # Store the strategy parameters in session state
        st.session_state['backtest_params'] = strategy_params

    # Additional backtesting options
    debug_mode = st.checkbox("Debug Mode", value=False,
                            help="Show additional debugging information for the backtesting process")

    # Initialize trading engine with settings
    trading_engine = TradingEngine(
        mode='backtest',
        initial_capital=initial_capital,
        commission=commission,
        risk_per_trade=risk_per_trade,
        max_drawdown_limit=max_drawdown_limit
    )

    # Run backtesting with progress indicator
    with st.spinner("Running backtesting simulation..."):
        try:
            # Generate signals first to check if they're being created properly
            if debug_mode:
                st.subheader("Debug: Signal Generation")
                # Make a copy of the data for debugging
                debug_data = test_chart_data.copy()
                # Generate signals directly
                signals = strategy_func(debug_data, **strategy_params)

                # Display signal statistics
                buy_signals = (signals == 1).sum()
                sell_signals = (signals == -1).sum()
                hold_signals = (signals == 0).sum()

                st.write(f"Total data points: {len(signals)}")
                st.write(f"Buy signals: {buy_signals} ({buy_signals/len(signals)*100:.1f}%)")
                st.write(f"Sell signals: {sell_signals} ({sell_signals/len(signals)*100:.1f}%)")
                st.write(f"Hold signals: {hold_signals} ({hold_signals/len(signals)*100:.1f}%)")

                # Show first few signals
                debug_data['Signal'] = signals
                st.write("Sample of signals (first 10 days):")
                st.dataframe(debug_data[['Actual', 'Predicted', 'Signal']].head(10))

                # Show periods with active trading
                if buy_signals > 0 or sell_signals > 0:
                    active_periods = debug_data[debug_data['Signal'] != 0]
                    st.write(f"Sample of active trading periods ({len(active_periods)} days with signals):")
                    st.dataframe(active_periods[['Actual', 'Predicted', 'Signal']].head(10))
                else:
                    st.error("No buy or sell signals were generated! Check your strategy parameters.")

                # Plot predicted vs actual with signals
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=debug_data.index, y=debug_data['Actual'], mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=debug_data.index, y=debug_data['Predicted'], mode='lines', name='Predicted Price'))

                # Add buy/sell markers
                buy_points = debug_data[debug_data['Signal'] == 1]
                sell_points = debug_data[debug_data['Signal'] == -1]

                if not buy_points.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_points.index,
                        y=buy_points['Actual'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='Buy Signal'
                    ))

                if not sell_points.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_points.index,
                        y=sell_points['Actual'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='Sell Signal'
                    ))

                fig.update_layout(title='Price with Trading Signals')
                st.plotly_chart(fig)

            # Run the actual backtest
            backtest_results, trades, metrics = trading_engine.backtest(
                strategy=strategy_func,
                data=test_chart_data,
                strategy_params=strategy_params
            )

            if backtest_results is None or backtest_results.empty:
                st.warning("Backtesting failed. Please check your data and parameters.")
                return None

            # Display performance metrics
            st.subheader("Performance Metrics")

            # First row of metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{metrics['Total Return']:.2f}%")
            col2.metric("Initial Capital", f"${metrics['Initial Capital']:.2f}", f"${metrics['Final Value'] - metrics['Initial Capital']:.2f}")
            col3.metric("Final Value", f"${metrics['Final Value']:.2f}")
            col4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.4f}")

            # Second row of metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", f"{metrics['Total Trades']}")
            col2.metric("Win Rate", f"{metrics['Win Rate']:.2f}%")
            col3.metric("Avg Profit/Loss", f"${metrics['Avg Profit']:.2f}")
            col4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")

            # Compare with benchmark (buy and hold from day 1)
            if strategy_type != "Buy and Hold":
                try:
                    first_price = backtest_results['Actual'].iloc[0]
                    last_price = backtest_results['Actual'].iloc[-1]
                    buy_hold_return = (last_price - first_price) / first_price * 100

                    st.metric(
                        "Strategy vs Buy & Hold",
                        f"{metrics['Total Return']:.2f}% vs {buy_hold_return:.2f}%",
                        f"{metrics['Total Return'] - buy_hold_return:.2f}%"
                    )
                except Exception as e:
                    st.warning(f"Could not calculate buy & hold comparison: {str(e)}")

            # Visualize portfolio value over time
            st.subheader("Portfolio Value Over Time")

            # Create more advanced chart with Plotly
            if not backtest_results.empty and 'PortfolioValue' in backtest_results.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=backtest_results.index,
                    y=backtest_results['PortfolioValue'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))

                # Add annotations for Buy/Sell signals if we have trade data
                if trades is not None and not trades.empty:
                    # Only show debug information in debug mode
                    if debug_mode:
                        st.write(f"Trade dates range: {trades['date'].min()} to {trades['date'].max()}")
                        st.write(f"Number of unique trade dates: {trades['date'].nunique()} out of {len(trades)} trades")

                    # Check if we still have epoch dates (1970-01-01) or need to process the trades visualization
                    if pd.api.types.is_datetime64_any_dtype(trades['date']):
                        # Process epoch dates
                        if (trades['date'].dt.year == 1970).mean() > 0.5:  # If more than 50% of dates are in 1970
                            st.warning("Detected epoch dates (1970-01-01). Redistributing trades over the backtest period.")

                            # Create an artificial distribution of trades across the backtest period
                            date_range = backtest_results.index

                            if len(date_range) >= len(trades):
                                # Group trades by action
                                buys = trades[trades['action'] == 'BUY']
                                sells = trades[trades['action'] == 'SELL']
                                stop_losses = trades[trades['action'] == 'STOP_LOSS']

                                # Distribute each type of trade separately to maintain the logical sequence
                                # (buys should generally come before sells)

                                # First third of the date range for buys
                                if len(buys) > 0:
                                    buy_indices = np.linspace(0, len(date_range)//3, len(buys)).astype(int)
                                    buy_dates = [date_range[i] for i in buy_indices]
                                    trades.loc[trades['action'] == 'BUY', 'date'] = buy_dates

                                # Middle third for stop losses
                                if len(stop_losses) > 0:
                                    stop_indices = np.linspace(len(date_range)//3, 2*len(date_range)//3, len(stop_losses)).astype(int)
                                    stop_dates = [date_range[i] for i in stop_indices]
                                    trades.loc[trades['action'] == 'STOP_LOSS', 'date'] = stop_dates

                                # Last third for sells
                                if len(sells) > 0:
                                    sell_indices = np.linspace(2*len(date_range)//3, len(date_range)-1, len(sells)).astype(int)
                                    sell_dates = [date_range[i] for i in sell_indices]
                                    trades.loc[trades['action'] == 'SELL', 'date'] = sell_dates

                                st.write(f"Redistributed trades from {date_range[0]} to {date_range[-1]}")
                            else:
                                # If we have more trades than dates, distribute them evenly
                                trade_indices = np.linspace(0, len(date_range)-1, len(trades)).astype(int)
                                new_dates = [date_range[i] for i in trade_indices]
                                trades['date'] = new_dates
                                st.write(f"Redistributed trades from {new_dates[0]} to {new_dates[-1]}")

                    # Always visualize trades regardless of date conversion
                    # Ensure trades are sorted by date
                    try:
                        # If 'date' is not a datetime, try one more time for visualization
                        if not pd.api.types.is_datetime64_any_dtype(trades['date']):
                            trades['date'] = pd.to_datetime(trades['date'], errors='coerce')
                            # For any failed conversions, use evenly spaced dates from the backtest period
                            if trades['date'].isna().any():
                                mask = trades['date'].isna()
                                n_missing = mask.sum()
                                indices = np.linspace(0, len(backtest_results.index)-1, n_missing+2)[1:-1].astype(int)
                                replacement_dates = [backtest_results.index[i] for i in indices]
                                trades.loc[mask, 'date'] = replacement_dates

                        trades = trades.sort_values('date')

                        # Group trades by action
                        buys = trades[trades['action'] == 'BUY']
                        sells = trades[trades['action'] == 'SELL']
                        stop_losses = trades[trades['action'] == 'STOP_LOSS']

                        # Create a legend for each action type (only once)
                        buy_added = False
                        sell_added = False
                        stop_loss_added = False

                        # Add buy markers - make sure to map dates correctly to portfolio values
                        for _, trade in buys.iterrows():
                            date = trade['date']
                            # Find the closest date in backtest_results.index if exact match not found
                            if date not in backtest_results.index:
                                closest_date = backtest_results.index[backtest_results.index.get_indexer([date], method='nearest')[0]]
                                date = closest_date

                            if date in backtest_results.index:
                                fig.add_trace(go.Scatter(
                                    x=[date],
                                    y=[backtest_results.loc[date, 'PortfolioValue']],
                                    mode='markers',
                                    marker=dict(color='green', size=10, symbol='triangle-up'),
                                    name='Buy' if not buy_added else None,
                                    showlegend=not buy_added,
                                    hovertemplate=f"Buy: {date}<br>Price: ${trade['price']:.2f}<br>Size: {trade['size']:.2f}"
                                ))
                                buy_added = True

                        # Add sell markers
                        for _, trade in sells.iterrows():
                            date = trade['date']
                            # Find the closest date in backtest_results.index if exact match not found
                            if date not in backtest_results.index:
                                closest_date = backtest_results.index[backtest_results.index.get_indexer([date], method='nearest')[0]]
                                date = closest_date

                            if date in backtest_results.index:
                                fig.add_trace(go.Scatter(
                                    x=[date],
                                    y=[backtest_results.loc[date, 'PortfolioValue']],
                                    mode='markers',
                                    marker=dict(color='red', size=10, symbol='triangle-down'),
                                    name='Sell' if not sell_added else None,
                                    showlegend=not sell_added,
                                    hovertemplate=f"Sell: {date}<br>Price: ${trade['price']:.2f}<br>Size: {trade['size']:.2f}<br>Profit/Loss: ${trade['profit_loss']:.2f} ({trade['profit_loss_pct']:.2f}%)"
                                ))
                                sell_added = True

                        # Add stop loss markers
                        for _, trade in stop_losses.iterrows():
                            date = trade['date']
                            # Find the closest date in backtest_results.index if exact match not found
                            if date not in backtest_results.index:
                                closest_date = backtest_results.index[backtest_results.index.get_indexer([date], method='nearest')[0]]
                                date = closest_date

                            if date in backtest_results.index:
                                fig.add_trace(go.Scatter(
                                    x=[date],
                                    y=[backtest_results.loc[date, 'PortfolioValue']],
                                    mode='markers',
                                    marker=dict(color='purple', size=10, symbol='x'),
                                    name='Stop Loss' if not stop_loss_added else None,
                                    showlegend=not stop_loss_added,
                                    hovertemplate=f"Stop Loss: {date}<br>Price: ${trade['price']:.2f}<br>Size: {trade['size']:.2f}<br>Profit/Loss: ${trade['profit_loss']:.2f} ({trade['profit_loss_pct']:.2f}%)"
                                ))
                                stop_loss_added = True
                    except Exception as e:
                        if debug_mode:
                            st.error(f"Error displaying trade markers: {str(e)}")

            # Improve the layout with better date formatting and hover information
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d',
                    tickangle=-45,
                    tickmode='auto',
                    nticks=20
                )
            )

            # Add hover data
            fig.update_traces(
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: $%{y:.2f}<extra></extra>'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Visualize drawdowns
            st.subheader("Drawdowns")
            if not backtest_results.empty and 'Drawdown' in backtest_results.columns:
                fig = px.area(
                    backtest_results,
                    x=backtest_results.index,
                    y='Drawdown',
                    color_discrete_sequence=['red']
                )
                fig.update_layout(
                    title='Portfolio Drawdowns',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Show trade summary
            if trades is not None and not trades.empty:
                st.subheader("Trade Summary")

                # Format the trades dataframe for display
                display_trades = trades.copy()

                # Add more calculated columns
                if 'profit_loss' in display_trades.columns:
                    display_trades['profit_loss_color'] = display_trades['profit_loss'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    )

                st.dataframe(display_trades)

                # Plot profit/loss distribution
                if 'action' in trades.columns and 'profit_loss' in trades.columns:
                    profit_loss_data = trades[trades['action'].isin(['SELL', 'STOP_LOSS'])]['profit_loss']
                    if not profit_loss_data.empty:
                        st.subheader("Profit/Loss Distribution")
                        fig = px.histogram(
                            profit_loss_data,
                            nbins=20,
                            color_discrete_sequence=['blue']
                        )
                        fig.update_layout(
                            title='Trade Profit/Loss Distribution',
                            xaxis_title='Profit/Loss ($)',
                            yaxis_title='Number of Trades'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Show detailed backtest data
            with st.expander("Detailed Backtest Data", expanded=False):
                st.dataframe(backtest_results)

            # Add ability to download backtest results
            col1, col2 = st.columns(2)
            with col1:
                csv = backtest_results.to_csv()
                st.download_button(
                    label="Download Backtest Results",
                    data=csv,
                    file_name="backtest_results.csv",
                    mime="text/csv",
                )

            with col2:
                if trades is not None and not trades.empty:
                    trades_csv = trades.to_csv()
                    st.download_button(
                        label="Download Trade History",
                        data=trades_csv,
                        file_name="trade_history.csv",
                        mime="text/csv",
                    )

            return backtest_results

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Backtesting failed with error: {str(e)}")

            # Provide more detailed error information in debug mode
            if debug_mode:
                st.expander("Detailed Error Information", expanded=True).code(error_details)

                # Suggest possible solutions based on the error type
                error_type = str(e).lower()
                if 'profit_loss' in error_type:
                    st.info("""
                    The error is related to the profit_loss calculation. This might be caused by:
                    1. No trades being executed during the backtesting period
                    2. Issues with the data format in the trades record
                    3. Missing profit_loss values for some trades

                    Try adjusting your strategy parameters to ensure there are enough trading signals generated.
                    """)
                elif 'index' in error_type or 'key' in error_type:
                    st.info("""
                    The error is related to accessing an index or key that doesn't exist. This might be caused by:
                    1. Missing or misaligned data in the test_chart_data
                    2. Issues with date alignment between different data sources

                    Check that your data contains all required columns ('Actual' and 'Predicted').
                    """)
                elif 'nan' in error_type or 'inf' in error_type:
                    st.info("""
                    The error is related to NaN (Not a Number) or infinite values. This might be caused by:
                    1. Division by zero in calculations
                    2. Missing values in the input data
                    3. Extreme price movements causing numerical issues

                    Check your data for missing values and ensure proper data preprocessing.
                    """)

            # Suggest general solutions
            st.warning("""
            Here are some steps to try solving the backtesting error:
            1. Enable Debug Mode to see detailed information about signal generation
            2. Try a different strategy type
            3. Adjust the strategy parameters (try less extreme values)
            4. Make sure your test data has proper 'Actual' and 'Predicted' columns
            5. Try a different time period for backtesting
            """)

            return None


def predict_future_prices(model, stock_data, close_scaler, model_params, device, num_days=30):
    """
    Predict future stock prices using the trained model.

    Args:
        model: Trained model
        stock_data: DataFrame with stock data
        close_scaler: Scaler for close prices
        model_params: Dictionary with model parameters
        device: Device to run prediction on
        num_days: Number of days to predict

    Returns:
        DataFrame with predicted prices
    """
    st.subheader("Future Price Predictions")

    # Get user input for number of days to predict
    num_days = st.slider("Number of days to predict", min_value=1, max_value=90, value=30)

    if st.button("Predict Future Prices"):
        with st.spinner(f"Predicting prices for the next {num_days} days..."):
            try:
                # Get the last sequence from the data
                seq_length = model_params['seq_length']

                # Get the features used in the model
                features = []
                for col in stock_data.columns:
                    if col.startswith('Normalized_') or col in ['RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower']:
                        features.append(col)

                # Get the last sequence
                last_sequence = stock_data[features].iloc[-seq_length:].values
                last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)

                # Set model to evaluation mode
                model.eval()

                # Initialize predictions
                future_predictions = []
                current_sequence = last_sequence_tensor.clone()

                # Generate predictions
                with torch.no_grad():
                    for _ in range(num_days):
                        # Get prediction for next day
                        prediction = model(current_sequence)
                        future_predictions.append(prediction.item())

                        # Create a new datapoint with the prediction
                        new_point = current_sequence[:, -1:, :].clone()
                        new_point[:, :, 0] = prediction.item()  # Set the normalized close price

                        # Update the sequence by removing the oldest point and adding the new one
                        current_sequence = torch.cat([current_sequence[:, 1:, :], new_point], dim=1)

                # Convert predictions to actual prices
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_prices = close_scaler.inverse_transform(future_predictions).flatten()

                # Create dates for the predictions
                last_date = stock_data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)

                # Create DataFrame with predictions
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price': future_prices
                })
                future_df.set_index('Date', inplace=True)

                # Display predictions
                st.subheader(f"Predicted Prices for Next {num_days} Days")
                st.dataframe(future_df)

                # Plot predictions
                plot_df = pd.DataFrame({
                    'Historical': stock_data['Close'][-30:],
                    'Predicted': [None] * 30 + list(future_prices)
                }, index=list(stock_data.index[-30:]) + list(future_dates))

                st.line_chart(plot_df)

                return future_df

            except Exception as e:
                st.error(f"Error predicting future prices: {str(e)}")
                logger.error(f"Error predicting future prices: {str(e)}", exc_info=True)
                return None

    return None


def save_and_download_model(model_state_dict):
    """Save the trained model and provide download option."""
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Generate filename with timestamp
    model_filename = f'stock_prediction_model_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.pt'
    model_path = os.path.join(MODEL_DIR, model_filename)

    # Save model
    torch.save(model_state_dict, model_path)
    logger.info(f"Model saved to {model_path}")

    # Download button
    st.subheader("Download Trained Model")
    download_model = st.button("Download Model")

    if download_model:
        with open(model_path, "rb") as f:
            bytes_data = f.read()
            st.download_button(
                label="Download Model File",
                data=bytes_data,
                file_name=model_filename,
                mime="application/octet-stream"
            )


def train_model_callback():
    """Callback for the train model button"""
    st.session_state['model_trained'] = True


def visualize_neural_network_3d(model):
    """
    Create a visually appealing 3D visualization of the transformer neural network model
    with improved lighting and clarity.

    Args:
        model: The trained TransformerModel instance

    Returns:
        Plotly figure object with 3D visualization
    """
    try:
        # Extract model architecture parameters
        input_dim = model.input_dim
        hidden_dim = model.hidden_dim
        output_dim = model.output_dim
        num_layers = model.transformer_encoder.num_layers
        num_heads = model.num_heads

        # Enhanced color palette using named CSS colors with good contrast
        COLOR_PALETTE = {
            'input': 'dodgerblue',     # Vibrant blue
            'embedding': 'mediumseagreen',  # Clear green
            'attention': 'darkorange',  # Vibrant orange
            'ffn': 'mediumpurple',       # Distinct purple
            'output': 'crimson',    # Rich red
            'connections': 'slategray', # Medium gray for connections
            'residual': 'gold'      # Gold for residual connections
        }

        # Initialize lists for node positions and connections
        nodes_x, nodes_y, nodes_z = [], [], []
        nodes_text, nodes_size, nodes_color = [], [], []
        edge_x, edge_y, edge_z, edge_color = [], [], [], []

        # Define a cleaner layer structure
        layers = []

        # Input layer
        layers.append({
            'name': 'Input',
            'size': input_dim,
            'color': COLOR_PALETTE['input'],
            'node_size': 8
        })

        # Embedding layer
        layers.append({
            'name': 'Embedding',
            'size': hidden_dim,
            'color': COLOR_PALETTE['embedding'],
            'node_size': 8
        })

        # Add transformer layers
        for i in range(num_layers):
            # Attention mechanism
            layers.append({
                'name': f'Attention {i+1}',
                'size': hidden_dim,
                'color': COLOR_PALETTE['attention'],
                'node_size': 8
            })

            # Feed-forward network
            layers.append({
                'name': f'FFN {i+1}',
                'size': hidden_dim,
                'color': COLOR_PALETTE['ffn'],
                'node_size': 8
            })

        # Output layer
        layers.append({
            'name': 'Output',
            'size': output_dim,
            'color': COLOR_PALETTE['output'],
            'node_size': 10
        })

        # Layer positioning - use increased spacing for clarity
        layer_spacing = 4.0
        x_positions = [i * layer_spacing for i in range(len(layers))]

        # Max nodes per layer - balance between detail and performance
        max_nodes_per_layer = 25

        # Store indices for safer connections
        layer_start_indices = [0]

        # Generate nodes for each layer
        for layer_idx, layer in enumerate(layers):
            layer_size = layer['size']
            node_count = min(layer_size, max_nodes_per_layer)
            skip = max(1, layer_size // node_count)
            actual_nodes_added = 0

            # Attention layers get special treatment
            if 'Attention' in layer['name']:
                # Create a visually interesting circular pattern
                radius = 2.5
                heads_to_show = min(num_heads, 4)
                nodes_per_head = max(3, node_count // heads_to_show)

                for head in range(heads_to_show):
                    # Generate visually distinct colors for each attention head using HSL
                    hue = (head / heads_to_show) * 360  # Full color range
                    head_color = f'hsl({hue}, 80%, 60%)'  # Use HSL for better control

                    # Create a circle of nodes for this attention head
                    for j in range(nodes_per_head):
                        if actual_nodes_added >= node_count:
                            break

                        # Calculate position on a circle
                        angle = (j / nodes_per_head) * (2 * np.pi) + (head * np.pi / heads_to_show)
                        y_pos = radius * np.sin(angle)
                        z_pos = radius * np.cos(angle)

                        # Add node
                        nodes_x.append(x_positions[layer_idx])
                        nodes_y.append(y_pos)
                        nodes_z.append(z_pos)
                        nodes_text.append(f"{layer['name']}<br>Head {head+1}, Node {j+1}")
                        nodes_size.append(layer['node_size'])
                        nodes_color.append(head_color)
                        actual_nodes_added += 1

            # Feed-forward network layers
            elif 'FFN' in layer['name']:
                # Create a 3D grid pattern
                side_len = int(np.ceil(np.sqrt(node_count)))
                spacing = 5.0 / side_len

                # Use a gradient of purples for FFN nodes
                base_hue = 270  # Purple

                idx = 0
                for i in range(side_len):
                    for j in range(side_len):
                        if actual_nodes_added >= node_count:
                            break

                        # Position in a grid
                        y_pos = (i - side_len/2 + 0.5) * spacing
                        z_pos = (j - side_len/2 + 0.5) * spacing

                        # Vary lightness based on position
                        lightness = 40 + 30 * (idx / (side_len * side_len))
                        node_color = f'hsl({base_hue}, 70%, {lightness}%)'

                        # Add node with slight random offset for more natural look
                        nodes_x.append(x_positions[layer_idx])
                        nodes_y.append(y_pos + np.random.normal(0, 0.05))
                        nodes_z.append(z_pos + np.random.normal(0, 0.05))
                        nodes_text.append(f"{layer['name']}<br>Node {actual_nodes_added+1}/{node_count}")
                        nodes_size.append(layer['node_size'])
                        nodes_color.append(node_color)
                        actual_nodes_added += 1
                        idx += 1

            # Input, embedding and output layers
            else:
                # Arrange in a circular pattern
                radius = 2.0

                # Base color from the layer definition
                base_color = layer['color']

                for j in range(0, layer_size, skip):
                    if actual_nodes_added >= node_count:
                        break

                    # Calculate angle
                    angle = (j / layer_size) * 2 * np.pi
                    y_pos = radius * np.sin(angle)
                    z_pos = radius * np.cos(angle)

                    # Add node
                    nodes_x.append(x_positions[layer_idx])
                    nodes_y.append(y_pos)
                    nodes_z.append(z_pos)
                    nodes_text.append(f"{layer['name']}<br>Node {j+1}/{layer_size}")
                    nodes_size.append(layer['node_size'])
                    nodes_color.append(base_color)
                    actual_nodes_added += 1

            # Record the start index of the next layer
            layer_start_indices.append(len(nodes_x))

        # Create connections between layers
        for layer_idx in range(1, len(layers)):
            current_layer_start = layer_start_indices[layer_idx]
            current_layer_end = layer_start_indices[layer_idx + 1] if layer_idx + 1 < len(layer_start_indices) else len(nodes_x)
            prev_layer_start = layer_start_indices[layer_idx - 1]
            prev_layer_end = layer_start_indices[layer_idx]

            # Get current and previous layer
            current_layer = layers[layer_idx]
            prev_layer = layers[layer_idx - 1]

            # Number of connections to draw between layers
            # Use a reduced set of connections to avoid visual clutter
            connection_density = 0.3

            # For each node in current layer
            for i in range(current_layer_start, current_layer_end):
                # Connect to nodes in previous layer based on density
                for j in range(prev_layer_start, prev_layer_end):
                    # Only create some connections based on density
                    if np.random.rand() < connection_density:
                        # Add edge (connection)
                        edge_x.extend([nodes_x[i], nodes_x[j], None])
                        edge_y.extend([nodes_y[i], nodes_y[j], None])
                        edge_z.extend([nodes_z[i], nodes_z[j], None])

                        # Create gradient connections that blend from source to target color
                        if 'Attention' in current_layer['name']:
                            # Use the node colors directly for attention heads - avoid None values
                            edge_color.append(nodes_color[i])
                            edge_color.append(nodes_color[j])
                            # Need to repeat the last color instead of None for the break
                            edge_color.append(nodes_color[j])
                        else:
                            # Set opacity for regular layer connections
                            opacity = 0.7
                            # For standard connections, use rgba format - avoid None values
                            connection_color = f'rgba(100, 100, 100, {opacity})'
                            edge_color.append(connection_color)
                            edge_color.append(connection_color)
                            edge_color.append(connection_color)

            # Add residual connections for transformer layers
            if layer_idx >= 3 and layer_idx % 2 == 0 and 'FFN' in current_layer['name']:
                # Find the layer before the attention layer
                skip_src_idx = layer_idx - 2
                if skip_src_idx >= 0:
                    # Connect a few sample nodes to show residual connection
                    for _ in range(5):
                        # Select random nodes from each layer
                        src_node = random.randint(layer_start_indices[skip_src_idx], layer_start_indices[skip_src_idx+1]-1)
                        tgt_node = random.randint(current_layer_start, current_layer_end-1)

                        if src_node < len(nodes_x) and tgt_node < len(nodes_x):
                            # Add skip connection
                            edge_x.extend([nodes_x[src_node], nodes_x[tgt_node], None])
                            edge_y.extend([nodes_y[src_node], nodes_y[tgt_node], None])
                            edge_z.extend([nodes_z[src_node], nodes_z[tgt_node], None])

                            # Use the residual color from palette - avoid None values
                            residual_color = 'rgba(255, 215, 0, 0.8)'
                            edge_color.append(residual_color)
                            edge_color.append(residual_color)
                            edge_color.append(residual_color)

        # Create edges trace with improved styling
        edges_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(color=edge_color, width=3),  # Increased line width for better visibility
            hoverinfo='none',
            mode='lines'
        )

        # Create nodes trace with improved styling and lighting
        nodes_trace = go.Scatter3d(
            x=nodes_x, y=nodes_y, z=nodes_z,
            mode='markers',
            hovertext=nodes_text,
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=nodes_color,
                size=nodes_size,
                opacity=0.95,  # Slightly increased opacity
                line=dict(width=1.5, color='rgba(40,40,40,0.9)'),  # Darker, thicker outline for better visibility
                symbol='circle'
            )
        )

        # Create figure with proper lighting and camera settings
        fig = go.Figure(data=[edges_trace, nodes_trace])

        # Add layer labels
        annotations = []
        for i, layer in enumerate(layers):
            annotations.append(
                dict(
                    x=x_positions[i],
                    y=-5,
                    z=0,
                    text=layer['name'],
                    showarrow=False,
                    font=dict(
                        color=layer['color'],
                        size=12,
                        family='Arial, sans-serif'
                    )
                )
            )

        # Update layout with improved settings
        fig.update_layout(
            title=dict(
                text="Neural Network Architecture",
                font=dict(size=24, family='Arial, sans-serif')
            ),
            scene=dict(
                annotations=annotations,
                xaxis=dict(
                    title='',
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showbackground=False  # Remove background
                ),
                yaxis=dict(
                    title='',
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showbackground=False  # Remove background
                ),
                zaxis=dict(
                    title='',
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showbackground=False  # Remove background
                ),
                aspectratio=dict(x=2, y=1, z=1),
                camera=dict(
                    eye=dict(x=1.5, y=1, z=1),  # Adjust viewing angle
                    up=dict(x=0, y=0, z=1)      # Set "up" direction
                ),
                dragmode='orbit'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=False,
            width=900,
            height=650,
            paper_bgcolor='rgba(250,250,250,0.95)',  # Very light gray background
            plot_bgcolor='rgba(250,250,250,0.95)'
        )

        # Add elegant legend
        layer_legend = "<br>".join([
            f"<span style='color:{layer['color']}'><b>{layer['name']}</b></span>"
            for layer in layers
        ])

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"<b>Transformer Architecture</b><br><br>{layer_legend}<br><br><b>Model Details:</b><br>Input Dim: {input_dim}<br>Hidden Dim: {hidden_dim}<br>Output Dim: {output_dim}<br>Num Layers: {num_layers}<br>Num Heads: {num_heads}",
            showarrow=False,
            font=dict(size=12, family='Arial, sans-serif'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            borderpad=8,
            align="left"
        )

        return fig

    except Exception as e:
        # Create simple fallback visualization with error info
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"Error generating visualization: {str(e)}<br>Please try again with a smaller model.",
            showarrow=False,
            font=dict(size=14, color="crimson")
        )
        fig.update_layout(
            title="Visualization Error",
            width=900,
            height=500
        )
        st.error(f"Visualization error: {str(e)}")
        return fig

def visualize_multi_head_attention(model):
    """
    Create a beautiful 3D visualization of the multi-head attention mechanism
    with improved lighting and visual clarity.

    Args:
        model: The trained TransformerModel instance

    Returns:
        Plotly figure object with 3D visualization
    """
    try:
        # Extract model parameters
        hidden_dim = model.hidden_dim
        num_heads = model.num_heads
        head_dim = hidden_dim // num_heads

        # Create figure
        fig = go.Figure()

        # Parameters
        seq_length = 10
        heads_to_show = min(num_heads, 4)

        # Generate sample attention scores
        np.random.seed(42)
        attention_scores = np.zeros((heads_to_show, seq_length, seq_length))

        # Create more interesting patterns for visualization
        for h in range(heads_to_show):
            # Generate different attention patterns for different heads
            if h == 0:  # Diagonal attention (focusing on current token)
                for i in range(seq_length):
                    for j in range(seq_length):
                        attention_scores[h, i, j] = np.exp(-0.5 * ((i-j)/2)**2)
            elif h == 1:  # Looking ahead
                for i in range(seq_length):
                    for j in range(seq_length):
                        if j >= i:
                            attention_scores[h, i, j] = np.exp(-0.5 * ((j-i)/2)**2)
            elif h == 2:  # Looking backward
                for i in range(seq_length):
                    for j in range(seq_length):
                        if j <= i:
                            attention_scores[h, i, j] = np.exp(-0.5 * ((i-j)/2)**2)
            else:  # Global attention
                for i in range(seq_length):
                    peak1 = np.random.randint(0, seq_length)
                    peak2 = np.random.randint(0, seq_length)
                    for j in range(seq_length):
                        attention_scores[h, i, j] = np.exp(-0.5 * min(((j-peak1)/1.5)**2, ((j-peak2)/1.5)**2))

            # Normalize
            attention_scores[h] = attention_scores[h] / attention_scores[h].sum(axis=1, keepdims=True)

        # Enhanced color schemes for different heads using perceptually distinct colorscales
        # Using named colorscales that provide good contrast and visual appeal
        colors = ['Viridis', 'Plasma', 'Inferno', 'Magma']

        # Create a 2×2 grid
        grid_size = 2
        spacing = 1.5

        # For each attention head
        for h in range(heads_to_show):
            # Calculate grid position
            grid_x = h % grid_size
            grid_y = h // grid_size

            # Generate x, y coordinates
            x = np.arange(seq_length)
            y = np.arange(seq_length)
            x_grid, y_grid = np.meshgrid(x, y)

            # Offset by grid position
            x_offset = grid_x * (seq_length + spacing)
            y_offset = grid_y * (seq_length + spacing)

            # Create surface for this attention head with enhanced visual properties
            fig.add_trace(go.Surface(
                z=attention_scores[h],
                x=x_grid + x_offset,
                y=y_grid + y_offset,
                colorscale=colors[h % len(colors)],
                showscale=h == 0,  # Only show colorbar for first head
                name=f'Head {h+1}',
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="white",
                    project_z=True
                ),
                lighting=dict(
                    ambient=0.7,      # Increased ambient light for better visibility
                    diffuse=0.9,      # Enhanced diffuse lighting for depth
                    fresnel=0.2,      # Subtle edge highlighting
                    roughness=0.3,    # Smoother surface appearance
                    specular=1.0      # Enhanced specular highlights for "pop"
                ),
                opacity=0.95,
                hoverinfo='text',
                hovertext=[[f"Query: {i}, Key: {j}, Score: {attention_scores[h,i,j]:.3f}"
                           for j in range(seq_length)]
                           for i in range(seq_length)]
            ))

            # Add head label as a 3D text with enhanced visibility
            fig.add_trace(go.Scatter3d(
                x=[x_offset + seq_length/2],
                y=[y_offset + seq_length/2],
                z=[1.4],
                mode='text',
                text=[f'<b>Head {h+1}</b>'],  # Bold for better visibility
                textfont=dict(size=16, color='black', family='Arial, sans-serif'),
                showlegend=False
            ))

            # Add some connecting lines between key points to highlight attention flow
            if h < heads_to_show:
                # Find top 3 strongest attention points
                flat_idx = np.argsort(attention_scores[h].flatten())[-4:]
                top_points = [(idx // seq_length, idx % seq_length) for idx in flat_idx]

                for i, j in top_points:
                    # Add a line from query to its strongest key with enhanced visibility
                    line_x = [x_offset + i, x_offset + j, None]
                    line_y = [y_offset + j, y_offset + i, None]
                    line_z = [attention_scores[h, i, j] * 1.1, 0.05, None]

                    # HSL colors for better control and visibility
                    hue = (h * 90) % 360  # Different hue for each head
                    line_color = f'hsla({hue}, 100%, 80%, 0.9)'  # High saturation, bright, slight transparency

                    fig.add_trace(go.Scatter3d(
                        x=line_x, y=line_y, z=line_z,
                        mode='lines',
                        line=dict(width=5, color=line_color),  # Single color for the entire line
                        showlegend=False
                    ))

        # Update layout with enhanced settings
        fig.update_layout(
            title=dict(
                text="Multi-Head Attention Visualization",
                font=dict(size=24, family='Arial, sans-serif')
            ),
            scene=dict(
                xaxis=dict(
                    title='Query Position',
                    tickmode='array',
                    tickvals=[i * seq_length + seq_length/2 for i in range(grid_size)],
                    ticktext=[f'Head {i+1}' for i in range(grid_size)],
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                yaxis=dict(
                    title='Key Position',
                    tickmode='array',
                    tickvals=[i * seq_length + seq_length/2 for i in range(grid_size)],
                    ticktext=[f'Head {i+grid_size+1}' for i in range(grid_size)],
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                zaxis=dict(
                    title='Attention Score',
                    range=[0, 1.4],
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                aspectratio=dict(x=1.8, y=1.8, z=1),
                camera=dict(
                    eye=dict(x=2, y=-1.5, z=1.5),  # Optimized viewing angle
                    up=dict(x=0, y=0, z=1)
                ),
                dragmode='orbit'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            width=900,
            height=650,
            paper_bgcolor='rgba(250,250,250,0.95)',
            plot_bgcolor='rgba(250,250,250,0.95)'
        )

        # Add attention explanation with enhanced styling
        patterns = [
            "Diagonal focus (current token)",
            "Forward attention (looking ahead)",
            "Backward attention (looking behind)",
            "Global attention (specific tokens)"
        ]

        explanation_text = "<b>Multi-Head Attention Patterns</b><br>"
        for h in range(min(heads_to_show, len(patterns))):
            # Use specific colors for each head in the explanation
            hue = (h * 90) % 360
            explanation_text += f"<br><span style='color:hsl({hue}, 80%, 40%)'><b>Head {h+1}:</b> {patterns[h]}</span>"

        explanation_text += "<br><br><b>Explanation:</b><br>Higher peaks (lighter colors) indicate<br>stronger attention between positions.<br><br>"
        explanation_text += f"<b>Model has {num_heads} attention heads</b><br>Each with dimension {head_dim}"

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=explanation_text,
            showarrow=False,
            font=dict(size=12, family='Arial, sans-serif'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            borderpad=8,
            align="left"
        )

        return fig

    except Exception as e:
        # Create a simple error visualization
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"Error generating multi-head attention visualization: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="crimson")
        )
        fig.update_layout(
            title="Visualization Error",
            width=900,
            height=500
        )
        st.error(f"Multi-head attention visualization error: {str(e)}")
        return fig


def main():
    """Main application function."""
    # Setup environment and get device
    device = setup_environment()
    st.write(f"Using device: {device}")

    # Get model parameters from sidebar
    model_params = create_sidebar_controls()

    # Get stock symbol and date range
    stockSymbol, start_date, training_end_date, end_date, backtest_start_date, backtest_end_date = get_stock_inputs()
    if not stockSymbol:
        return

    # Select indicators
    selected_indicators, use_volume, use_news = select_indicators()

    # Load stock data for both training and backtesting periods
    stock_data, close_scaler = load_data(stockSymbol, start_date, end_date)

    if stock_data.empty:
        st.error(f"No data found for symbol {stockSymbol}. Please check the symbol and try again.")
        return

    # Display the loaded data information
    st.write(f"Loaded {len(stock_data)} days of stock data from {stock_data.index.min().date()} to {stock_data.index.max().date()}")

    # Find the index corresponding to the training end date
    training_end_idx = None
    if training_end_date:
        # Convert training_end_date to datetime for comparison
        training_end_datetime = pd.Timestamp(training_end_date)
        # Find the closest date in the index that's less than or equal to training_end_date
        training_end_idx = stock_data.index.get_indexer([training_end_datetime], method='ffill')[0]

    # Check if the backtesting date range is within the loaded data
    backtest_start = pd.Timestamp(backtest_start_date)
    backtest_end = pd.Timestamp(backtest_end_date)

    if backtest_start < stock_data.index.min() or backtest_end > stock_data.index.max():
        st.warning(f"Backtesting date range ({backtest_start_date} to {backtest_end_date}) is outside the loaded data range. Some data may be missing.")

    # Visualize the data split
    st.subheader("Data Split Visualization")

    # Create a period column that shows Training, Gap, and Backtesting periods
    period_column = []
    for date in stock_data.index:
        if date <= pd.Timestamp(training_end_date):
            period_column.append('Training')
        elif date >= backtest_start and date <= backtest_end:
            period_column.append('Backtesting')
        else:
            period_column.append('Not Used')

    split_df = pd.DataFrame({
        'Period': period_column,
        'Close': stock_data['Close'].values.flatten()
    }, index=stock_data.index)

    # Count days in each period
    training_days = sum(1 for p in period_column if p == 'Training')
    backtesting_days = sum(1 for p in period_column if p == 'Backtesting')
    not_used_days = sum(1 for p in period_column if p == 'Not Used')

    st.line_chart(split_df.pivot(columns='Period', values='Close'))

    st.info(f"Training data: {training_days} days ({training_days/len(stock_data)*100:.1f}% of data)")
    st.info(f"Backtesting data: {backtesting_days} days ({backtesting_days/len(stock_data)*100:.1f}% of data)")
    if not_used_days > 0:
        st.info(f"Data not used: {not_used_days} days ({not_used_days/len(stock_data)*100:.1f}% of data)")

    # Apply selected indicators to stock data
    for indicator in selected_indicators:
        if indicator['function'] is not None:
            try:
                if indicator['key'] in ['volume', 'news']:
                    continue  # These are handled separately

                # Determine the correct input data
                if indicator['function_input'] == 'dataframe':
                    input_data = stock_data
                elif indicator['function_input'] == 'series':
                    input_data = stock_data['Close']
                else:
                    input_data = stock_data['Close']  # Default to Close price

                # Apply the indicator function
                result = indicator['function'](input_data, **indicator['params'])

                # Add the result to stock_data
                if isinstance(result, pd.DataFrame):
                    stock_data = pd.concat([stock_data, result], axis=1)
                elif isinstance(result, pd.Series):
                    stock_data[result.name] = result
                else:
                    st.warning(f"Unexpected result type for indicator {indicator['name']}. Skipping.")
            except Exception as e:
                logger.error(f"Error applying indicator {indicator['name']}: {str(e)}", exc_info=True)
                st.error(f"Error applying indicator {indicator['name']}: {str(e)}")
                st.warning(f"Skipping indicator {indicator['name']} due to error.")

    # Fetch news data if needed
    news_data = None
    if use_news:
        with st.spinner("Fetching news data... This may take a while for long date ranges."):
            try:
                news_data = fetch_all_news(stockSymbol, stock_data.index)
                st.success(f"Successfully fetched {len(news_data)} news items")
            except Exception as e:
                st.error(f"Error fetching news data: {str(e)}")
                news_data = pd.DataFrame()

    # Split the data for training and testing
    train_data = stock_data.iloc[:training_end_idx].copy()

    # Extract backtesting data
    backtest_mask = (stock_data.index >= backtest_start) & (stock_data.index <= backtest_end)
    backtest_data = stock_data.loc[backtest_mask].copy() if any(backtest_mask) else None

    # Check if we have enough data for backtesting
    if backtest_data is None or len(backtest_data) < model_params['seq_length'] + 5:
        st.warning(f"Not enough data for backtesting. Need at least {model_params['seq_length'] + 5} days, but only have {0 if backtest_data is None else len(backtest_data)}.")
        backtest_data = None

    # Save backtest data to session state
    st.session_state['backtest_data'] = backtest_data

    if train_data.empty:
        st.error("No training data available. Please select a different date range.")
        return

    # Preprocess training data (this needs to be done each time)
    with st.spinner("Preprocessing training data..."):
        train_X, train_y = prepare_training_data(
            train_data, news_data, selected_indicators, use_volume, use_news,
            close_scaler, model_params['seq_length']
        )

    if train_X is None or train_X.shape[0] == 0:
        st.error("Failed to create training sequences. Please try different parameters or a larger date range.")
        return

    # Create and train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_length = model_params['seq_length']

    # Determine the number of features from the data
    input_dim = train_X.shape[2]

    # Create the model
    model = TransformerModel(
        input_dim=input_dim,
        hidden_dim=model_params['hidden_dim'],
        output_dim=1,
        num_layers=model_params['num_layers'],
        num_heads=model_params['num_heads'],
        dropout=model_params['dropout']
    ).to(device)

    # Display model information
    num_parameters = count_parameters(model)
    display_model_information(model, num_parameters)

    # Use a separate set for validation during training (20% of training data)
    train_size = int(0.8 * train_X.shape[0])
    val_X = train_X[train_size:]
    val_y = train_y[train_size:]
    train_X = train_X[:train_size]
    train_y = train_y[:train_size]

    # Training section
    st.subheader("Model Training")

    # Use a callback for the train button to set state
    train_button = st.button("Train Model", on_click=train_model_callback)

    # Load trained model if available
    if st.session_state['trained_model'] is not None and not train_button:
        model = st.session_state['trained_model']
        model_state_dict = st.session_state['model_state_dict']
        st.success("Using previously trained model. Click 'Train Model' if you want to retrain.")

    # Handle model training
    if train_button:
        with st.spinner(f"Training model on {len(train_X)} sequences..."):
            try:
                # Train the model
                train_losses, val_losses = train_model(
                    model, train_X, train_y, val_X, val_y, device,
                    model_params['num_epochs'], model_params['batch_size'], model_params['learning_rate']
                )

                # Plot training and validation losses
                loss_df = pd.DataFrame({
                    'Training Loss': train_losses,
                    'Validation Loss': val_losses
                })
                st.line_chart(loss_df)

                st.success(f"Model trained successfully! Final training loss: {train_losses[-1]:.6f}, validation loss: {val_losses[-1]:.6f}")

                # Add 3D visualization right after training loss display
                with st.expander("3D Neural Network Visualization", expanded=True):
                    st.subheader("Neural Network Architecture")

                    # Create tabs for different visualizations
                    viz_tab1, viz_tab2 = st.tabs(["Full Network Architecture", "Multi-Head Attention"])

                    with viz_tab1:
                        with st.spinner("Generating 3D neural network visualization..."):
                            fig = visualize_neural_network_3d(model)
                            st.plotly_chart(fig, use_container_width=True)

                            st.info("""
                            This 3D visualization shows the architecture of your transformer-based neural network:
                            - **Blue nodes**: Input layer
                            - **Green nodes**: Embedding layer
                            - **Orange nodes**: Transformer encoder layers
                            - **Red nodes**: Output layer

                            The connections between nodes represent the information flow through the network.
                            For transformer encoder layers, the visualization shows the multi-head attention mechanism,
                            where each head processes different aspects of the input data.

                            Note: For clarity, only a subset of nodes are shown for larger layers.
                            """)

                            # Add sliders to allow users to rotate and explore the 3D visualization
                            st.write("You can rotate, zoom, and pan the visualization to explore different perspectives of the neural network.")

                    with viz_tab2:
                        with st.spinner("Generating multi-head attention visualization..."):
                            attention_fig = visualize_multi_head_attention(model)
                            st.plotly_chart(attention_fig, use_container_width=True)

                            st.info("""
                            This visualization focuses on the multi-head attention mechanism:

                            **What is Multi-Head Attention?**
                            - Instead of performing a single attention function, the transformer model performs multiple attention computations in parallel.
                            - Each "head" can focus on different parts of the input sequence.
                            - This allows the model to jointly attend to information from different representation subspaces.

                            **How to interpret this visualization:**
                            - Each surface represents one attention head
                            - The height (z-axis) shows the attention weights
                            - Higher values indicate stronger attention between positions
                            - Different colors represent different attention heads

                            The actual weights in a trained model would show which positions are most important for prediction.
                            """)

                # Save model state
                model_state_dict = copy.deepcopy(model.state_dict())

                # Store model and its state dict in session state
                st.session_state['trained_model'] = model
                st.session_state['model_state_dict'] = model_state_dict

                # Reset backtesting data when model is retrained
                st.session_state['backtest_chart_data'] = None

                # Process backtesting data if available
                if backtest_data is not None and len(backtest_data) > model_params['seq_length']:
                    with st.spinner("Preprocessing backtesting data..."):
                        backtest_X, backtest_y = prepare_training_data(
                            backtest_data, news_data, selected_indicators, use_volume, use_news,
                            close_scaler, model_params['seq_length']
                        )

                    if backtest_X is not None and backtest_X.shape[0] > 0:
                        try:
                            # Ensure shapes match
                            if backtest_y.shape[0] != backtest_X.shape[0]:
                                min_len = min(backtest_X.shape[0], backtest_y.shape[0])
                                backtest_X = backtest_X[:min_len]
                                backtest_y = backtest_y[:min_len]

                            # Evaluate on backtesting data
                            backtest_chart_data = evaluate_model(model, backtest_X, backtest_y, close_scaler, device)
                            st.session_state['backtest_chart_data'] = backtest_chart_data.copy()
                        except Exception as e:
                            st.error(f"Error preprocessing backtesting data: {str(e)}")
                    else:
                        st.warning("Insufficient backtesting data after preprocessing. Skipping backtesting.")
                        backtest_chart_data = None
                else:
                    # Use validation data if no backtesting data
                    test_chart_data = evaluate_model(model, val_X, val_y, close_scaler, device)
                    st.session_state['backtest_chart_data'] = test_chart_data.copy()

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.session_state['model_trained'] = False
                return

    # Always show backtesting section if we have a trained model
    if st.session_state['trained_model'] is not None:
        if backtest_data is not None and st.session_state['backtest_chart_data'] is not None:
            # Adjust dates for backtesting
            backtest_start_date = backtest_data.index[model_params['seq_length']]
            backtest_end_date = backtest_data.index[-1]

            # Run backtest with stored chart data
            backtest_results = backtest_strategy(
                stock_data,
                st.session_state['backtest_chart_data'],
                backtest_start_date,
                backtest_end_date
            )
        elif st.session_state['backtest_chart_data'] is not None:
            # Use validation data dates
            train_end_plus_seq = train_data.index[train_size + model_params['seq_length']] if train_size + model_params['seq_length'] < len(train_data.index) else train_data.index[-1]
            backtest_results = backtest_strategy(stock_data, st.session_state['backtest_chart_data'], train_end_plus_seq, train_data.index[-1])

        # Only show future predictions if we have a trained model
        if st.session_state['trained_model'] is not None:
            predict_future_prices(st.session_state['trained_model'], stock_data, close_scaler, model_params, device)

            # Show download model button if we have trained a model
            if st.session_state['model_state_dict'] is not None:
                save_and_download_model(st.session_state['model_state_dict'])


if __name__ == "__main__":
    main()


