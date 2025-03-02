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
        page_title="Stock Prediction with Transformer Model",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state variables
    if 'stockSymbol' not in st.session_state:
        st.session_state['stockSymbol'] = 'AAPL'

    if 'stop_training' not in st.session_state:
        st.session_state['stop_training'] = False

    # Determine the compute device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) for computation")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA for computation")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for computation")

    return device


def create_sidebar_controls():
    """Create and display the sidebar controls for model parameters."""
    st.sidebar.header("Model Parameters")

    # Model architecture parameters
    st.sidebar.subheader("Architecture")
    hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=512, value=128, step=32)
    num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=6, value=2, step=1)
    num_heads = st.sidebar.slider("Number of Heads", min_value=1, max_value=16, value=8, step=1)
    dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

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
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }


def get_stock_inputs():
    """Get user inputs for stock symbol and date range."""
    st.title("Stock Price Prediction with Transformer Model")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        stockSymbol = st.text_input("Enter Stock Symbol", st.session_state['stockSymbol'])

    with col2:
        start_date = st.date_input("Start Date", datetime(2020, 1, 1))

    with col3:
        end_date = st.date_input("End Date", datetime.now())

    # Validate inputs
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return None, None, None

    if not stockSymbol:
        st.error("Please enter a stock symbol")
        return None, None, None

    return stockSymbol, start_date, end_date


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


def prepare_training_data(stock_data, news_data, selected_indicators, use_volume, use_news, close_scaler, seq_length=20):
    """Prepare the data for model training and evaluation."""
    # If no data is available, return empty arrays
    if stock_data.empty:
        return (np.array([]), np.array([]),
                np.array([]), np.array([]),
                pd.DataFrame(), None)

    # Drop rows with NaN values
    stock_data.dropna(inplace=True)

    # Split into train and test sets
    train_size = int(len(stock_data) * 0.8)
    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:]

    # Preprocess data
    train_X, train_y = preprocess_data(train_data, news_data, selected_indicators, close_scaler, seq_length)
    test_X, test_y = preprocess_data(test_data, news_data, selected_indicators, close_scaler, seq_length)

    # Calculate input size
    input_size = 1  # Always include 'Normalized_Close'
    if use_volume:
        input_size += 1  # 'Normalized_Volume'
    if use_news:
        input_size += 1  # 'News sentiment'
    for indicator in selected_indicators:
        if indicator['key'] not in ['volume', 'news']:
            input_size += indicator['size']

    return train_X, train_y, test_X, test_y, test_data, input_size


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
        st.markdown(f"**Input Size:** {model.input_size}")
        st.markdown(f"**Hidden Size:** {model.hidden_size}")
        st.markdown(f"**Layers:** {len(model.transformer_encoder.layers)}")
        st.markdown(f"**Attention Heads:** {model.num_heads}")


def evaluate_model(model, test_X, test_y, close_scaler, device):
    """Evaluate the model on test data and display results."""
    st.subheader("Model Evaluation")

    # Check if test data is available
    if test_X.size == 0 or test_y.size == 0:
        st.warning("Insufficient data for testing predictions.")
        return

    # Convert to PyTorch tensors
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(device)

    # Run prediction
    model.eval()
    with torch.no_grad():
        test_predicted = model(test_X_tensor)
        test_predicted = test_predicted.cpu().numpy()

    # Check for NaNs in predictions
    if np.isnan(test_predicted).any():
        st.error("NaN values found in model predictions. Check model training and data preprocessing.")
        return

    # Prepare data for visualization
    test_actual_data = pd.DataFrame(close_scaler.inverse_transform(test_y), columns=['Actual'])
    test_predicted_data = pd.DataFrame(close_scaler.inverse_transform(test_predicted), columns=['Predicted'])

    # Concatenate the DataFrames
    test_chart_data = pd.concat([test_actual_data.reset_index(drop=True), test_predicted_data], axis=1)

    # Calculate evaluation metrics
    mse = ((test_chart_data['Actual'] - test_chart_data['Predicted']) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = abs(test_chart_data['Actual'] - test_chart_data['Predicted']).mean()

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    col3.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")

    # Visualize results
    st.subheader("Test Set Predictions")
    st.line_chart(test_chart_data)

    # Return the test chart data for potential backtesting
    return test_chart_data


def backtest_strategy(test_chart_data, initial_capital=10000, commission=0.001):
    """
    Backtest a simple trading strategy based on model predictions.

    Args:
        test_chart_data: DataFrame with 'Actual' and 'Predicted' columns
        initial_capital: Starting capital for the simulation
        commission: Commission rate per trade (as a decimal)

    Returns:
        DataFrame with backtesting results
    """
    st.subheader("Strategy Backtesting")

    if test_chart_data is None or test_chart_data.empty:
        st.warning("No test data available for backtesting.")
        return

    # Backtesting controls
    with st.expander("Backtesting Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

        with col2:
            commission = st.number_input("Commission Rate (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100

        with col3:
            strategy_type = st.selectbox(
                "Strategy Type",
                options=["Trend Following", "Mean Reversion", "Buy and Hold"],
                index=0
            )

        # Additional strategy parameters
        if strategy_type == "Trend Following":
            threshold = st.slider("Price Change Threshold (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1) / 100
        elif strategy_type == "Mean Reversion":
            lookback = st.slider("Mean Lookback Period (days)", min_value=1, max_value=60, value=20, step=1)
            std_dev = st.slider("Standard Deviation Threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

    # Create a copy of the test data
    backtest_data = test_chart_data.copy()

    # Generate trading signals based on selected strategy
    backtest_data['Signal'] = 0  # 0: no action, 1: buy, -1: sell

    if strategy_type == "Trend Following":
        # Calculate predicted price changes (today's prediction vs tomorrow's prediction)
        backtest_data['PredictedChange'] = backtest_data['Predicted'].shift(-1) - backtest_data['Predicted']
        backtest_data['PredictedChangePercent'] = backtest_data['PredictedChange'] / backtest_data['Predicted']

        # Generate signals based on predicted changes and threshold
        backtest_data.loc[backtest_data['PredictedChangePercent'] > threshold, 'Signal'] = 1  # Buy signal
        backtest_data.loc[backtest_data['PredictedChangePercent'] < -threshold, 'Signal'] = -1  # Sell signal

    elif strategy_type == "Mean Reversion":
        # Calculate rolling mean and standard deviation
        backtest_data['RollingMean'] = backtest_data['Actual'].rolling(window=lookback).mean()
        backtest_data['RollingStd'] = backtest_data['Actual'].rolling(window=lookback).std()

        # Calculate z-score
        backtest_data['ZScore'] = (backtest_data['Actual'] - backtest_data['RollingMean']) / backtest_data['RollingStd']

        # Generate signals based on z-score
        backtest_data.loc[backtest_data['ZScore'] < -std_dev, 'Signal'] = 1  # Buy when price is below mean
        backtest_data.loc[backtest_data['ZScore'] > std_dev, 'Signal'] = -1  # Sell when price is above mean

    elif strategy_type == "Buy and Hold":
        # Simple buy and hold strategy
        backtest_data.loc[backtest_data.index[0], 'Signal'] = 1  # Buy on first day

    # Initialize portfolio metrics
    backtest_data['Position'] = backtest_data['Signal'].shift(1).fillna(0).cumsum()
    backtest_data['Holdings'] = backtest_data['Position'] * backtest_data['Actual']

    # Calculate costs for each trade
    backtest_data['Trade'] = backtest_data['Position'].diff()
    backtest_data['TradeCost'] = abs(backtest_data['Trade'] * backtest_data['Actual'] * commission)

    # Calculate cash and portfolio value
    backtest_data['Cash'] = initial_capital - (backtest_data['Holdings'] + backtest_data['TradeCost']).cumsum()
    backtest_data['PortfolioValue'] = backtest_data['Holdings'] + backtest_data['Cash']

    # Calculate daily returns
    backtest_data['Returns'] = backtest_data['PortfolioValue'].pct_change()

    # Calculate performance metrics
    total_trades = (backtest_data['Trade'] != 0).sum()
    profitable_trades = ((backtest_data['Trade'] != 0) & (backtest_data['Returns'] > 0)).sum()
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    final_value = backtest_data['PortfolioValue'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = backtest_data['Returns'].mean() / backtest_data['Returns'].std() * np.sqrt(252) if backtest_data['Returns'].std() > 0 else 0

    # Calculate max drawdown
    backtest_data['Cumulative_Returns'] = (1 + backtest_data['Returns']).cumprod()
    backtest_data['Cumulative_Max'] = backtest_data['Cumulative_Returns'].cummax()
    backtest_data['Drawdown'] = (backtest_data['Cumulative_Returns'] / backtest_data['Cumulative_Max']) - 1
    max_drawdown = backtest_data['Drawdown'].min() * 100

    # Display performance metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Return", f"{total_return:.2f}%")
    col2.metric("Initial Capital", f"${initial_capital:.2f}", f"${final_value - initial_capital:.2f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", f"{total_trades}")
    col2.metric("Win Rate", f"{win_rate:.2%}")
    col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")
    col2.metric("Final Portfolio Value", f"${final_value:.2f}")
    col3.metric("Strategy Type", strategy_type)

    # Compare with benchmark (buy and hold from day 1)
    if strategy_type != "Buy and Hold":
        first_price = backtest_data['Actual'].iloc[0]
        last_price = backtest_data['Actual'].iloc[-1]
        buy_hold_return = (last_price - first_price) / first_price * 100
        buy_hold_value = initial_capital * (1 + buy_hold_return/100)

        st.metric(
            "Strategy vs Buy & Hold",
            f"{total_return:.2f}% vs {buy_hold_return:.2f}%",
            f"{total_return - buy_hold_return:.2f}%"
        )

    # Visualize portfolio value over time
    st.subheader("Portfolio Value Over Time")
    st.line_chart(backtest_data[['PortfolioValue']])

    # Visualize drawdowns
    st.subheader("Drawdowns")
    st.area_chart(backtest_data[['Drawdown']])

    # Show detailed backtest data
    with st.expander("Detailed Backtest Data", expanded=False):
        st.dataframe(backtest_data)

    # Add ability to download backtest results
    csv = backtest_data.to_csv(index=False)
    st.download_button(
        label="Download Backtest Results",
        data=csv,
        file_name="backtest_results.csv",
        mime="text/csv",
    )

    return backtest_data


def predict_future_prices(model_state_dict, test_X, test_data, close_scaler, model_params, device):
    """Predict future stock prices and display results."""
    st.subheader("Future Price Predictions")

    # Get number of days to predict
    num_days = st.number_input("Number of days to predict", min_value=1, max_value=30, value=7)

    # Predict button
    if st.button("Predict Future"):
        if test_X.size == 0:
            st.error("Insufficient data for making predictions.")
            return

        # Get last observed sequence
        last_sequence = torch.tensor(test_X[-1], dtype=torch.float32).to(device)

        # Make prediction
        future_predictions = predict_future(
            model_state_dict,
            last_sequence,
            num_days,
            close_scaler,
            model_params['input_size'],
            model_params['hidden_size'],
            model_params['num_layers'],
            model_params['num_heads'],
            model_params['dropout'],
            device
        )

        if future_predictions is None:
            st.error("Failed to generate future predictions.")
            return

        # Create DataFrame for the future predictions
        future_dates = pd.date_range(
            start=test_data.index[-1] + timedelta(days=1),
            periods=num_days
        )
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions
        })
        future_df['Date'] = pd.to_datetime(future_df['Date'])

        # Combine actual data with future predictions for visualization
        last_actual_price = test_data['Close'].iloc[-1]
        combined_df = pd.concat([
            test_data[['Close']].tail(30).rename(columns={'Close': 'Actual Price'}),
            future_df.set_index('Date')
        ])

        # Display predictions
        st.write("Future Price Predictions:")
        st.dataframe(future_df, hide_index=True)

        # Visualize predictions
        st.subheader("Actual vs Predicted Prices")
        st.line_chart(combined_df)

        # Calculate and display the predicted price change
        price_change = float(future_predictions[-1] - last_actual_price)
        percent_change = (price_change / float(last_actual_price)) * 100

        # Display price change metrics
        col1, col2 = st.columns(2)
        col1.metric(
            f"Predicted price change over {num_days} days",
            f"${price_change:.2f}",
            f"{percent_change:.2f}%",
            delta_color="normal"
        )
        col2.metric(
            "Predicted final price",
            f"${float(future_predictions[-1]):.2f}"
        )


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


def main():
    """Main application function."""
    # Setup environment and get device
    device = setup_environment()
    st.write(f"Using device: {device}")

    # Get model parameters from sidebar
    model_params = create_sidebar_controls()

    # Get stock symbol and date range
    stockSymbol, start_date, end_date = get_stock_inputs()
    if not stockSymbol:
        return

    # Select indicators
    selected_indicators, use_volume, use_news = select_indicators()

    # Load stock data
    stock_data, close_scaler = load_data(stockSymbol, start_date, end_date)

    if stock_data.empty:
        st.error(f"No data found for symbol {stockSymbol}. Please check the symbol and try again.")
        return

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

    # Fetch and prepare news data if needed
    stock_dates = stock_data.index.normalize()
    news_data = fetch_all_news(stockSymbol, stock_dates) if use_news else pd.DataFrame()

    # Prepare data for training
    train_X, train_y, test_X, test_y, test_data, input_size = prepare_training_data(
        stock_data, news_data, selected_indicators, use_volume, use_news, close_scaler
    )

    # Add input size to model parameters
    model_params['input_size'] = input_size

    # Display information about the data
    with st.expander("Data Information", expanded=False):
        st.write("Columns in training data after applying indicators:")
        st.write(stock_data.columns.tolist())
        st.write(f"Input size: {input_size}")
        st.write(f"Training data shape: {train_X.shape if train_X.size > 0 else 'No data'}")
        st.write(f"Testing data shape: {test_X.shape if test_X.size > 0 else 'No data'}")

    # Check for NaNs in data
    if train_X.size == 0 or train_y.size == 0 or test_X.size == 0 or test_y.size == 0:
        st.error("No data available after preprocessing. Please adjust your indicators or data range.")
        return
    elif np.isnan(train_X).any() or np.isnan(train_y).any() or np.isnan(test_X).any() or np.isnan(test_y).any():
        st.error("NaN values found in the data. Please check your indicators and data preprocessing.")
        return

    # Store training parameters in a dictionary for caching
    training_params = {
        'input_size': input_size,
        'hidden_size': model_params['hidden_size'],
        'num_layers': model_params['num_layers'],
        'num_heads': model_params['num_heads'],
        'dropout': model_params['dropout'],
        'num_epochs': model_params['num_epochs'],
        'batch_size': model_params['batch_size'],
        'learning_rate': model_params['learning_rate'],
        'train_X_shape': train_X.shape,
        'train_y_shape': train_y.shape,
        'stockSymbol': stockSymbol,
        'start_date': start_date,
        'end_date': end_date,
        'selected_indicators': [indicator['key'] for indicator in selected_indicators],
    }

    # Train the model only if necessary
    if 'model_state_dict' not in st.session_state or st.session_state.get('training_params') != training_params:
        # Check if we have data to train
        if train_X.size == 0 or train_y.size == 0:
            st.error("No data available for training. Please select at least one indicator.")
            return

        # Reset stop training flag
        st.session_state['stop_training'] = False

        # Train the model
        with st.spinner("Training model..."):
            model_state_dict = train_model(
                train_X, train_y, test_X, test_y,
                input_size, model_params['hidden_size'], model_params['num_layers'], model_params['num_heads'],
                model_params['dropout'], model_params['num_epochs'], model_params['batch_size'],
                model_params['learning_rate'], device
            )

        # Save model state if training successful
        if model_state_dict is not None:
            st.session_state['model_state_dict'] = model_state_dict
            st.session_state['training_params'] = training_params
            st.success("Model training completed successfully!")
        else:
            st.error("Model training failed. Please check the logs for details.")
            return
    else:
        model_state_dict = st.session_state['model_state_dict']

    # Create model instance for evaluation and prediction
    model = TransformerModel(
        input_size=input_size,
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        num_heads=model_params['num_heads'],
        dropout=model_params['dropout']
    )
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # Check if the stock symbol has changed
    if stockSymbol != st.session_state['stockSymbol']:
        st.session_state['stockSymbol'] = stockSymbol
        st.session_state['stop_training'] = False
        st.experimental_rerun()

    # Display model information
    num_parameters = count_parameters(model)
    display_model_information(model, num_parameters)

    # Display news data if selected
    if use_news:
        with st.expander("News Data", expanded=False):
            st.subheader("Latest News Headlines")
            st.write(f"Fetched {len(news_data)} news articles for {stockSymbol}")
            if not news_data.empty:
                st.dataframe(news_data)

    # Evaluate model on test data
    test_chart_data = evaluate_model(model, test_X, test_y, close_scaler, device)

    # Backtest strategy
    backtest_data = backtest_strategy(test_chart_data)

    # Predict future prices
    predict_future_prices(model_state_dict, test_X, test_data, close_scaler, model_params, device)

    # Save and download model
    save_and_download_model(model_state_dict)


if __name__ == "__main__":
    main()
