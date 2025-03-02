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

    # Add a separator for training end date (which is also the backtest start date)
    st.subheader("Data Separation")
    st.info("To prevent data leakage, separate your data into training and backtesting periods.")

    # Calculate a default training end date (60% of the data)
    default_training_end = start_date + timedelta(days=int((end_date - start_date).days * 0.6))

    training_end_date = st.date_input(
        "Training End Date (Backtest Start Date)",
        default_training_end,
        help="Data before this date will be used for training, data after will be used for backtesting."
    )

    # Validate inputs
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return None, None, None, None

    if start_date >= training_end_date or training_end_date >= end_date:
        st.error("Training end date must be between start date and end date")
        return None, None, None, None

    if not stockSymbol:
        st.error("Please enter a stock symbol")
        return None, None, None, None

    return stockSymbol, start_date, training_end_date, end_date


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


def prepare_training_data(stock_data, news_data, selected_indicators, use_volume, use_news, close_scaler, seq_length=20, training_end_idx=None):
    """Prepare the data for model training and evaluation."""
    # If no data is available, return empty arrays
    if stock_data.empty:
        return (np.array([]), np.array([]),
                np.array([]), np.array([]),
                pd.DataFrame(), None, pd.DataFrame())

    # Drop rows with NaN values
    stock_data.dropna(inplace=True)

    # Split into train and test sets based on the training_end_idx
    if training_end_idx is not None:
        train_data = stock_data[:training_end_idx]
        test_data = stock_data[training_end_idx:]
    else:
        # Default split if no specific index is provided (80/20)
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

    # Return backtest_data separately for backtesting
    backtest_data = stock_data[training_end_idx:] if training_end_idx is not None else None

    return train_X, train_y, test_X, test_y, test_data, input_size, backtest_data


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
    from trading import TradingEngine, trend_following_strategy, mean_reversion_strategy, combined_strategy
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime

    st.subheader("Strategy Backtesting")

    if test_chart_data is None or test_chart_data.empty:
        st.warning("No test data available for backtesting.")
        return None

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
                options=["Trend Following", "Mean Reversion", "Combined Strategy", "Buy and Hold"],
                index=0
            )

        # Risk management settings
        col1, col2 = st.columns(2)
        with col1:
            risk_per_trade = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100

        with col2:
            max_drawdown_limit = st.slider("Max Drawdown Limit (%)", min_value=5.0, max_value=50.0, value=25.0, step=5.0) / 100

        # Additional strategy parameters
        if strategy_type == "Trend Following":
            threshold = st.slider("Price Change Threshold (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1) / 100
            strategy_params = {'threshold': threshold}
            strategy_func = trend_following_strategy

        elif strategy_type == "Mean Reversion":
            col1, col2 = st.columns(2)
            with col1:
                lookback = st.slider("Mean Lookback Period (days)", min_value=1, max_value=60, value=20, step=1)
            with col2:
                std_dev = st.slider("Standard Deviation Threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
            strategy_params = {'lookback': lookback, 'std_dev': std_dev}
            strategy_func = mean_reversion_strategy

        elif strategy_type == "Combined Strategy":
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Trend Threshold (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1) / 100
                lookback = st.slider("MR Lookback Period (days)", min_value=1, max_value=60, value=20, step=1)
            with col2:
                std_dev = st.slider("MR Std Dev Threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
                weight_trend = st.slider("Trend Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
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

    # Initialize trading engine with settings
    trading_engine = TradingEngine(
        mode='backtest',
        initial_capital=initial_capital,
        commission=commission,
        risk_per_trade=risk_per_trade,
        max_drawdown_limit=max_drawdown_limit
    )

    # Make sure test_chart_data has proper index
    if not isinstance(test_chart_data.index, pd.DatetimeIndex):
        try:
            # Try to convert index to datetime if it's not already
            test_chart_data.index = pd.to_datetime(test_chart_data.index)
            st.info(f"Converted data index to datetime. Date range: {test_chart_data.index[0]} to {test_chart_data.index[-1]}")
        except:
            # If conversion fails, create a new datetime index
            st.warning("Data index is not datetime format. Creating artificial dates for backtesting.")
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=len(test_chart_data))
            test_chart_data.index = pd.date_range(start=start_date, periods=len(test_chart_data), freq='D')
            st.info(f"Created date range from {test_chart_data.index[0]} to {test_chart_data.index[-1]}")

    # Check for epoch dates (1970-01-01) in the index
    if (test_chart_data.index.year == 1970).any():
        st.warning("Detected epoch dates (1970-01-01) in data index. Creating new date range.")
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=len(test_chart_data))
        test_chart_data.index = pd.date_range(start=start_date, periods=len(test_chart_data), freq='D')
        st.info(f"Created new date range from {test_chart_data.index[0]} to {test_chart_data.index[-1]}")

    # Run backtesting with progress indicator
    with st.spinner("Running backtesting simulation..."):
        try:
            backtest_results, trades, metrics = trading_engine.backtest(
                strategy=strategy_func,
                data=test_chart_data,
                strategy_params=strategy_params
            )
        except Exception as e:
            st.error(f"Backtesting failed with error: {str(e)}")
            return None

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
            # Debug information
            st.write(f"Trade dates range: {trades['date'].min()} to {trades['date'].max()}")
            st.write(f"Number of unique trade dates: {trades['date'].nunique()} out of {len(trades)} trades")

            # Ensure date column is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(trades['date']):
                try:
                    trades['date'] = pd.to_datetime(trades['date'])
                except Exception as e:
                    st.error(f"Error converting dates: {str(e)}")

            # Check if we still have epoch dates (1970-01-01)
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

            # Ensure trades are sorted by date
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
    stockSymbol, start_date, training_end_date, end_date = get_stock_inputs()
    if not stockSymbol:
        return

    # Select indicators
    selected_indicators, use_volume, use_news = select_indicators()

    # Load stock data
    stock_data, close_scaler = load_data(stockSymbol, start_date, end_date)

    if stock_data.empty:
        st.error(f"No data found for symbol {stockSymbol}. Please check the symbol and try again.")
        return

    # Find the index corresponding to the training end date
    training_end_idx = None
    if training_end_date:
        # Convert training_end_date to datetime for comparison
        training_end_datetime = pd.Timestamp(training_end_date)
        # Find the closest date in the index that's less than or equal to training_end_date
        training_end_idx = stock_data.index.get_indexer([training_end_datetime], method='ffill')[0]

        # Visualize the data split
        st.subheader("Data Split Visualization")
        split_df = pd.DataFrame({
            'Period': ['Training'] * training_end_idx + ['Backtesting'] * (len(stock_data) - training_end_idx),
            'Close': stock_data['Close'].values.flatten()  # Flatten the 2D array to 1D
        }, index=stock_data.index)

        st.line_chart(split_df.pivot(columns='Period', values='Close'))

        st.info(f"Training data: {training_end_idx} days ({training_end_idx/len(stock_data)*100:.1f}% of data)")
        st.info(f"Backtesting data: {len(stock_data) - training_end_idx} days ({(len(stock_data) - training_end_idx)/len(stock_data)*100:.1f}% of data)")

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

    # Prepare data for training with the training/backtesting split
    train_X, train_y, test_X, test_y, test_data, input_size, backtest_data = prepare_training_data(
        stock_data, news_data, selected_indicators, use_volume, use_news, close_scaler,
        training_end_idx=training_end_idx
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
        if backtest_data is not None:
            st.write(f"Backtesting data shape: {backtest_data.shape}")

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
        st.rerun()

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

    # Backtest strategy using the separate backtesting data
    if backtest_data is not None and not backtest_data.empty:
        st.subheader("Backtesting on Separate Data")
        st.info("Backtesting is performed on data that was NOT used for training to prevent data leakage.")

        # Prepare backtesting data
        backtest_X, backtest_y = preprocess_data(backtest_data, news_data, selected_indicators, close_scaler, 20)

        if backtest_X.size > 0 and backtest_y.size > 0:
            # Convert to PyTorch tensors
            backtest_X_tensor = torch.tensor(backtest_X, dtype=torch.float32).to(device)
            backtest_y_tensor = torch.tensor(backtest_y, dtype=torch.float32).to(device)

            # Run prediction on backtesting data
            model.eval()
            with torch.no_grad():
                backtest_predicted = model(backtest_X_tensor)
                backtest_predicted = backtest_predicted.cpu().numpy()

            # Prepare data for backtesting
            backtest_actual_data = pd.DataFrame(close_scaler.inverse_transform(backtest_y), columns=['Actual'])
            backtest_predicted_data = pd.DataFrame(close_scaler.inverse_transform(backtest_predicted), columns=['Predicted'])

            # Concatenate the DataFrames
            backtest_chart_data = pd.concat([backtest_actual_data.reset_index(drop=True), backtest_predicted_data], axis=1)

            # Display backtesting chart
            st.subheader("Backtesting Predictions")
            st.line_chart(backtest_chart_data)

            # Run backtesting strategy
            backtest_data = backtest_strategy(backtest_chart_data)
        else:
            st.warning("Insufficient backtesting data after preprocessing. Skipping backtesting.")
    else:
        # Fallback to using test data for backtesting if no separate backtesting data is available
        backtest_data = backtest_strategy(test_chart_data)

    # Predict future prices
    predict_future_prices(model_state_dict, test_X, test_data, close_scaler, model_params, device)

    # Save and download model
    save_and_download_model(model_state_dict)


if __name__ == "__main__":
    main()


