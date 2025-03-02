"""
Live Trading Module for the Stock Prediction Application.

This module handles the live trading process using the Alpaca API, either
with paper trading or real money, based on model predictions.
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import torch
import plotly.graph_objects as go
import plotly.express as px

# Import from our own modules
from trading import TradingEngine, trend_following_strategy, mean_reversion_strategy, combined_strategy
from data_utils import fetch_stock_data, preprocess_data
import model as model_module

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('live_trading')

# Load environment variables
load_dotenv()


def initialize_live_trading():
    """Initialize the live trading interface in Streamlit."""
    st.title("Live Trading with Alpaca")

    # Trading mode selection
    trading_mode = st.radio(
        "Trading Mode",
        options=["Paper Trading", "Live Trading"],
        index=0,
        horizontal=True
    )

    if trading_mode == "Live Trading":
        st.warning("⚠️ You are in LIVE TRADING mode. Real money will be used for trades.")
    else:
        st.info("📝 You are in PAPER TRADING mode. No real money will be used.")

    # Initialize trading engine based on mode
    mode = 'paper' if trading_mode == "Paper Trading" else 'live'

    try:
        engine = TradingEngine(mode=mode)
        account_info = engine.get_account_summary()

        if account_info:
            # Display account information
            st.subheader("Account Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Equity", f"${account_info['equity']:.2f}")
            col2.metric("Cash", f"${account_info['cash']:.2f}")
            col3.metric("Buying Power", f"${account_info['buying_power']:.2f}")

            # Display current positions
            positions = engine.get_positions()
            if positions:
                st.subheader("Current Positions")
                positions_df = pd.DataFrame(positions)
                # Format as a styled table
                st.dataframe(
                    positions_df,
                    column_config={
                        "unrealized_pl": st.column_config.NumberColumn(
                            "Unrealized P/L",
                            format="$%.2f",
                        ),
                        "unrealized_plpc": st.column_config.NumberColumn(
                            "P/L %",
                            format="%.2f%%",
                        ),
                        "current_price": st.column_config.NumberColumn(
                            "Current Price",
                            format="$%.2f",
                        ),
                        "avg_entry_price": st.column_config.NumberColumn(
                            "Entry Price",
                            format="$%.2f",
                        ),
                        "market_value": st.column_config.NumberColumn(
                            "Market Value",
                            format="$%.2f",
                        ),
                    }
                )
            else:
                st.info("No open positions")

            # Trading controls
            st.subheader("Trading Controls")

            # Close all positions button
            if st.button("Close All Positions"):
                with st.spinner("Closing all positions..."):
                    result = engine.close_all_positions()
                    if result:
                        st.success("All positions closed successfully")
                    else:
                        st.error("Failed to close positions")

            # Strategy configuration
            st.subheader("Trading Strategy Configuration")

            col1, col2 = st.columns(2)
            with col1:
                symbol = st.text_input("Stock Symbol", value="AAPL").upper()

                strategy_type = st.selectbox(
                    "Strategy Type",
                    options=["Trend Following", "Mean Reversion", "Combined Strategy"],
                    index=0
                )

            with col2:
                risk_per_trade = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100
                max_drawdown = st.slider("Max Drawdown Limit (%)", min_value=5.0, max_value=50.0, value=25.0, step=5.0) / 100

            # Strategy-specific parameters
            if strategy_type == "Trend Following":
                threshold = st.slider("Price Change Threshold (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1) / 100
            elif strategy_type == "Mean Reversion":
                col1, col2 = st.columns(2)
                with col1:
                    lookback = st.slider("Lookback Period (days)", min_value=1, max_value=60, value=20, step=1)
                with col2:
                    std_dev = st.slider("Standard Deviation Threshold", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
            elif strategy_type == "Combined Strategy":
                col1, col2 = st.columns(2)
                with col1:
                    trend_threshold = st.slider("Trend Threshold (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1) / 100
                    mr_lookback = st.slider("MR Lookback (days)", min_value=1, max_value=60, value=20, step=1)
                with col2:
                    mr_std_dev = st.slider("MR Std Dev", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
                    weight_trend = st.slider("Trend Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

            # Model selection for predictions
            st.subheader("Model Selection")

            model_path = st.text_input("Model Path", value="tesla_prediction_model_07-08-2024_17-06-39____.pt")

            if not os.path.exists(model_path):
                st.error(f"Model file {model_path} not found. Please provide a valid model path.")
            else:
                # Load model for predictions
                try:
                    model_state_dict = torch.load(model_path)
                    model_params = model_state_dict.get('model_params', {})

                    # Get input_size from model_params or use default
                    input_size = model_params.get('input_size', 1)
                    hidden_size = model_params.get('hidden_size', 128)
                    num_layers = model_params.get('num_layers', 2)
                    output_size = model_params.get('output_size', 1)
                    num_heads = model_params.get('num_heads', 8)

                    # Create and load model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model_module.TransformerModel(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        output_size=output_size,
                        num_heads=num_heads
                    )
                    model.load_state_dict(model_state_dict['model_state'])
                    model.to(device)
                    model.eval()

                    st.success(f"Model loaded successfully: {model_path}")

                    # Button to start live trading
                    if st.button("Start Trading"):
                        # This will trigger the live trading process
                        start_live_trading(
                            engine=engine,
                            model=model,
                            symbol=symbol,
                            strategy_type=strategy_type,
                            strategy_params={
                                'threshold': threshold if strategy_type == "Trend Following" else None,
                                'lookback': lookback if strategy_type == "Mean Reversion" else None,
                                'std_dev': std_dev if strategy_type == "Mean Reversion" else None,
                                'trend_threshold': trend_threshold if strategy_type == "Combined Strategy" else None,
                                'mr_lookback': mr_lookback if strategy_type == "Combined Strategy" else None,
                                'mr_std_dev': mr_std_dev if strategy_type == "Combined Strategy" else None,
                                'weight_trend': weight_trend if strategy_type == "Combined Strategy" else None
                            },
                            risk_per_trade=risk_per_trade,
                            max_drawdown=max_drawdown,
                            model_params=model_params
                        )
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

        else:
            st.error("Failed to retrieve account information from Alpaca")

    except Exception as e:
        st.error(f"Error initializing trading engine: {str(e)}")


def start_live_trading(engine, model, symbol, strategy_type, strategy_params, risk_per_trade, max_drawdown, model_params):
    """
    Start the live trading process with the configured settings.

    Args:
        engine: Trading engine instance
        model: Loaded prediction model
        symbol: Stock symbol to trade
        strategy_type: Type of trading strategy to use
        strategy_params: Parameters for the strategy
        risk_per_trade: Risk per trade as a fraction
        max_drawdown: Maximum drawdown limit
        model_params: Model parameters
    """
    st.subheader("Live Trading Session")

    # Create placeholder for live updates
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    trades_placeholder = st.empty()

    # Initialize session state for tracking trades if not already there
    if 'live_trades' not in st.session_state:
        st.session_state.live_trades = []

    if 'last_signal' not in st.session_state:
        st.session_state.last_signal = 0

    # Update trading engine settings
    engine.risk_per_trade = risk_per_trade
    engine.max_drawdown_limit = max_drawdown

    # Select the appropriate strategy function
    if strategy_type == "Trend Following":
        strategy_func = trend_following_strategy
    elif strategy_type == "Mean Reversion":
        strategy_func = mean_reversion_strategy
    elif strategy_type == "Combined Strategy":
        strategy_func = combined_strategy
    else:
        status_placeholder.error(f"Unknown strategy type: {strategy_type}")
        return

    # Filter strategy_params to only include relevant parameters
    filtered_params = {}
    if strategy_type == "Trend Following" and 'threshold' in strategy_params:
        filtered_params['threshold'] = strategy_params['threshold']
    elif strategy_type == "Mean Reversion":
        if 'lookback' in strategy_params:
            filtered_params['lookback'] = strategy_params['lookback']
        if 'std_dev' in strategy_params:
            filtered_params['std_dev'] = strategy_params['std_dev']
    elif strategy_type == "Combined Strategy":
        for key in ['trend_threshold', 'mr_lookback', 'mr_std_dev', 'weight_trend']:
            if key in strategy_params and strategy_params[key] is not None:
                filtered_params[key] = strategy_params[key]

    # Create a stop button
    stop_button = st.button("Stop Trading")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start trading loop
    while not stop_button:
        try:
            # Fetch latest data
            status_placeholder.info(f"Fetching latest data for {symbol}...")

            # Fetch a longer history for proper feature calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Get 60 days of data

            # Fetch stock data
            stock_data = fetch_stock_data(symbol, start_date, end_date)

            if stock_data.empty:
                status_placeholder.error(f"No data available for {symbol}")
                time.sleep(60)  # Wait before retrying
                continue

            # Preprocess data for model input
            # This should be adapted based on your model's preprocessing requirements
            from sklearn.preprocessing import MinMaxScaler

            # Scale the data
            close_scaler = MinMaxScaler()
            close_scaler.fit(stock_data[['Close']])

            # Prepare the data - simplified version, adjust based on your actual preprocessing
            seq_length = 20  # Adjust based on your model

            # Get selected indicators from model_params
            selected_indicators = model_params.get('selected_indicators', [])
            use_volume = model_params.get('use_volume', False)
            use_news = model_params.get('use_news', False)

            # Prepare data for prediction
            X, y = preprocess_data(stock_data, pd.DataFrame(), selected_indicators, close_scaler, seq_length)

            if X.size == 0:
                status_placeholder.error("Insufficient data for making predictions")
                time.sleep(60)
                continue

            # Make predictions
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                predictions = model(X_tensor)
                predictions = predictions.cpu().numpy()

            # Inverse transform to get actual prices
            predicted_prices = close_scaler.inverse_transform(predictions)
            actual_prices = close_scaler.inverse_transform(y)

            # Create a dataframe for strategy
            prediction_df = pd.DataFrame({
                'Actual': actual_prices.flatten(),
                'Predicted': predicted_prices.flatten()
            }, index=stock_data.index[-len(actual_prices):])

            # Generate trading signals
            signals = strategy_func(prediction_df, **filtered_params)

            # Get the latest signal (most recent)
            latest_signal = signals.iloc[-1] if not signals.empty else 0

            # Update chart with predictions and actual prices
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prediction_df.index,
                y=prediction_df['Actual'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=prediction_df.index,
                y=prediction_df['Predicted'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='red')
            ))

            # Add buy/sell markers based on signals
            buy_signals = prediction_df[signals == 1]
            sell_signals = prediction_df[signals == -1]

            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Actual'],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Buy Signal'
                ))

            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Actual'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Sell Signal'
                ))

            fig.update_layout(
                title=f'{symbol} - Price and Predictions',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )

            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Execute trade if signal has changed
            if latest_signal != 0 and latest_signal != st.session_state.last_signal:
                # Only trade if we have a new signal that's different from the last one
                status_placeholder.info(f"Executing {'BUY' if latest_signal > 0 else 'SELL'} order for {symbol}")

                # Get current price
                current_price = stock_data['Close'].iloc[-1]

                # Execute the trade
                if engine.mode == 'paper':
                    order = engine.paper_trade(symbol, latest_signal, price=current_price)
                else:
                    order = engine.live_trade(symbol, latest_signal, price=current_price)

                if order:
                    # Record the trade
                    trade_info = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY' if latest_signal > 0 else 'SELL',
                        'price': current_price,
                        'signal': latest_signal,
                        'order_id': order.id if hasattr(order, 'id') else None
                    }

                    st.session_state.live_trades.append(trade_info)
                    status_placeholder.success(f"Order executed: {trade_info['action']} {symbol} at ${current_price:.2f}")
                else:
                    status_placeholder.error(f"Failed to execute {'BUY' if latest_signal > 0 else 'SELL'} order")

                # Update last signal
                st.session_state.last_signal = latest_signal
            else:
                if latest_signal == 0:
                    status_placeholder.info(f"No trading signal for {symbol}")
                else:
                    status_placeholder.info(f"Signal unchanged ({latest_signal}), no new order")

            # Display trade history
            if st.session_state.live_trades:
                trades_df = pd.DataFrame(st.session_state.live_trades)
                trades_placeholder.subheader("Trade History")
                trades_placeholder.dataframe(trades_df, use_container_width=True)

            # Check if stop button has been pressed
            if stop_button:
                status_placeholder.warning("Trading stopped by user")
                break

            # Wait before next update
            time.sleep(300)  # 5 minutes between updates

            # Refresh the stop button (this will re-render the page)
            stop_button = st.button("Stop Trading", key=f"stop_{time.time()}")

        except Exception as e:
            status_placeholder.error(f"Error in trading loop: {str(e)}")
            logger.error(f"Error in trading loop: {str(e)}", exc_info=True)
            time.sleep(60)  # Wait before retrying

    status_placeholder.warning("Trading session ended")


def main():
    """Main function to run when this script is executed directly."""
    st.set_page_config(
        page_title="Stock Trading Bot",
        page_icon="📈",
        layout="wide"
    )

    initialize_live_trading()


if __name__ == "__main__":
    main()
