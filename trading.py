"""
Trading module for the Stock Prediction Application.

This module provides functionality for backtesting and live trading using Alpaca API.
It implements realistic trading strategies with proper risk management.
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading')

# Initialize Alpaca API
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

class TradingEngine:
    """
    Trading engine that supports both backtesting and live trading with Alpaca API.
    """

    def __init__(self, mode='backtest', initial_capital=10000, commission=0.0025,
                 risk_per_trade=0.02, max_drawdown_limit=0.25):
        """
        Initialize the trading engine.

        Args:
            mode: 'backtest', 'paper', or 'live'
            initial_capital: Starting capital
            commission: Commission per trade (percentage as decimal)
            risk_per_trade: Maximum risk per trade as fraction of portfolio
            max_drawdown_limit: Maximum allowed drawdown before stopping trades
        """
        self.mode = mode
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_limit = max_drawdown_limit
        self.positions = {}
        self.orders = []
        self.portfolio_history = []
        self.max_portfolio_value = initial_capital

        # Initialize Alpaca clients
        if mode in ['paper', 'live']:
            if mode == 'paper':
                api_key = os.getenv('ALPACA_PAPER_KEY_API_ID')
                api_secret = os.getenv('ALPACA_PAPER_API_SECRET')
                self.base_url = 'https://paper-api.alpaca.markets'
            else:
                api_key = os.getenv('ALPACA_LIVE_KEY_ID')
                api_secret = os.getenv('ALPACA_LIVE_SECRET')
                self.base_url = 'https://api.alpaca.markets'

            self.trading_client = TradingClient(api_key, api_secret, paper=mode == 'paper')
            self.data_client = StockHistoricalDataClient(api_key, api_secret)
            logger.info(f"Initialized Alpaca client in {mode} mode")

            # Check account info
            self.account = self.trading_client.get_account()
            logger.info(f"Account status: {self.account.status}")
            logger.info(f"Current balance: ${float(self.account.equity)}")
            logger.info(f"Cash available: ${float(self.account.cash)}")

            if mode == 'live':
                if self.account.trading_blocked or not self.account.account_blocked:
                    logger.warning("Account is blocked for trading. Please resolve before continuing.")

    def calculate_position_size(self, price, stop_loss, portfolio_value):
        """
        Calculate position size based on risk per trade.

        Args:
            price: Current price of the asset
            stop_loss: Stop loss price
            portfolio_value: Current portfolio value

        Returns:
            Number of shares to buy/sell
        """
        if price <= stop_loss:
            logger.warning("Stop loss must be lower than current price for long positions.")
            return 0

        # Calculate risk amount based on portfolio value
        risk_amount = portfolio_value * self.risk_per_trade

        # Calculate risk per share
        risk_per_share = price - stop_loss

        # Calculate position size based on risk
        if risk_per_share <= 0:
            return 1  # Minimum position size

        position_size = risk_amount / risk_per_share

        # Round down to whole number of shares
        return max(int(position_size), 1)  # Ensure at least 1 share

    def backtest(self, strategy, data, strategy_params=None):
        """
        Run backtesting simulation.

        Args:
            strategy: Trading strategy function
            data: DataFrame with 'Actual' and 'Predicted' columns
            strategy_params: Optional parameters for the strategy

        Returns:
            DataFrame with backtesting results
        """
        if self.mode != 'backtest':
            logger.warning("Backtesting is only available in 'backtest' mode.")
            return None

        if strategy_params is None:
            strategy_params = {}

        # Create a copy of the data for backtesting
        backtest_data = data.copy()

        # Ensure we have required columns
        required_columns = ['Actual', 'Predicted']
        for col in required_columns:
            if col not in backtest_data.columns:
                logger.error(f"Required column {col} not found in backtest data")
                return None, None, {
                    'Initial Capital': self.initial_capital,
                    'Final Value': self.initial_capital,
                    'Total Return': 0.0,
                    'Sharpe Ratio': 0.0,
                    'Max Drawdown': 0.0,
                    'Win Rate': 0.0,
                    'Total Trades': 0,
                    'Winning Trades': 0,
                    'Losing Trades': 0,
                    'Avg Profit Per Trade': 0.0,
                    'Avg Loss Per Trade': 0.0,
                    'Profit Factor': 0.0
                }

        # Ensure data has a proper DatetimeIndex
        if not isinstance(backtest_data.index, pd.DatetimeIndex):
            try:
                backtest_data.index = pd.to_datetime(backtest_data.index)
                logger.info(f"Converted backtest data index to datetime. Sample dates: {backtest_data.index[:3].tolist()}")
            except:
                # If conversion fails, create artificial dates - use more realistic date range
                logger.warning("Data index is not datetime format. Creating artificial dates for backtesting.")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=len(backtest_data))
                backtest_data.index = pd.date_range(start=start_date, periods=len(backtest_data), freq='D')
                logger.info(f"Created date range from {backtest_data.index[0]} to {backtest_data.index[-1]}")

        # Verify we don't have epoch dates in the index
        if (backtest_data.index.year == 1970).any():
            logger.warning("Detected epoch dates (1970-01-01) in backtest data index. Creating new date range.")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(backtest_data))
            backtest_data.index = pd.date_range(start=start_date, periods=len(backtest_data), freq='D')
            logger.info(f"Created new date range from {backtest_data.index[0]} to {backtest_data.index[-1]}")

        # Initialize portfolio and positions
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}
        trades_record = []

        # Initialize portfolio tracking columns
        backtest_data['Cash'] = float(cash)
        backtest_data['Holdings'] = 0.0
        backtest_data['PortfolioValue'] = float(portfolio_value)
        backtest_data['DailyReturn'] = 0.0

        # Generate signals from strategy
        try:
            signals = strategy(backtest_data, **strategy_params)

            # Ensure signals have the same length as backtest_data
            if len(signals) != len(backtest_data):
                logger.warning(f"Strategy returned {len(signals)} signals, but we need {len(backtest_data)}. Adjusting signals.")

                if len(signals) > len(backtest_data):
                    # Truncate signals if too long
                    signals = signals[:len(backtest_data)]
                elif len(signals) < len(backtest_data):
                    # Extend signals with zeros if too short
                    missing = len(backtest_data) - len(signals)
                    signals = pd.concat([signals, pd.Series([0] * missing)])

                # If signals is not a pandas Series with the same index, convert it
                if not isinstance(signals, pd.Series) or not signals.index.equals(backtest_data.index):
                    signals = pd.Series(signals.values, index=backtest_data.index)

            # Convert signals to a Series with the same index as backtest_data if needed
            if not isinstance(signals, pd.Series):
                signals = pd.Series(signals, index=backtest_data.index)
            elif not signals.index.equals(backtest_data.index):
                signals = pd.Series(signals.values, index=backtest_data.index)

            backtest_data['Signal'] = signals

        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}", exc_info=True)
            # Create default signals (all zeros)
            backtest_data['Signal'] = 0
            # Set first position to buy
            backtest_data.iloc[0, backtest_data.columns.get_indexer(['Signal'])[0]] = 1

        # Add price slippage model (simulated by adjusting the execution price by a small random amount)
        np.random.seed(42)  # For reproducibility
        backtest_data['Slippage'] = np.random.normal(0, 0.001, len(backtest_data)) * backtest_data['Actual']
        backtest_data['ExecutionPrice'] = backtest_data['Actual'] + backtest_data['Slippage']

        # Initialize the first row's portfolio value
        backtest_data.iloc[0, backtest_data.columns.get_indexer(['PortfolioValue'])[0]] = float(portfolio_value)
        backtest_data.iloc[0, backtest_data.columns.get_indexer(['Cash'])[0]] = float(cash)

        # Track positions and calculate returns
        for i, (idx, row) in enumerate(backtest_data.iterrows()):
            # Get current price
            price = row['ExecutionPrice']
            if pd.isna(price) or price <= 0:
                logger.warning(f"Invalid price at index {idx}: {price}, using previous valid price")
                # Use last valid price
                if i > 0:
                    price = backtest_data['ExecutionPrice'].iloc[i-1]
                else:
                    # Skip this iteration if no valid price available
                    continue

            # Calculate stop loss price (using fixed percentage for now)
            stop_loss_pct = 0.05  # 5% stop loss
            stop_loss = price * (1 - stop_loss_pct)

            symbol = 'BACKTEST'  # Placeholder for test symbol

            # Check stop-loss for existing positions
            if symbol in positions:
                position = positions[symbol]
                if price <= position['stop_loss']:
                    # Stop loss triggered
                    sale_amount = position['size'] * price
                    transaction_cost = sale_amount * self.commission
                    cash += (sale_amount - transaction_cost)

                    # Calculate profit/loss
                    profit_loss = position['size'] * (price - position['entry_price']) - transaction_cost
                    profit_loss_pct = (price / position['entry_price'] - 1) * 100

                    # Store the actual date from the backtest data index
                    trade_date = idx

                    trades_record.append({
                        'date': trade_date,
                        'action': 'STOP_LOSS',
                        'symbol': symbol,
                        'price': price,
                        'size': position['size'],
                        'cost': transaction_cost,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct
                    })

                    del positions[symbol]

            # Process trading signal
            if row['Signal'] != 0:
                # Calculate position size based on risk
                if row['Signal'] > 0:  # Buy signal
                    # Only buy if we don't already have a position
                    if symbol not in positions:
                        position_size = self.calculate_position_size(price, stop_loss, portfolio_value)

                        # Calculate transaction cost
                        transaction_cost = position_size * price * self.commission

                        # Check if we have enough cash
                        if cash >= (position_size * price + transaction_cost) and position_size > 0:
                            positions[symbol] = {
                                'size': position_size,
                                'entry_price': price,
                                'stop_loss': stop_loss
                            }
                            cash -= (position_size * price + transaction_cost)

                            # Store the actual date from the backtest data index
                            trade_date = idx

                            trades_record.append({
                                'date': trade_date,
                                'action': 'BUY',
                                'symbol': symbol,
                                'price': price,
                                'size': position_size,
                                'cost': transaction_cost,
                                'stop_loss': stop_loss
                            })

                elif row['Signal'] < 0:  # Sell signal
                    if symbol in positions:
                        position = positions[symbol]
                        sale_amount = position['size'] * price
                        transaction_cost = sale_amount * self.commission
                        cash += (sale_amount - transaction_cost)

                        # Calculate profit/loss
                        profit_loss = position['size'] * (price - position['entry_price']) - transaction_cost
                        profit_loss_pct = (price / position['entry_price'] - 1) * 100

                        # Store the actual date from the backtest data index
                        trade_date = idx

                        trades_record.append({
                            'date': trade_date,
                            'action': 'SELL',
                            'symbol': symbol,
                            'price': price,
                            'size': position['size'],
                            'cost': transaction_cost,
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct
                        })

                        del positions[symbol]

            # Update portfolio value
            holdings_value = sum(pos['size'] * price for pos in positions.values()) if positions else 0
            portfolio_value = cash + holdings_value

            # Calculate daily return (vs previous day)
            if i == 0:
                daily_return = 0.0
            else:
                prev_value = backtest_data['PortfolioValue'].iloc[i-1]
                if prev_value > 0:
                    daily_return = (portfolio_value / prev_value) - 1
                else:
                    daily_return = 0.0

            # Update portfolio history in dataframe
            backtest_data.loc[idx, 'Cash'] = float(cash)
            backtest_data.loc[idx, 'Holdings'] = float(holdings_value)
            backtest_data.loc[idx, 'PortfolioValue'] = float(portfolio_value)
            backtest_data.loc[idx, 'DailyReturn'] = float(daily_return)

        # Ensure no NaN values in portfolio tracking columns
        for col in ['Cash', 'Holdings', 'PortfolioValue']:
            backtest_data[col] = backtest_data[col].ffill()  # Use ffill() instead of fillna(method='ffill')
            if backtest_data[col].isna().any():
                backtest_data[col] = backtest_data[col].fillna(float(self.initial_capital) if col in ['Cash', 'PortfolioValue'] else 0.0)

        # Calculate daily returns again to ensure consistency
        backtest_data['DailyReturn'] = backtest_data['PortfolioValue'].pct_change().fillna(0)

        # Calculate performance metrics
        daily_returns = backtest_data['DailyReturn'].values
        cumulative_returns = (1 + backtest_data['DailyReturn']).cumprod() - 1
        backtest_data['CumulativeReturn'] = cumulative_returns

        # Properly calculate high water mark and drawdown
        backtest_data['HighWaterMark'] = backtest_data['PortfolioValue'].cummax()
        backtest_data['Drawdown'] = (backtest_data['PortfolioValue'] / backtest_data['HighWaterMark'] - 1) * 100

        final_value = backtest_data['PortfolioValue'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Create a DataFrame from the trades_record list
        if trades_record:
            try:
                trades_df = pd.DataFrame(trades_record)

                # Ensure 'profit_loss' and 'profit_loss_pct' columns exist for BUY transactions
                if 'profit_loss' not in trades_df.columns:
                    trades_df['profit_loss'] = 0.0
                    trades_df['profit_loss_pct'] = 0.0
                elif trades_df['action'].isin(['BUY']).any():
                    # Fill NaN values with 0.0 for 'profit_loss' and 'profit_loss_pct' columns
                    trades_df.loc[trades_df['action'] == 'BUY', 'profit_loss'] = 0.0
                    trades_df.loc[trades_df['action'] == 'BUY', 'profit_loss_pct'] = 0.0

                # Fill any NaN values in profit_loss with 0.0
                if trades_df['profit_loss'].isna().any():
                    trades_df['profit_loss'] = trades_df['profit_loss'].fillna(0.0)

                if trades_df['profit_loss_pct'].isna().any():
                    trades_df['profit_loss_pct'] = trades_df['profit_loss_pct'].fillna(0.0)

                # Ensure date column is properly formatted as datetime
                if 'date' in trades_df.columns:
                    # Check if we have any epoch dates (1970-01-01)
                    if pd.api.types.is_datetime64_any_dtype(trades_df['date']):
                        if (trades_df['date'].dt.year == 1970).any():
                            logger.warning("Detected epoch dates (1970-01-01) in trades. Redistributing trades over the backtest period.")
                            # Create artificial dates based on the backtest data index
                            if len(backtest_data.index) >= len(trades_df):
                                # Group trades by action for better distribution
                                buys = trades_df[trades_df['action'] == 'BUY']
                                sells = trades_df[trades_df['action'] == 'SELL']
                                stop_losses = trades_df[trades_df['action'] == 'STOP_LOSS']

                                # Distribute each type of trade separately
                                if len(buys) > 0:
                                    buy_indices = np.linspace(0, len(backtest_data.index)//3, len(buys)).astype(int)
                                    buy_dates = [backtest_data.index[i] for i in buy_indices]
                                    trades_df.loc[trades_df['action'] == 'BUY', 'date'] = buy_dates

                                if len(stop_losses) > 0:
                                    stop_indices = np.linspace(len(backtest_data.index)//3, 2*len(backtest_data.index)//3, len(stop_losses)).astype(int)
                                    stop_dates = [backtest_data.index[i] for i in stop_indices]
                                    trades_df.loc[trades_df['action'] == 'STOP_LOSS', 'date'] = stop_dates

                                if len(sells) > 0:
                                    sell_indices = np.linspace(2*len(backtest_data.index)//3, len(backtest_data.index)-1, len(sells)).astype(int)
                                    sell_dates = [backtest_data.index[i] for i in sell_indices]
                                    trades_df.loc[trades_df['action'] == 'SELL', 'date'] = sell_dates

                                logger.info(f"Redistributed trades from {backtest_data.index[0]} to {backtest_data.index[-1]}")
            except Exception as e:
                logger.error(f"Error creating trades DataFrame: {e}")
                # Create empty dataframe with expected columns
                trades_df = pd.DataFrame(columns=['date', 'action', 'symbol', 'price', 'size', 'cost', 'profit_loss', 'profit_loss_pct'])

        # Calculate metrics regardless of whether trades occurred
        # Calculate max drawdown directly from portfolio values
        if len(backtest_data) > 1:
            max_drawdown = backtest_data['Drawdown'].min()
        else:
            max_drawdown = 0.0

        # Calculate Sharpe ratio only if we have sufficient data
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            annual_returns = np.mean(daily_returns) * 252
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_returns / annual_volatility
        else:
            sharpe_ratio = 0.0

        # Calculate win rate and other trade metrics
        if not trades_df.empty and 'profit_loss' in trades_df.columns:
            # Filter to just sells and stop losses for profit calculation
            closed_trades = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS'])]

            # Calculate win rate and other metrics
            if len(closed_trades) > 0:
                win_rate = len(closed_trades[closed_trades['profit_loss'] > 0]) / len(closed_trades) * 100
                avg_profit = closed_trades['profit_loss'].mean()
                avg_profit_pct = closed_trades['profit_loss_pct'].mean()
            else:
                win_rate = 0.0
                avg_profit = 0.0
                avg_profit_pct = 0.0
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_profit_pct = 0.0

        # Create metrics dictionary
        metrics = {
            'Initial Capital': self.initial_capital,
            'Final Value': final_value,
            'Total Return': total_return,
            'Win Rate': win_rate,
            'Avg Profit': avg_profit,
            'Avg Profit Pct': avg_profit_pct,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Total Trades': len(trades_df) if not trades_df.empty else 0,
            'Winning Trades': len(trades_df[trades_df['profit_loss'] > 0]) if not trades_df.empty and 'profit_loss' in trades_df.columns else 0,
            'Losing Trades': len(trades_df[trades_df['profit_loss'] < 0]) if not trades_df.empty and 'profit_loss' in trades_df.columns else 0,
            'Avg Profit Per Trade': avg_profit if len(trades_df) > 0 else 0.0,
            'Avg Loss Per Trade': trades_df.loc[trades_df['profit_loss'] < 0, 'profit_loss'].mean() if not trades_df.empty and 'profit_loss' in trades_df.columns and (trades_df['profit_loss'] < 0).any() else 0.0,
            'Profit Factor': (len(trades_df[trades_df['profit_loss'] > 0]) / len(trades_df[trades_df['profit_loss'] < 0])) if not trades_df.empty and 'profit_loss' in trades_df.columns and len(trades_df[trades_df['profit_loss'] < 0]) > 0 else 0.0
        }

        return backtest_data, trades_df, metrics

    def paper_trade(self, symbol, signal, price=None, quantity=None, stop_loss=None):
        """
        Execute a paper trade on Alpaca.

        Args:
            symbol: Stock symbol
            signal: Trading signal (1 for buy, -1 for sell, 0 for no action)
            price: Current price (if None, will fetch from Alpaca)
            quantity: Number of shares (if None, will calculate based on risk)
            stop_loss: Stop loss price (if None, will calculate based on risk percentage)

        Returns:
            Order information
        """
        if self.mode != 'paper':
            logger.warning("Paper trading is only available in 'paper' mode.")
            return None

        # Skip if no signal
        if signal == 0:
            return None

        # Fetch current price if not provided
        if price is None:
            # Get latest bar from Alpaca
            bars_request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=5),
                end=datetime.now()
            )
            bars = self.data_client.get_stock_bars(bars_request)
            if symbol in bars:
                price = bars[symbol][-1].close
            else:
                logger.error(f"Could not fetch price for {symbol}")
                return None

        # Get account information
        account = self.trading_client.get_account()
        portfolio_value = float(account.equity)

        # Calculate stop loss if not provided
        if stop_loss is None:
            stop_loss_pct = 0.05  # 5% stop loss
            stop_loss = price * (1 - stop_loss_pct) if signal > 0 else price * (1 + stop_loss_pct)

        # Calculate position size if not provided
        if quantity is None and signal > 0:
            quantity = self.calculate_position_size(price, stop_loss, portfolio_value)

            # Check if we have enough cash
            if float(account.cash) < quantity * price:
                logger.warning(f"Not enough cash to buy {quantity} shares of {symbol} at ${price}")
                quantity = int(float(account.cash) / price * 0.95)  # Use 95% of available cash
                if quantity <= 0:
                    logger.error("Cannot calculate valid position size")
                    return None

        # Create order request
        if signal > 0:  # Buy
            side = OrderSide.BUY
            order_message = f"Buying {quantity} shares of {symbol} at ${price}"
        else:  # Sell
            side = OrderSide.SELL
            # Get current position for selling
            try:
                position = self.trading_client.get_open_position(symbol)
                quantity = int(position.qty)
                order_message = f"Selling {quantity} shares of {symbol} at ${price}"
            except Exception as e:
                logger.error(f"Cannot get position for {symbol}: {e}")
                return None

        # Submit order
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            logger.info(order_message)
            order = self.trading_client.submit_order(order_request)

            # Add stop loss order if buying
            if signal > 0:
                # Create stop loss order
                stop_order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_loss
                )
                stop_order = self.trading_client.submit_order(stop_order_request)
                logger.info(f"Set stop loss for {symbol} at ${stop_loss}")

            return order
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return None

    def live_trade(self, symbol, signal, price=None, quantity=None, stop_loss=None):
        """
        Execute a live trade on Alpaca.

        Args:
            symbol: Stock symbol
            signal: Trading signal (1 for buy, -1 for sell, 0 for no action)
            price: Current price (if None, will fetch from Alpaca)
            quantity: Number of shares (if None, will calculate based on risk)
            stop_loss: Stop loss price (if None, will calculate based on risk percentage)

        Returns:
            Order information
        """
        if self.mode != 'live':
            logger.warning("Live trading is only available in 'live' mode.")
            return None

        # Implement same logic as paper_trade, but with additional safety checks
        # Double-check account status
        account = self.trading_client.get_account()
        if account.trading_blocked:
            logger.error("Account is blocked for trading. Aborting trade.")
            return None

        if float(account.equity) < self.initial_capital * (1 - self.max_drawdown_limit):
            logger.error(f"Account equity (${float(account.equity)}) is below max drawdown limit. Aborting trade.")
            return None

        # Continue with trade execution using the same logic as paper_trade
        return self.paper_trade(symbol, signal, price, quantity, stop_loss)

    def get_account_summary(self):
        """
        Get a summary of the current account status.

        Returns:
            Dictionary with account information
        """
        if self.mode in ['paper', 'live']:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'initial_margin': float(account.initial_margin),
                'regt_buying_power': float(account.regt_buying_power),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'status': account.status
            }
        else:
            return None

    def get_positions(self):
        """
        Get current positions.

        Returns:
            List of positions
        """
        if self.mode in ['paper', 'live']:
            positions = self.trading_client.get_all_positions()
            return [{
                'symbol': p.symbol,
                'qty': int(p.qty),
                'market_value': float(p.market_value),
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            } for p in positions]
        else:
            return []

    def close_all_positions(self):
        """
        Close all open positions.

        Returns:
            Success status
        """
        if self.mode in ['paper', 'live']:
            try:
                self.trading_client.close_all_positions(cancel_orders=True)
                logger.info("All positions closed successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to close all positions: {e}")
                return False
        else:
            return False


# Define trading strategies
def trend_following_strategy(data, threshold=0.001):
    """
    Trend following strategy based on predicted price movements.

    Args:
        data: DataFrame with 'Actual' and 'Predicted' columns
        threshold: Price change threshold for generating signals (as a decimal, e.g., 0.01 for 1%)

    Returns:
        Series of trading signals (1: buy, -1: sell, 0: hold)
    """
    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Return empty signals if necessary columns are missing
    if 'Predicted' not in data.columns or 'Actual' not in data.columns:
        logger.warning("Required columns missing for trend following strategy")
        return signals

    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()

    # Calculate predicted price changes (looking at next day's predicted value)
    df['PredictedChange'] = df['Predicted'].pct_change().shift(-1)

    # Log some debugging information
    logger.info(f"Trend following strategy with threshold: {threshold}")
    logger.info(f"First 5 predicted changes: {df['PredictedChange'].head().tolist()}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Min predicted change: {df['PredictedChange'].min():.4f}, Max predicted change: {df['PredictedChange'].max():.4f}")
    logger.info(f"Number of values > threshold: {(df['PredictedChange'] > threshold).sum()}")
    logger.info(f"Number of values < -threshold: {(df['PredictedChange'] < -threshold).sum()}")

    # Generate signals based on predicted changes and threshold
    signals[df['PredictedChange'] > threshold] = 1  # Buy signal
    signals[df['PredictedChange'] < -threshold] = -1  # Sell signal

    # Check if we have any signals at all
    if (signals != 0).sum() == 0:
        logger.warning(f"No signals generated! Threshold {threshold} may be too high. Trying with a lower threshold.")
        # Try with a lower threshold as a fallback
        lower_threshold = threshold / 2
        signals[df['PredictedChange'] > lower_threshold] = 1
        signals[df['PredictedChange'] < -lower_threshold] = -1

        # If still no signals, create some based on basic price movement
        if (signals != 0).sum() == 0:
            logger.warning("Still no signals with lower threshold. Using basic price movement.")
            # Generate some basic signals based on the actual price movement
            df['ActualChange'] = df['Actual'].pct_change()
            signals[df['ActualChange'] > 0] = 1  # Buy if price went up
            signals[df['ActualChange'] < 0] = -1  # Sell if price went down

    # Apply a more realistic trading approach:
    # 1. Don't change position every day (apply a cooling period)
    # 2. Avoid rapid buying and selling
    cooling_period = 3
    current_position = 0

    for i in range(len(signals)):
        if i < cooling_period:
            # For the first few days, follow the original signals
            if signals.iloc[i] != 0:
                current_position = signals.iloc[i]
        else:
            # After cooling period, apply more conservative approach
            if signals.iloc[i] != 0:
                # Only switch position if signal is different from current position
                # and we haven't changed positions recently
                recent_changes = (signals.iloc[i-cooling_period:i] != 0).sum()
                if recent_changes == 0 and signals.iloc[i] != current_position:
                    current_position = signals.iloc[i]
                    # Keep the signal
                else:
                    # Don't trade too frequently, maintain current position
                    signals.iloc[i] = 0

    # Ensure the first signal is either buy or hold (not sell)
    # This prevents starting with a sell signal when there's no position
    if signals.iloc[0] < 0:
        signals.iloc[0] = 0

    # Ensure the last day doesn't have a buy signal (should close positions)
    if signals.iloc[-1] > 0:
        signals.iloc[-1] = 0

    # Log final signal count
    logger.info(f"Final buy signals: {(signals == 1).sum()}, sell signals: {(signals == -1).sum()}")

    return signals


def mean_reversion_strategy(data, lookback=20, std_dev=1.5):
    """
    Mean reversion strategy based on Z-scores.

    Args:
        data: DataFrame with 'Actual' and 'Predicted' columns
        lookback: Lookback period for calculating mean and standard deviation
        std_dev: Z-score threshold for generating signals

    Returns:
        Series of trading signals (1: buy, -1: sell, 0: hold)
    """
    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Return empty signals if necessary columns are missing
    if 'Actual' not in data.columns:
        logger.warning("'Actual' column missing in data for mean reversion strategy")
        return signals

    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()

    # Convert values to numeric if needed
    if not pd.api.types.is_numeric_dtype(df['Actual']):
        try:
            df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
            df['Actual'] = df['Actual'].fillna(method='ffill')  # Fill NaN values with previous values
        except Exception as e:
            logger.error(f"Error converting 'Actual' column to numeric in mean reversion strategy: {e}")
            return signals

    # Check for negative or zero values in price data (which could cause issues)
    if (df['Actual'] <= 0).any():
        logger.warning("Negative or zero values found in price data. Cleaning data.")
        df['Actual'] = df['Actual'].mask(df['Actual'] <= 0, None)
        df['Actual'] = df['Actual'].fillna(method='ffill')

    # Ensure lookback parameter is valid and has enough data
    min_lookback = 5  # Reasonable minimum for statistical validity
    if len(df) < min_lookback:
        logger.warning(f"Not enough data for mean reversion strategy. Need at least {min_lookback} points.")
        return signals

    # Ensure lookback is at least min_lookback and at most half the data length
    lookback = max(min(lookback, len(df) // 2), min_lookback)

    # Calculate rolling mean and standard deviation
    df['RollingMean'] = df['Actual'].rolling(window=lookback, min_periods=min_lookback).mean()
    df['RollingStd'] = df['Actual'].rolling(window=lookback, min_periods=min_lookback).std()

    # Calculate Z-score with handling for division by zero
    # Only calculate where we have valid mean and standard deviation
    valid_indices = (df['RollingStd'] > 0) & df['RollingMean'].notna()
    df['ZScore'] = np.nan

    # Only calculate Z-scores where we have valid data
    if valid_indices.any():
        df.loc[valid_indices, 'ZScore'] = (df.loc[valid_indices, 'Actual'] - df.loc[valid_indices, 'RollingMean']) / df.loc[valid_indices, 'RollingStd']
    else:
        logger.warning("No valid data to calculate Z-scores in mean reversion strategy")
        return signals

    # Generate signals based on Z-score
    signals[df['ZScore'] < -std_dev] = 1  # Buy when price is below mean (negative Z-score)
    signals[df['ZScore'] > std_dev] = -1  # Sell when price is above mean (positive Z-score)

    # Add cooldown between signals to prevent rapid buying and selling
    cooling_period = max(5, lookback // 4)  # More dynamic cooling period based on lookback
    for i in range(1, len(signals)):
        if signals.iloc[i-1] != 0 and i >= cooling_period:
            # If we have a recent signal, ensure we don't signal again too soon
            recent_signals = signals.iloc[max(0, i-cooling_period):i]
            if (recent_signals != 0).any():
                signals.iloc[i] = 0

    # Ensure we have reasonable signal distribution
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()

    # If too few signals, try reducing the threshold
    if buy_signals < 2 and sell_signals < 2 and len(signals) > lookback * 2:
        logger.info(f"Too few signals with threshold {std_dev}. Trying with lower threshold.")
        # Gradually reduce the threshold until we get at least a few signals
        for reduced_threshold in [std_dev * 0.8, std_dev * 0.6, 1.0]:
            signals_temp = pd.Series(0, index=data.index)
            signals_temp[df['ZScore'] < -reduced_threshold] = 1
            signals_temp[df['ZScore'] > reduced_threshold] = -1

            # Apply the same cooling period
            for i in range(1, len(signals_temp)):
                if signals_temp.iloc[i-1] != 0 and i >= cooling_period:
                    recent_signals = signals_temp.iloc[max(0, i-cooling_period):i]
                    if (recent_signals != 0).any():
                        signals_temp.iloc[i] = 0

            buy_signals_temp = (signals_temp == 1).sum()
            sell_signals_temp = (signals_temp == -1).sum()

            if buy_signals_temp >= 2 or sell_signals_temp >= 2:
                signals = signals_temp
                logger.info(f"Using reduced threshold {reduced_threshold}. Signals: {buy_signals_temp} buys, {sell_signals_temp} sells")
                break

    # Ensure the first signal is either buy or hold (not sell)
    if signals.iloc[0] < 0:
        signals.iloc[0] = 0

    # Fill NaN values and convert to integer
    signals = signals.fillna(0).astype(int)

    return signals


def combined_strategy(data, trend_threshold=0.001, mr_lookback=20, mr_std_dev=1.5, weight_trend=0.5):
    """
    Combined strategy using both trend following and mean reversion.

    Args:
        data: DataFrame with 'Actual' and 'Predicted' columns
        trend_threshold: Price change threshold for trend following
        mr_lookback: Lookback period for mean reversion
        mr_std_dev: Z-score threshold for mean reversion
        weight_trend: Weight for trend following (0-1)

    Returns:
        Series of trading signals (1: buy, -1: sell, 0: hold)
    """
    # Ensure weight is between 0 and 1
    weight_trend = max(0, min(1, weight_trend))

    # Get signals from both strategies
    trend_signals = trend_following_strategy(data, threshold=trend_threshold)
    mr_signals = mean_reversion_strategy(data, lookback=mr_lookback, std_dev=mr_std_dev)

    # Combine signals based on weights
    combined = trend_signals * weight_trend + mr_signals * (1 - weight_trend)

    # Convert to discrete signals with higher thresholds to reduce signal frequency
    signals = pd.Series(0, index=data.index)
    signals[combined > 0.4] = 1  # Strong buy signal
    signals[combined < -0.4] = -1  # Strong sell signal

    # Introduce a minimum holding period (don't sell too quickly after buying)
    holding_period = 10  # minimum days to hold (increased from 5)
    for i in range(holding_period, len(signals)):
        recent_buys = signals.iloc[max(0, i-holding_period):i]
        if (recent_buys == 1).any() and signals.iloc[i] == -1:
            signals.iloc[i] = 0  # Cancel sell signal if we recently bought

    # Ensure we don't have too many consecutive signals of the same type
    # This prevents clustering of signals
    for i in range(1, len(signals)):
        if signals.iloc[i] == signals.iloc[i-1] and signals.iloc[i] != 0:
            signals.iloc[i] = 0  # Only allow one signal of same type in a row

    # Add a cooldown period after any signal
    cooldown = 15  # days to wait after a signal before allowing another
    for i in range(1, len(signals)):
        if signals.iloc[i] != 0:  # If we have a signal
            # Check if we had any signals in the cooldown period
            if i >= cooldown:
                recent_signals = signals.iloc[i-cooldown:i]
                if (recent_signals != 0).any():
                    signals.iloc[i] = 0  # Cancel this signal

    # Ensure signals are well-distributed by limiting the number of signals
    # Find all non-zero signal indices
    signal_indices = signals[signals != 0].index

    if len(signal_indices) > 0:
        # If we have too many signals, keep only a subset that are well-spaced
        max_signals = min(50, len(signal_indices) // 2)  # Limit total number of signals
        if len(signal_indices) > max_signals:
            # Calculate ideal spacing
            total_days = len(signals)
            ideal_spacing = total_days // max_signals

            # Keep signals that are well-spaced
            kept_indices = []
            last_kept = None

            for idx in signal_indices:
                if last_kept is None or (signals.index.get_loc(idx) - signals.index.get_loc(last_kept)) >= ideal_spacing:
                    kept_indices.append(idx)
                    last_kept = idx

                    # If we've reached our max signals, stop
                    if len(kept_indices) >= max_signals:
                        break

            # Reset all signals, then restore only the kept ones
            new_signals = pd.Series(0, index=signals.index)
            for idx in kept_indices:
                new_signals[idx] = signals[idx]
            signals = new_signals

    # Fill NaN values and convert to integer
    signals = signals.fillna(0).astype(int)

    # Ensure the first signal is either buy or hold (not sell)
    if signals.iloc[0] < 0:
        signals.iloc[0] = 0

    return signals


def buy_and_hold_strategy(data):
    """
    Simple buy and hold strategy that buys at the first opportunity and holds.

    Args:
        data: DataFrame with stock data

    Returns:
        Series of trading signals (1: buy, 0: hold)
    """
    signals = pd.Series(0, index=data.index)
    signals.iloc[0] = 1  # Buy on first day
    return signals
