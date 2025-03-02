# Stock Price Prediction with Transformer Model

A Streamlit application that uses a Transformer neural network to predict stock prices and backtest trading strategies based on model predictions.

## Features

- **Stock Data Loading**: Load historical stock data from Yahoo Finance for any ticker symbol
- **Technical Analysis**: Apply various technical indicators to enhance prediction accuracy
- **News Sentiment Analysis**: Incorporate news sentiment for more comprehensive predictions (requires Alpaca API keys)
- **Transformer Model**: State-of-the-art neural network architecture for time series forecasting
- **Customizable Parameters**: Adjust model architecture, training parameters, and more
- **Interactive Visualizations**: Visualize predictions, model performance, and backtest results
- **Strategy Backtesting**: Test different trading strategies based on model predictions

## Backtesting Features

The application now includes a comprehensive backtesting system that allows you to:

1. **Simulate Trading Strategies**:

   - Trend Following: Trade based on predicted price movements
   - Mean Reversion: Trade based on deviations from historical averages
   - Buy and Hold: Simple buy and hold strategy for comparison

2. **Customize Backtesting Parameters**:

   - Initial capital
   - Commission rates
   - Strategy-specific parameters (thresholds, lookback periods, etc.)

3. **Performance Metrics**:

   - Total return
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Comparison against buy & hold benchmark

4. **Visualizations**:
   - Portfolio value over time
   - Drawdown analysis
   - Detailed trade history

## Model Performance

The Transformer model shows promising results for stock price prediction:

- Low training loss indicates good fit on historical data
- Testing on out-of-sample data shows reasonable generalization
- Model can capture both trend and some volatility patterns
- Performance varies by stock and market conditions

## Limitations and Considerations

1. **Market Efficiency**: Markets are highly efficient, making consistent prediction challenging
2. **Black Swan Events**: Unpredictable events can dramatically affect stock prices
3. **Past ≠ Future**: Historical patterns don't guarantee future results
4. **Risk Management**: Always employ proper risk management regardless of model predictions

## Getting Started

1. **Installation**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Setting up API Keys** (optional for news data):
   Create a `.env` file with your Alpaca API keys:

   ```
   ALPACA_LIVE_KEY_ID=your_key_here
   ALPACA_LIVE_SECRET=your_secret_here
   ```

3. **Running the Application**:

   ```bash
   streamlit run main.py
   ```

4. **Using the App**:
   - Select a stock symbol and date range
   - Choose technical indicators to include
   - Adjust model parameters as needed
   - Train the model and view predictions
   - Backtest trading strategies

## Disclaimer

This application is for educational purposes only. The predictions and backtesting results should not be used as financial advice. Always consult with a qualified financial advisor before making investment decisions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-green.svg)

![Application Screenshot](https://github.com/zachrizzo/transfomer_stocks/assets/placeholder/screenshot.png)

## Project Structure

- `main.py`: Main application entry point and Streamlit interface
- `model.py`: Transformer model architecture and training logic
- `data_utils.py`: Data loading, preprocessing, and feature engineering
- `indicators.py`: Technical indicator calculations
- `config.py`: Configuration and indicator definitions
- `requirements.txt`: Required Python packages

## Model Architecture

This project uses a Transformer-based neural network, inspired by the architecture from the paper "Attention Is All You Need" by Vaswani et al. The model includes:

- Input embedding layer
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

The Transformer architecture is particularly well-suited for time series forecasting as it can efficiently capture long-range dependencies in sequential data without the limitations of recurrent architectures.

## Data Sources

- Stock price data: Yahoo Finance API via the `yfinance` library
- News data: Alpaca Market Data API

## Customization

You can easily extend the application by:

1. Adding new technical indicators in `indicators.py`
2. Registering them in `config.py`
3. Implementing new data preprocessing techniques in `data_utils.py`
4. Modifying the Transformer architecture in `model.py`

## Performance Considerations

- Model training time depends on the selected date range, number of indicators, and training parameters.
- Using GPU acceleration (CUDA) or MPS (on Apple Silicon) significantly improves training speed.
- Larger models (more layers/hidden size) may improve accuracy but require more computational resources.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the web application framework
- [Yahoo Finance](https://finance.yahoo.com/) for stock price data
- [Alpaca](https://alpaca.markets/) for news data
- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis

---
