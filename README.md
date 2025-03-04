# Stock Price Prediction with Transformer Neural Networks

A sophisticated Streamlit application that leverages state-of-the-art Transformer neural networks to predict stock prices and backtest trading strategies. This project combines technical analysis, sentiment analysis from financial news, and advanced deep learning to provide a comprehensive toolkit for stock analysis and trading strategy development.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-green.svg)

## 🌟 Key Features

- **Multi-source Data Integration**:

  - Historical stock price data from Yahoo Finance
  - Real-time and historical news data with sentiment analysis (via Alpaca API)
  - Technical indicators and volume data

- **Advanced Transformer Model**:

  - Multi-head self-attention mechanism for capturing temporal dependencies
  - Customizable model architecture (layers, heads, dimensions)
  - Interactive model training with real-time loss visualization
  - 3D neural network visualization and attention pattern exploration

- **Comprehensive Trading Module**:

  - Multiple backtesting strategies (Trend Following, Mean Reversion, Combined)
  - Risk management with position sizing, stop-loss, and drawdown limits
  - Complete performance metrics and visualization
  - Option for paper trading and live trading via Alpaca API

- **Interactive User Experience**:
  - Intuitive Streamlit interface with real-time model training
  - Dynamic visualization of predictions and backtesting results
  - Model saving and loading capabilities
  - Customizable parameters for all aspects of the system

## 📊 Model Performance

The Transformer architecture offers several advantages for stock price prediction:

- **Attention Mechanisms**: Captures long-range temporal dependencies without the limitations of RNNs
- **Parallel Processing**: Faster training compared to sequential models
- **Feature Importance**: Self-attention provides interpretable insights into which time periods influence predictions
- **Adaptability**: Performs well across different market conditions and stock types

Performance metrics typically include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy (% of correct trend predictions)
- Trading performance metrics (Sharpe Ratio, Max Drawdown, Returns)

## 💹 Trading Strategies

The application implements several trading strategies:

### 1. Trend Following

- Trades in the direction of predicted price movements
- Customizable entry/exit thresholds
- Risk management with dynamic position sizing

### 2. Mean Reversion

- Identifies potential price reversals based on deviations from historical norms
- Adjustable lookback periods and standard deviation thresholds
- Integrated with technical indicators

### 3. Combined Strategy

- Blends trend following and mean reversion approaches
- Weighted signal generation for more balanced decision-making
- Adaptive to different market conditions

### 4. Buy and Hold

- Simple benchmark strategy for comparison
- Evaluates the effectiveness of active trading strategies

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU recommended for faster training

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/transfomer_stocks.git
   cd transfomer_stocks
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys** (for news data and live trading):
   Create a `.env` file with your Alpaca API keys:

   ```
   ALPACA_LIVE_KEY_ID=your_key_here
   ALPACA_LIVE_SECRET=your_secret_here
   ```

4. **Launch the application**:
   ```bash
   streamlit run main.py
   ```

## 🖥️ Using the Application

### Data Selection

1. Enter a stock symbol (default: AAPL)
2. Select date range for training and testing
3. Choose technical indicators to include in the model
4. Toggle news sentiment analysis and volume data

### Model Configuration

1. Adjust model architecture (layers, heads, dimensions)
2. Set training parameters (epochs, batch size, learning rate)
3. Train the model and monitor loss in real-time
4. Visualize model predictions on test data

### Backtesting

1. Select a trading strategy
2. Configure strategy parameters and risk management settings
3. Run backtest and analyze performance
4. Compare against benchmark strategies

### Live/Paper Trading (with Alpaca API)

1. Set up trading parameters and risk limits
2. Enable trading mode (paper or live)
3. Monitor trades and performance in real-time

## 📁 Project Structure

- `main.py`: Main Streamlit application and UI
- `model.py`: Transformer model architecture and training logic
- `data_utils.py`: Data loading, preprocessing, and feature engineering
- `indicators.py`: Technical indicator calculations
- `config.py`: Configuration settings and parameter definitions
- `trading.py`: Trading strategies and backtesting engine
- `live_trading.py`: Live and paper trading implementation
- `pages/`: Additional Streamlit application pages

## 🔧 Technical Implementation

### Model Architecture

The Transformer model consists of:

```
TransformerModel
├── Input Embedding Layer
├── Positional Encoding
├── Transformer Encoder Layers (configurable)
│   ├── Multi-Head Self-Attention
│   │   ├── Query, Key, Value Projections
│   │   └── Attention Scaling and Softmax
│   ├── Feed-Forward Network
│   ├── Layer Normalization
│   └── Residual Connections
└── Output Layer
```

### Data Preprocessing Pipeline

1. Load historical price data
2. Calculate technical indicators
3. Fetch and process news sentiment (optional)
4. Normalize features
5. Create sequence data for training
6. Split into training and validation sets

### Trading Engine

The trading engine simulates realistic market conditions:

- Commission and slippage modeling
- Position sizing based on available capital
- Stop-loss implementation
- Portfolio performance tracking

## ⚠️ Limitations and Considerations

1. **Market Unpredictability**: Financial markets are influenced by countless factors, many of which cannot be captured in historical data
2. **Data Limitations**: News sentiment analysis may not capture all market-relevant information
3. **Model Complexity**: Transformer models require significant data and computational resources
4. **Overfitting Risk**: Financial models can easily overfit to historical patterns that don't persist
5. **Market Regimes**: Models trained in one market regime may perform poorly when conditions change

## 📜 Disclaimer

This application is provided for educational and research purposes only. The predictions and backtesting results should not be used as financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the interactive web application framework
- [Yahoo Finance](https://finance.yahoo.com/) for historical stock data
- [Alpaca Markets](https://alpaca.markets/) for news data and trading API
- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis

---
