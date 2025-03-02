# Stock Price Prediction with Transformer Model

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-green.svg)

A sophisticated web application that uses a Transformer neural network model to predict stock prices. The application combines historical stock price data with technical indicators and news sentiment analysis to improve prediction accuracy.

![Application Screenshot](https://github.com/zachrizzo/transfomer_stocks/assets/placeholder/screenshot.png)

## Features

- Interactive web interface built with Streamlit
- Transformer-based neural network architecture for time series forecasting
- Support for multiple technical indicators:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Moving Average Convergence Divergence (MACD)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - Average True Range (ATR)
  - Average Directional Index (ADX)
- Integration with news sentiment analysis to incorporate market sentiment
- Real-time data fetching from Yahoo Finance
- News data fetching from Alpaca API
- Interactive model training with adjustable parameters
- Future stock price prediction with visualization
- Model download capability for external use

## Installation

### Prerequisites

- Python 3.9 or higher
- Alpaca API keys (for news data)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/zachrizzo/transfomer_stocks.git
cd transfomer_stocks
```

2. Create and activate a virtual environment (optional but recommended):

```bash
# Using conda
conda create -n stock_prediction python=3.9
conda activate stock_prediction

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Alpaca API keys:

```env
ALPACA_LIVE_KEY_ID=your_alpaca_api_key
ALPACA_LIVE_SECRET=your_alpaca_api_secret
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run main.py
```

2. The application will open in your default web browser.

3. Enter a stock symbol (e.g., AAPL, TSLA, MSFT).

4. Select a date range for historical data.

5. Choose the technical indicators you want to include in the model.

6. Adjust model parameters in the sidebar if desired:

   - Hidden Size: Dimension of the model's hidden layers
   - Number of Layers: Depth of the transformer encoder
   - Number of Heads: Attention heads in each encoder layer
   - Dropout: Regularization parameter
   - Number of Epochs: Training iterations
   - Batch Size: Number of samples per gradient update
   - Learning Rate: Step size for optimizer

7. The model will automatically train on the selected data.

8. View the model evaluation metrics and visualizations.

9. Predict future stock prices by specifying the number of days to forecast.

10. Download the trained model for external use if desired.

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
