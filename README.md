# Stock Prediction with News

This repository contains a Streamlit application that predicts stock prices using a PyTorch Transformer model and news sentiment analysis. The app fetches historical stock data from Yahoo Finance and news articles from Alpaca, processes the data, trains a Transformer model, and visualizes the results.

## Table of Contents

- [Stock Prediction with News](#stock-prediction-with-news)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Architecture](#model-architecture)
  - [Data Fetching and Preprocessing](#data-fetching-and-preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zachrizzo/transfomer_stocks.git
   cd stock-prediction-with-news
   ```

2. Create and activate a virtual environment with Anaconda:

   ```bash
   conda create --name stock_prediction_env python=3.9
   conda activate stock_prediction_env
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your Alpaca API keys:

   ```env
   ALPACA_PAPER_KEY_ID=your_alpaca_api_key_id
   ALPACA_PAPER_SECRET=your_alpaca_api_secret_key
   ```

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Enter a stock symbol (e.g., `TSLA`).
   - Select a start date and an end date for the data.
   - Optionally, check "Use Volume Data" and "Use News Data" to include these features in the model.
   - Adjust the training parameters in the sidebar.
   - Click "Rerun Training" to train the model with the selected parameters.

## Model Architecture

The model used in this application is a Transformer model implemented in PyTorch. The model consists of:

- An embedding layer to convert the input features into a higher-dimensional space.
- A Transformer encoder with multiple layers and attention heads.
- A fully connected layer to produce the final output.

## Data Fetching and Preprocessing

The application fetches historical stock data from Yahoo Finance using the `yfinance` library and news articles from Alpaca's API. The data is normalized using MinMaxScaler from `sklearn`. News sentiment is analyzed using `TextBlob`.

## Training

The application trains the Transformer model using the following parameters, adjustable in the Streamlit sidebar:

- Hidden Size
- Number of Layers
- Number of Heads
- Dropout
- Number of Epochs
- Batch Size
- Learning Rate

The training process includes:

- Creating sequences from the normalized data.
- Training the model using MSE loss and the Adam optimizer.
- Displaying the training progress and visualizing the training and testing predictions.

## Evaluation

The model is evaluated on the testing set, and the predictions are visualized using Streamlit's line chart. The number of parameters (neurons) in the model is also displayed.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Alpaca](https://alpaca.markets/)
- [TextBlob](https://textblob.readthedocs.io/)

---
