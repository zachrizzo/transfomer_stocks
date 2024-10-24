import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import dotenv
from model import TransformerModel, train_model, predict_future, count_parameters
from data_utils import load_data, preprocess_data, fetch_all_news, create_sequences, normalize, denormalize
from config import list_of_indicators, get_indicator_by_key

# Load environment variables
dotenv.load_dotenv()

# Set the API keys
alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')

# Set page title
st.set_page_config(page_title="Stock Prediction with News")

# Initialize the stock symbol variable
if 'stockSymbol' not in st.session_state:
    st.session_state['stockSymbol'] = 'TSLA'

# Check if MPS is available and use it; otherwise use CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

st.write(f"Using device: {device}")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    stockSymbol = st.text_input("Enter Stock Symbol", st.session_state['stockSymbol'])

with col2:
    start_date = st.date_input("Start Date", datetime(2018, 1, 1))

with col3:
    end_date = st.date_input("End Date", datetime(2023, 1, 1))

# Dynamically generate checkboxes for indicators
st.subheader("Select Indicators")
for indicator in list_of_indicators:
    indicator['selected'] = st.checkbox(indicator['name'], key=f"checkbox_{indicator['name']}")

# Get the selected indicators
use_volume = get_indicator_by_key('volume')['selected']
use_news = get_indicator_by_key('news')['selected']

# Load and prepare the data
stock_data, close_scaler = load_data(stockSymbol, start_date, end_date)
train_size = int(len(stock_data) * 0.8)
train_data = stock_data[:train_size]
test_data = stock_data[train_size:]

# Fetch and prepare news data
dates = stock_data.index.date
news_data = fetch_all_news(stockSymbol, dates) if use_news else pd.DataFrame()

for indicator in list_of_indicators:
    if indicator['selected'] and indicator['function'] is not None:
        try:
            if indicator['key'] in ['volume', 'news']:
                continue

            if indicator['key'] == 'sma':
                result = indicator['function'](stock_data['Close'], **indicator['params'])
            elif indicator['key'] == 'macd':
                result = indicator['function'](stock_data, **indicator['params'])
            else:
                result = indicator['function'](stock_data['Close'], **indicator['params'])

            if isinstance(result, pd.DataFrame):
                stock_data = pd.concat([stock_data, result], axis=1)
            elif isinstance(result, pd.Series):
                stock_data[indicator['name']] = result
            else:
                st.warning(f"Unexpected result type for indicator {indicator['name']}. Skipping.")
        except Exception as e:
            st.error(f"Error applying indicator {indicator['name']}: {str(e)}")
            st.warning(f"Skipping indicator {indicator['name']} due to error.")
            indicator['selected'] = False

# Recalculate input_size after potential indicator failures
input_size = 1  # Always include 'Close' price
for indicator in list_of_indicators:
    if indicator['selected']:
        input_size += indicator['size']

# Preprocess data
seq_length = 20
train_X, train_y = preprocess_data(train_data, news_data, list_of_indicators, close_scaler, seq_length)
test_X, test_y = preprocess_data(test_data, news_data, list_of_indicators, close_scaler, seq_length)

# Get user input for training parameters
st.sidebar.header("Training Parameters")
hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=512, value=128, step=32)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=6, value=2, step=1)
num_heads = st.sidebar.slider("Number of Heads", min_value=1, max_value=16, value=8, step=1)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=5000, value=50, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=32, step=16)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")

# Store training parameters in a dictionary
training_params = {
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'num_heads': num_heads,
    'dropout': dropout,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'train_X_shape': train_X.shape,
    'train_y_shape': train_y.shape,
    'stockSymbol': stockSymbol,
    'start_date': start_date,
    'end_date': end_date,
    'selected_indicators': [indicator['key'] for indicator in list_of_indicators if indicator['selected']],
}

if 'model_state_dict' not in st.session_state or st.session_state.get('training_params') != training_params:
    if train_X.size == 0 or train_y.size == 0:
        st.error("No data available for training. Please select at least one indicator.")
    else:
        # Train the model
        model_state_dict = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate, device)
        st.session_state['model_state_dict'] = model_state_dict
        st.session_state['training_params'] = training_params
else:
    model_state_dict = st.session_state['model_state_dict']

# Create a model instance for evaluation and prediction
model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# Add stop training button
if 'stop_training' not in st.session_state:
    st.session_state['stop_training'] = False

def stop_training():
    st.session_state['stop_training'] = True

def reset_training_flag():
    st.session_state['stop_training'] = False

if st.sidebar.button("Stop Training"):
    stop_training()

# Check if the stock symbol has changed
if stockSymbol != st.session_state['stockSymbol']:
    st.session_state['stockSymbol'] = stockSymbol
    reset_training_flag()
    st.experimental_rerun()

# Save the model
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_filename = f'stock_prediction_model_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.pt'
model_path = os.path.join(model_dir, model_filename)
torch.save(model_state_dict, model_path)

# Display the model summary
st.subheader("Model Summary")
st.write(model)

# Fetch and display news headlines for the stock symbol
if use_news:
    st.subheader("Latest News Headlines")
    news_data = fetch_all_news(stockSymbol, dates)
    st.write(f"Fetched {len(news_data)} news articles for {stockSymbol} from {dates.min()} to {dates.max()}")
    with st.expander("Show News Data"):
        st.write(news_data)

# Calculate and display the number of neurons
num_neurons = count_parameters(model)
st.write(f"Number of parameters: {num_neurons}")

# Evaluate the model on the testing set
test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    test_predicted = model(test_X_tensor)
    test_predicted = test_predicted.cpu().numpy()

# Prepare data for visualization
test_actual_data = pd.DataFrame(close_scaler.inverse_transform(test_y[:, 0].reshape(-1, 1)), columns=['Actual'])
test_predicted_data = pd.DataFrame(close_scaler.inverse_transform(test_predicted[:, 0].reshape(-1, 1)), columns=['Predicted'])

# Concatenate the DataFrames
test_chart_data = pd.concat([test_actual_data, test_predicted_data], axis=1)

# Visualize the results on the testing set
st.subheader("Testing Set Predictions")
st.line_chart(test_chart_data)

# Predict future stock prices
st.subheader("Predict Future Stock Prices")
num_days = st.number_input("Number of days to predict", min_value=1, max_value=30, value=7)

if st.button("Predict Future"):
    last_sequence = torch.tensor(test_X[-1], dtype=torch.float32).to(device)
    future_predictions = predict_future(model_state_dict, last_sequence, num_days, close_scaler, input_size, hidden_size, num_layers, num_heads, dropout, device)

    # Create a DataFrame for the future predictions
    future_dates = pd.date_range(start=test_data.index[-1] + timedelta(days=1), periods=num_days)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

    # Combine actual data with future predictions for visualization
    last_actual_price = test_data['Close'].iloc[-1]
    combined_df = pd.concat([
        test_data[['Close']].tail(30).rename(columns={'Close': 'Actual Price'}),
        future_df.set_index('Date')
    ])

    st.write("Future Price Predictions:")
    st.write(future_df)

    st.subheader("Actual vs Predicted Prices")
    st.line_chart(combined_df)

    # Calculate and display the predicted price change
    price_change = future_predictions[-1] - last_actual_price
    percent_change = (price_change / last_actual_price) * 100
    st.write(f"Predicted price change over {num_days} days: ${price_change:.2f} ({percent_change:.2f}%)")

# Download the model
download_model = st.button("Download Model")
if download_model:
    with open(model_path, "rb") as f:
        bytes_data = f.read()
        st.download_button(
            label="Download Model",
            data=bytes_data,
            file_name=model_filename,
            mime="application/octet-stream"
        )
