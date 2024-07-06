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

# Load environment variablesa
dotenv.load_dotenv()

# Set the API keys
alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')

# Set page title
st.set_page_config(page_title="Stock Prediction with News")

st.write(alpaca_api_key_id)
st.write(alpaca_api_secret_key)

header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.page_link("main.py", label="Home", icon="🏠")
with header_col2:
    st.page_link("pages/trading.py", label="Trading", icon="📈")
st.divider()

# Add a title and description
st.title("Stock Prediction using Transformer Model with News")
st.write("This app predicts stock prices using a PyTorch Transformer model and news sentiment analysis.")

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
    end_date = st.date_input("End Date", datetime(2024, 1, 1))

use_volume = st.checkbox("Use Volume Data")
use_news = st.checkbox("Use News Data")

# Load and prepare the data
stock_data, close_scaler = load_data(stockSymbol, start_date, end_date)
train_size = int(len(stock_data) * 0.8)
train_data = stock_data[:train_size]
test_data = stock_data[train_size:]

# Fetch and prepare news data
dates = stock_data.index.date
news_data = fetch_all_news(stockSymbol, dates) if use_news else pd.DataFrame()

# Ensure the input data is properly formatted
seq_length = 20
input_size = 2 if use_volume else 1
input_size += 1 if use_news else 0

# Create sequences for training and testing sets
train_X, train_y = preprocess_data(train_data, news_data, use_volume, use_news, close_scaler, seq_length)
test_X, test_y = preprocess_data(test_data, news_data, use_volume, use_news, close_scaler, seq_length)

# Get user input for training parameters
st.sidebar.header("Training Parameters")
hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=1024, value=512, step=32)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=10, value=3, step=1)
num_heads = st.sidebar.slider("Number of Heads", min_value=0, max_value=600, value=0, step=64)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=10000, value=1, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=240, value=32, step=16)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")

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
    stock_data, scaler = load_data(stockSymbol, start_date, end_date)
    train_size = int(len(stock_data) * 0.8)
    train_data = stock_data[:train_size]
    test_data = stock_data[train_size:]
    news_data = fetch_all_news(stockSymbol, dates) if use_news else pd.DataFrame()
    train_X, train_y = preprocess_data(train_data, news_data, use_volume, use_news, scaler, seq_length)
    test_X, test_y = preprocess_data(test_data, news_data, use_volume, use_news, scaler, seq_length)
    model = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate, device)
    st.experimental_rerun()

# Train the model (cached)
model_state_dict = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate, device)

model_dir = 'models'
model_filename = f'stock_prediction_model_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.pt'
model_path = os.path.join(model_dir, model_filename)
torch.save(model_state_dict, model_path)

# Create a model instance for evaluation and prediction
model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

# Train the model (cached)
if st.button("Rerun Training"):
    reset_training_flag()
    model = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate, device)
    # Save the retrained model
    model_path = os.path.join(model_dir, model_filename + datetime.now().strftime("%Y%m%d%H%M%S"))
    torch.save(model, model_path)
    st.experimental_rerun()

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
    with st.expander("Show News Data"):
        news_container = st.container()
        with news_container:
            for _, row in news_data.iterrows():
                st.write(f"**{row['title']}**")
                st.write(f"[Read more]({row['link']})")
                st.write(f"Published at: {row['publishedAt']}")
                st.write(f"Sentiment: {row['sentiment']}")
                st.write("---")

        st.markdown(
            """
            <style>
            .streamlit-expanderContent {
                max-height: 300px;
                overflow-y: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Calculate and display the number of neurons
num_neurons = count_parameters(model)
st.write(f"Number of neurons: {num_neurons}")

model.to(device)

# Evaluate the model on the testing set
# Ensure data is on the correct device
test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(device)

# Ensure the model is in evaluation mode
model.eval()

# Predictions with MPS handling
with torch.no_grad():
    test_predicted = model(test_X_tensor)
    test_predicted = test_predicted.cpu().numpy()

# Visualize the results on the testing set
test_actual_data = pd.DataFrame(test_y[:, 0], columns=['Actual'])
test_predicted_data = pd.DataFrame(test_predicted[:, 0], columns=['Predicted'])
test_chart_data = pd.concat([test_actual_data, test_predicted_data], axis=1)
st.subheader("Testing Set Predictions")
st.line_chart(test_chart_data)

download_model = st.button("Download Model")
if download_model:
    model_path = os.path.join(model_dir, model_filename)
    if os.path.isfile(model_path):
        with open(model_path, "rb") as f:
            bytes = f.read()
            st.download_button(
                label="Download Model",
                data=bytes,
                file_name=model_filename,
                mime="application/octet-stream"
            )
    else:
        st.error('Model file not found.')

model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
selected_model = st.selectbox("Select Model", model_files)

# Load the model
if selected_model:
    model_path = os.path.join(model_dir, selected_model)
    model_state_dict = torch.load(model_path, map_location=device)
    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    st.success(f"Model {selected_model} loaded successfully!")

# Set the default start and end dates
default_start_date = start_date  # Example start date
default_end_date = end_date  # Example end date

st.write(stock_data)

st.subheader("Predict Future Stock Prices")
num_days = st.number_input("Number of days to predict", min_value=1, max_value=30, value=7)

if st.button("Predict Future"):
    last_sequence = torch.tensor(test_X[-1], dtype=torch.float32).to(device)
    input_size = last_sequence.shape[1]  # Get the input size from the last sequence
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
