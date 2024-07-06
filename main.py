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

# Load environment variables
dotenv.load_dotenv()

# Set the API keys
alpaca_api_key_id = os.getenv('ALPACA_LIVE_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_LIVE_SECRET')



# Set page title
st.set_page_config(page_title="Stock Prediction with News")

st.write (alpaca_api_key_id)
st.write (alpaca_api_secret_key)

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

@st.cache_resource
def fetch_news_batch(stockSymbol, start_date, end_date):
    url = f"https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&symbols={stockSymbol}&sort=desc&limit=50"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": alpaca_api_key_id,
        "APCA-API-SECRET-KEY": alpaca_api_secret_key
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as err:
        st.error(f"Error fetching news: {err}")
        return pd.DataFrame()
    except ValueError:
        st.error("Error parsing the JSON response")
        return pd.DataFrame()

    articles = []
    for item in news_data.get('news', []):
        articles.append({
            'title': item['headline'],
            'link': item['url'],
            'publishedAt': item['created_at'],
            'sentiment': TextBlob(item['headline']).sentiment.polarity
        })
    news_df = pd.DataFrame(articles)
    if not news_df.empty:
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt']).dt.date
    return news_df

@st.cache_resource
def fetch_all_news(stockSymbol, stock_data_dates):
    all_news = pd.DataFrame()
    stock_data_dates = pd.to_datetime(stock_data_dates).date
    unique_dates = sorted(set(stock_data_dates))  # Ensure dates are unique and sorted

    for i in range(0, len(unique_dates), 50):
        start_date = unique_dates[i]
        end_date = unique_dates[min(i + 49, len(unique_dates) - 1)]
        batch_news = fetch_news_batch(stockSymbol, start_date, end_date)
        all_news = pd.concat([all_news, batch_news])

    # Ensure the index is a DatetimeIndex and remove any duplicates
    all_news.index = pd.to_datetime(all_news['publishedAt']).dt.date
    all_news = all_news[~all_news.index.duplicated(keep='first')]

    return all_news

@st.cache_resource
def load_data(stockSymbol, start_date, end_date):
    stock_data = yf.download(stockSymbol, start=start_date, end=end_date)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['Normalized_Close'] = close_scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    if 'Volume' in stock_data.columns:
        volume_scaler = MinMaxScaler(feature_range=(0, 1))
        stock_data['Normalized_Volume'] = volume_scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))

    return stock_data, close_scaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict_future(model_state_dict, last_sequence, num_days, close_scaler, input_size, hidden_size, num_layers, num_heads, dropout):
    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    future_predictions = []
    current_sequence = last_sequence.clone()

    for _ in range(num_days):
        with torch.no_grad():
            prediction = model(current_sequence.unsqueeze(0))

        # Extract only the close price prediction (first element)
        close_prediction = prediction[0, 0].item()
        future_predictions.append(close_prediction)

        # Create a new datapoint with the correct number of features
        new_datapoint = torch.zeros(input_size, device=device)
        new_datapoint[0] = close_prediction

        # If using volume, set a placeholder value (e.g., last known volume)
        if input_size > 1:
            new_datapoint[1] = current_sequence[-1, 1].item()

        # If using news, set a neutral sentiment (0.5 after normalization)
        if input_size > 2:
            new_datapoint[2] = 0.5

        # Update the sequence for the next prediction
        current_sequence = torch.cat((current_sequence[1:], new_datapoint.unsqueeze(0)), dim=0)

    # Denormalize the predictions
    future_predictions = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

def preprocess_data(stock_data, news_data, use_volume, use_news, close_scaler, seq_length):
    features = [stock_data['Normalized_Close'].values.reshape(-1, 1)]

    if use_volume and 'Normalized_Volume' in stock_data.columns:
        features.append(stock_data['Normalized_Volume'].values.reshape(-1, 1))

    if use_news and not news_data.empty:
        # Ensure news_data has the same index as stock_data
        news_data = news_data.reindex(stock_data.index, fill_value=0)

        # Add sentiment to stock_data
        stock_data['sentiment'] = news_data['sentiment']

        # Normalize sentiment
        sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_sentiment = sentiment_scaler.fit_transform(stock_data['sentiment'].values.reshape(-1, 1))
        features.append(normalized_sentiment)

    # Combine all features
    data_combined = np.hstack(features)

    X, y = create_sequences(data_combined, seq_length)
    return X, y

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1, self.input_size)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

@st.cache_resource
def train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate):
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(device)

    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create placeholders for live updating
    progress_text = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)

    total_steps = num_epochs * (train_X.size(0) // batch_size)
    current_step = 0

    # Initialize a placeholder for the chart
    chart_placeholder = st.empty()

    for epoch in range(num_epochs):
        if st.session_state.get('stop_training', False):
            break  # Exit the training loop if stop button is pressed

        model.train()
        for i in range(0, train_X.size(0), batch_size):
            if st.session_state.get('stop_training', False):
                break  # Exit the training loop if stop button is pressed

            batch_X = train_X[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            current_step += 1

            # Update progress text and progress bar
            if current_step % 10 == 0:
                progress_text.text(f"Epoch [{epoch+1}/{num_epochs}], Step [{current_step}/{total_steps}], Loss: {loss.item():.4f}")
            progress_bar.progress((i + 1) / len(train_X))

        # Test the model every epoch
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                test_predicted = model(test_X_tensor)
                test_predicted = test_predicted.cpu().numpy()
                test_actual_data = pd.DataFrame(test_y[:, 0], columns=['Actual'])
                test_predicted_data = pd.DataFrame(test_predicted[:, 0], columns=['Predicted'])
                test_chart_data = pd.concat([test_actual_data, test_predicted_data], axis=1)

                # Update the chart with the new data
                chart_placeholder.line_chart(test_chart_data, width=800, height=800)

    return model.state_dict()


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
    model = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)
    st.experimental_rerun()


# When saving the model, save the state_dict instead of the entire model
model_state_dict = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)

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
    model = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)
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

# Normalize data function
def normalize(data, mean, std):
    return (data - mean) / std

# Denormalize data function
def denormalize(data, mean, std):
    return data * std + mean

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
    future_predictions = predict_future(model_state_dict, last_sequence, num_days, close_scaler, input_size, hidden_size, num_layers, num_heads, dropout)

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
