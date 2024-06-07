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
alpaca_api_key_id = os.getenv('ALPACA_PAPER_KEY_ID')
alpaca_api_secret_key = os.getenv('ALPACA_PAPER_SECRET')

# Set page title
st.set_page_config(page_title="Stock Prediction with News")

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
    unique_dates = list(stock_data_dates)
    print(stock_data_dates)

    for i in range(0, len(unique_dates), 50):
        start_date = unique_dates[i]
        end_date = unique_dates[min(i + 49, len(unique_dates) - 1)]
        batch_news = fetch_news_batch(stockSymbol, start_date, end_date)
        all_news = pd.concat([all_news, batch_news])

    return all_news

@st.cache_resource
def load_data(stockSymbol, start_date, end_date):
    stock_data = yf.download(stockSymbol, start=start_date, end=end_date)
    st.write(stock_data)
    print(stock_data.head())
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['Normalized_Close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    return stock_data, scaler

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

def preprocess_data(stock_data, news_data, use_volume, use_news, scaler, seq_length):
    if use_volume:
        stock_data['Normalized_Volume'] = scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))
        data_combined = np.column_stack((stock_data['Normalized_Close'].values, stock_data['Normalized_Volume'].values))
    else:
        data_combined = stock_data['Normalized_Close'].values.reshape(-1, 1)

    if use_news and not news_data.empty:
        news_data = news_data.set_index('publishedAt')
        stock_data = stock_data.join(news_data['sentiment'], how='left')
        stock_data['sentiment'] = stock_data['sentiment'].fillna(0)

        # Ensure the lengths match
        if len(stock_data) != len(data_combined):
            stock_data = stock_data.iloc[-len(data_combined):]

        data_combined = np.column_stack((data_combined, stock_data['sentiment'].values))

    # Ensure data_combined has the correct shape
    expected_shape = (len(stock_data) - seq_length, seq_length, data_combined.shape[1])
    if data_combined.shape[0] < seq_length:
        st.error(f"Insufficient data points. Required at least {seq_length + 1} but got {data_combined.shape[0]}.")
        return np.empty(expected_shape), np.empty((0,))

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

    return model

# Load and prepare the data
stock_data, scaler = load_data(stockSymbol, start_date, end_date)
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
train_X, train_y = preprocess_data(train_data, news_data, use_volume, use_news, scaler, seq_length)
test_X, test_y = preprocess_data(test_data, news_data, use_volume, use_news, scaler, seq_length)

# Get user input for training parameters
st.sidebar.header("Training Parameters")
hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=1024, value=512, step=32)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=10, value=3, step=1)
num_heads = st.sidebar.slider("Number of Heads", min_value=4, max_value=100, value=4, step=4)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=10000, value=1, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
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

# Train the model (cached)
model = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)

model_dir = 'models'
model_filename = 'stock_prediction_model'
model_path = os.path.join(model_dir, model_filename) + f'{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.pt'
torch.save(model.state_dict(), model_path)

# Train the model (cached)
if st.button("Rerun Training"):
    reset_training_flag()
    model = train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)
    # Save the retrained model
    model_path = os.path.join(model_dir, model_filename + datetime.now().strftime("%Y%m%d%H%M%S"))
    torch.save(model.state_dict(), model_path)
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Ensure the model is in evaluation mode
    st.success(f"Model {selected_model} loaded successfully!")

# Set the default start and end dates
default_start_date = start_date  # Example start date
default_end_date = end_date  # Example end date

# Load and scale stock data
stock_data, scaler = load_data(stockSymbol, start_date, end_date)

# Input for prediction period
prediction_period = st.number_input('Enter the number of days for prediction:', min_value=1, max_value=365, value=10)

# Ensure we have enough data for the model
seq_length = 20  # Adjust this to match the sequence length used in your model
total_period = seq_length + prediction_period

# Ensure end_date is set to the last date in stock_data
end_date = stock_data.index[-1]  # Adjust this line according to your actual end_date
future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=prediction_period).date  # Generate future dates

# Create a DataFrame for future dates
future_data = stock_data.copy()
future_data = future_data.append(pd.DataFrame(index=future_dates))

# Fill any NaN values in future_data
future_data.fillna(method='ffill', inplace=True)

# Fetch and prepare news data for the future dates
future_news_data = fetch_all_news(stockSymbol, future_dates) if use_news else pd.DataFrame()

# Ensure the input data is properly formatted
future_X, _ = preprocess_data(future_data, future_news_data, use_volume, use_news, scaler, seq_length)

# Debugging: Print shapes to verify
st.write("Future data shape:", future_data.shape)
st.write("Future news data shape:", future_news_data.shape)
st.write("Preprocessed future_X shape:", future_X.shape)

# Dynamically set input size based on user selections
input_size = 1  # Base input size for 'Normalized_Close'
if use_volume:
    input_size += 1  # Add volume data if selected
if use_news:
    input_size += 1  # Add news sentiment if selected

# Ensure future_X is reshaped to match model input expectations
future_X_tensor = torch.tensor(future_X, dtype=torch.float32).to(device)

# Ensure the tensor shape is (batch_size, sequence_length, input_size)
batch_size, seq_length, actual_input_size = future_X_tensor.shape

# Check if input size matches expected size before reshaping
if actual_input_size != input_size:
    st.error(f"Shape mismatch: future_X_tensor cannot be reshaped to ({batch_size}, {seq_length}, {input_size}) because actual input size is {actual_input_size}")
else:
    future_X_tensor = future_X_tensor.view(batch_size, seq_length, input_size)

# Debugging: Print the tensor to verify no NaN values are present
st.write("Initial future_X_tensor:", future_X_tensor)

# Predict the future prices one step at a time
future_predictions = []
for _ in range(prediction_period):
    with torch.no_grad():
        future_pred = model(future_X_tensor)
        st.write("Future prediction tensor:", future_pred)  # Debugging: Print the prediction tensor
        future_predictions.append(future_pred.cpu().numpy()[-1, 0])  # Ensure to take the correct prediction value

    # Check for NaN values in the prediction
    if torch.isnan(future_pred).any():
        st.error("NaN values found in future prediction tensor.")
        break

    # Append the predicted value to the future data for the next prediction
    new_data = torch.cat([future_X_tensor[:, 1:, :], future_pred[:, None, :]], dim=1)
    future_X_tensor = new_data

# Ensure the predicted values match the prediction period
future_predicted = np.array(future_predictions).reshape(-1, 1)  # Ensure the shape is correct for inverse transform

# Check the length of future_dates
future_dates_for_index = pd.date_range(start=end_date + timedelta(days=1), periods=prediction_period).date

# Debugging: Print lengths to verify
st.write("Future dates for index length:", len(future_dates_for_index))
st.write("Future predicted length:", len(future_predicted))

if len(future_dates_for_index) != len(future_predicted):
    st.error(f"Length mismatch: future_dates_for_index has {len(future_dates_for_index)} elements, but future_predicted has {len(future_predicted)} elements.")
else:
    # Inverse transform the predicted values
    # Create a scaler for inverse transformation based only on the Close prices
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]

    predicted_df = pd.DataFrame(future_predicted, columns=['Normalized_Close'])
    predicted_df['Predicted_Close'] = close_scaler.inverse_transform(predicted_df[['Normalized_Close']])
    predicted_df.index = future_dates_for_index

    # Save numeric predicted values for plotting
    numeric_predicted_close = predicted_df['Predicted_Close'].copy()

    # Convert predicted values to monetary form for display
    predicted_df['Predicted_Close'] = predicted_df['Predicted_Close'].apply(lambda x: f"${x:,.2f}")

    # Display the predicted prices
    st.subheader(f"Predicted Stock Prices for the Next {prediction_period} Days")
    st.write(predicted_df)

    # Plot the future prices
    st.subheader("Future Stock Prices")

    # Concatenate the last 10 days of actual data with the predictions
    last_10_days = stock_data[['Close']].iloc[-10:]
    future_chart_data = pd.concat([last_10_days, numeric_predicted_close], axis=1)
    future_chart_data.columns = ['Actual_Close', 'Predicted_Close']

    # Plotting using st.line_chart
    st.line_chart(future_chart_data)
