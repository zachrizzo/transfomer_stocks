import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

# Set page title
st.set_page_config(page_title="Stock Prediction")

# Add a title and description
st.title("Stock Prediction using Transformer Model")
st.write("This app predicts stock prices using a PyTorch Transformer model.")

# Initialize the stock symbol variable
if 'stockSymbol' not in st.session_state:
    st.session_state['stockSymbol'] = 'NVDA'

# Check if MPS is available and use it; otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

stockSymbol = st.text_input("Enter Stock Symbol", st.session_state['stockSymbol'])
use_volume = st.checkbox("Use Volume Data")

def load_data(stockSymbol):
    tesla_data = yf.download(stockSymbol, start='2018-01-01', end='2023-01-01')
    st.write(tesla_data.head())
    print(tesla_data.head())
    scaler = MinMaxScaler(feature_range=(0, 1))
    tesla_data['Normalized_Close'] = scaler.fit_transform(tesla_data['Close'].values.reshape(-1, 1))
    return tesla_data, scaler

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

def preprocess_data(data, use_volume, scaler, seq_length):
    if use_volume:
        data['Normalized_Volume'] = scaler.fit_transform(data['Volume'].values.reshape(-1, 1))
        data_combined = np.column_stack((data['Normalized_Close'].values, data['Normalized_Volume'].values))
    else:
        data_combined = data['Normalized_Close'].values.reshape(-1, 1)

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
def train_model(train_X, train_y, test_X, test_y, use_volume, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate):
    input_size = 2 if use_volume else 1
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
        model.train()
        for i in range(0, train_X.size(0), batch_size):
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

        # Test the model every 500 epochs
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
tesla_data, scaler = load_data(stockSymbol)
train_size = int(len(tesla_data) * 0.8)
train_data = tesla_data[:train_size]
test_data = tesla_data[train_size:]

# Ensure the input data is properly formatted
seq_length = 20

# Create sequences for training and testing sets
train_X, train_y = preprocess_data(train_data, use_volume, scaler, seq_length)
test_X, test_y = preprocess_data(test_data, use_volume, scaler, seq_length)

# Get user input for training parameters
st.sidebar.header("Training Parameters")
hidden_size = st.sidebar.slider("Hidden Size", min_value=32, max_value=1024, value=512, step=32)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=10, value=3, step=1)
num_heads = st.sidebar.slider("Number of Heads", min_value=4, max_value=100, value=4, step=4)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=100000, value=1, step=1)
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")

# Check if the stock symbol has changed
if stockSymbol != st.session_state['stockSymbol']:
    st.session_state['stockSymbol'] = stockSymbol
    tesla_data, scaler = load_data(stockSymbol)
    train_size = int(len(tesla_data) * 0.8)
    train_data = tesla_data[:train_size]
    test_data = tesla_data[train_size:]
    train_X, train_y = preprocess_data(train_data, use_volume, scaler, seq_length)
    test_X, test_y = preprocess_data(test_data, use_volume, scaler, seq_length)
    model = train_model(train_X, train_y, test_X, test_y, use_volume, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)
    st.experimental_rerun()

# Train the model (cached)
model = train_model(train_X, train_y, test_X, test_y, use_volume, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)

# Save the trained model
model_dir = '/Users/zachrizzo/programing/aI_assistant/stocks/models'
model_filename = 'stock_prediction_model.pt'
model_path = os.path.join(model_dir, model_filename)
torch.save(model.state_dict(), model_path)

if st.button("Rerun Training"):
    model = train_model(train_X, train_y, test_X, test_y, use_volume, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate)
    # Save the retrained model
    torch.save(model.state_dict(), model_path)
    st.experimental_rerun()

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

if selected_model:
    model_path = os.path.join(model_dir, selected_model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    st.success(f"Model {selected_model} loaded successfully!")