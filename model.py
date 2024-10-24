import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

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
        self.fc = nn.Linear(hidden_size, input_size)  # Output same size as input

    def forward(self, x):
        x = x.float()
        x = x.view(x.size(0), -1, self.input_size)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]
        x = self.fc(x)
        x = torch.sigmoid(x)  # Added sigmoid activation
        return x

def train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate, device):
    if train_X.size == 0 or train_y.size == 0:
        st.error("No data available for training. Please select at least one indicator.")
        return None

    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(device)

    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Progress bar for epochs
    progress_bar = st.sidebar.progress(0)
    epoch_text = st.sidebar.empty()

    for epoch in range(num_epochs):
        if st.session_state.get('stop_training', False):
            break

        model.train()
        permutation = torch.randperm(train_X.size(0))

        for i in range(0, train_X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = train_X[indices].to(device)
            batch_y = train_y[indices].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Update progress bar and text
        progress_bar.progress((epoch + 1) / num_epochs)
        epoch_text.text(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model.state_dict()

def predict_future(model_state_dict, last_sequence, num_days, close_scaler, input_size, hidden_size, num_layers, num_heads, dropout, device):
    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    future_predictions = []
    current_sequence = last_sequence.clone()

    for _ in range(num_days):
        with torch.no_grad():
            prediction = model(current_sequence.unsqueeze(0))
        close_prediction = prediction[0, 0].item()
        future_predictions.append(close_prediction)

        # Create a new datapoint
        new_datapoint = torch.zeros(input_size, device=device)
        new_datapoint[0] = close_prediction

        # Copy the last known values for other features
        if input_size > 1:
            new_datapoint[1:] = current_sequence[-1, 1:]

        # Update the sequence
        current_sequence = torch.cat((current_sequence[1:], new_datapoint.unsqueeze(0)), dim=0)

    # Denormalize the predictions
    future_predictions = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
