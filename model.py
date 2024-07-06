# model.py
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



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
def train_model(train_X, train_y, test_X, test_y, input_size, hidden_size, num_layers, num_heads, dropout, num_epochs, batch_size, learning_rate, device):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


