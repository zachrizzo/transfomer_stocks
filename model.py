"""
Neural Network Model Module for Stock Price Prediction.

This module defines the Transformer-based model architecture and provides
functions for training the model and making predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import time
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from typing import Dict, Optional, Union, Tuple, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransformerModel(nn.Module):
    """
    Transformer-based neural network model for time series forecasting.

    This model uses a transformer encoder architecture to process sequential data
    and predict the next value in the sequence.

    Important Notes:
    - The model takes sequences of multi-feature data but predicts only a single value.
    - The input shape is [batch_size, seq_length, input_dim] where input_dim can be any number of features.
    - The output shape is [batch_size, output_dim] where output_dim is typically 1 for stock price prediction.
    - When using this model, ensure your target data has the same number of features as output_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, num_heads: int, dropout: float):
        """
        Initialize the transformer model.

        Args:
            input_dim: Number of input features
            hidden_dim: Size of hidden layers
            output_dim: Number of output features (typically 1 for stock price prediction)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
        """
        super(TransformerModel, self).__init__()

        # Validate input parameters
        if num_heads > hidden_dim:
            raise ValueError(f"Number of attention heads ({num_heads}) must be less than or equal to hidden dimension ({hidden_dim})")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension ({hidden_dim}) must be divisible by number of heads ({num_heads})")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Define model layers
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=False  # Input expected in shape [seq_length, batch_size, hidden_dim]
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        x = x.float()
        x = self.embedding(x)  # [batch_size, seq_length, hidden_dim]
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, hidden_dim]
        x = self.transformer_encoder(x)  # [seq_length, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_length, hidden_dim]
        x = x[:, -1, :]  # [batch_size, hidden_dim] - Take only the last sequence step
        x = self.fc(x)  # [batch_size, output_dim]
        return x

    def __str__(self) -> str:
        """
        String representation of the model.

        Returns:
            Description of the model architecture
        """
        return (
            f"TransformerModel(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  num_heads={self.num_heads},\n"
            f"  layers={len(self.transformer_encoder.layers)},\n"
            f"  parameters={count_parameters(self):,}\n"
            f")"
        )


def train_model(
    model: TransformerModel,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    device: torch.device,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
) -> Tuple[List[float], List[float]]:
    """
    Train the transformer model.

    Args:
        model: The TransformerModel instance to train
        train_X: Training input features
        train_y: Training target values
        val_X: Validation input features
        val_y: Validation target values
        device: Device to train on
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (training_losses, validation_losses)
    """
    logger.info(f"Training model for {num_epochs} epochs with batch size {batch_size}")

    # Move data to device
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Create batches
        num_batches = len(train_X) // batch_size

        for i in range(0, len(train_X), batch_size):
            # Get batch
            batch_X = train_X[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()
            val_losses.append(val_loss)

        # Update progress bar and status text
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check for early stopping
        if 'stop_training' in st.session_state and st.session_state['stop_training']:
            logger.info("Early stopping requested by user")
            status_text.text(f"Training stopped at epoch {epoch+1}/{num_epochs}")
            break

    # Clear progress bar and status text
    progress_bar.empty()
    status_text.empty()

    return train_losses, val_losses


def predict_future(
    model_state_dict: Dict[str, torch.Tensor],
    last_sequence: torch.Tensor,
    num_days: int,
    close_scaler: MinMaxScaler,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    device: torch.device
) -> Optional[np.ndarray]:
    """
    Predict future stock prices using the trained model.

    Args:
        model_state_dict: Trained model state dictionary
        last_sequence: Last known sequence of data
        num_days: Number of days to predict
        close_scaler: Scaler used to normalize close prices
        input_dim: Number of input features
        hidden_dim: Size of hidden layers
        output_dim: Number of output features
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        device: Device to run the model on

    Returns:
        Array of predicted prices
    """
    try:
        seq_length = last_sequence.shape[0]

        # Check if input dimensions match
        if last_sequence.shape[1] != input_dim:
            logger.error(f"Last sequence has incorrect shape: {last_sequence.shape}. Expected ({seq_length}, {input_dim})")
            st.error(f"Last sequence has incorrect shape: {last_sequence.shape}. Expected ({seq_length}, {input_dim})")
            return None

        # Initialize model and load weights
        model = TransformerModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        # Initialize sequence for prediction
        future_predictions = []
        current_sequence = last_sequence.clone().unsqueeze(0)  # Shape: (1, seq_length, input_dim)

        # Generate predictions for each future day
        with torch.no_grad():
            for _ in range(num_days):
                # Get prediction for next day
                prediction = model(current_sequence)
                close_prediction = prediction[0, 0].item()
                future_predictions.append(close_prediction)

                # Create a new datapoint
                new_datapoint = torch.zeros((1, 1, input_dim), device=device)  # Shape: (1, 1, input_dim)
                new_datapoint[0, 0, 0] = close_prediction  # Set the 'Close' price

                # For any additional features, copy the last known values
                if input_dim > 1:
                    new_datapoint[0, 0, 1:] = current_sequence[0, -1, 1:]

                # Update sequence by removing oldest day and adding new prediction
                current_sequence = torch.cat([current_sequence[:, 1:, :], new_datapoint], dim=1)

        # Convert normalized predictions back to original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = close_scaler.inverse_transform(future_predictions).flatten()

        return future_predictions

    except Exception as e:
        logger.error(f"Error in predict_future: {str(e)}", exc_info=True)
        st.error(f"Error predicting future prices: {str(e)}")
        return None


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
