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
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float):
        """
        Initialize the transformer model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
        """
        super(TransformerModel, self).__init__()

        # Validate input parameters
        if num_heads > hidden_size:
            raise ValueError(f"Number of attention heads ({num_heads}) must be less than or equal to hidden size ({hidden_size})")
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by number of heads ({num_heads})")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Define model layers
        self.embedding = nn.Linear(input_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=False  # Input expected in shape [seq_length, batch_size, hidden_size]
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_size, 1)  # Output single value for 'Close' prediction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer model.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size]

        Returns:
            Output tensor of shape [batch_size, 1]
        """
        x = x.float()
        x = self.embedding(x)  # [batch_size, seq_length, hidden_size]
        x = x.permute(1, 0, 2)  # [seq_length, batch_size, hidden_size]
        x = self.transformer_encoder(x)  # [seq_length, batch_size, hidden_size]
        x = x.permute(1, 0, 2)  # [batch_size, seq_length, hidden_size]
        x = x[:, -1, :]  # [batch_size, hidden_size] - Take only the last sequence step
        x = self.fc(x)  # [batch_size, 1]
        return x

    def __str__(self) -> str:
        """
        String representation of the model.

        Returns:
            Description of the model architecture
        """
        return (
            f"TransformerModel(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_heads={self.num_heads},\n"
            f"  layers={len(self.transformer_encoder.layers)},\n"
            f"  parameters={count_parameters(self):,}\n"
            f")"
        )


def train_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Train the transformer model.

    Args:
        train_X: Training input features
        train_y: Training target values
        test_X: Testing input features
        test_y: Testing target values
        input_size: Number of input features
        hidden_size: Size of hidden layers
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on (CPU, CUDA, or MPS)

    Returns:
        Model state dictionary if training successful, None otherwise
    """
    # Validate input data
    if train_X.size == 0 or train_y.size == 0:
        logger.error("No data available for training")
        st.error("No data available for training. Please select at least one indicator.")
        return None

    try:
        # Convert numpy arrays to PyTorch tensors
        train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

        # Move test tensors to the selected device
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
        test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(device)

        logger.info(f"Training model with input_size={input_size}, hidden_size={hidden_size}, "
                    f"num_layers={num_layers}, num_heads={num_heads}")

        # Initialize model, loss function and optimizer
        model = TransformerModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Set up progress tracking
        progress_bar = st.sidebar.progress(0)
        epoch_text = st.sidebar.empty()
        loss_history = []

        # Display model architecture
        logger.info(f"Model architecture: {model}")
        logger.info(f"Training on device: {device}")
        logger.info(f"Number of parameters: {count_parameters(model):,}")

        # Training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            if st.session_state.get('stop_training', False):
                logger.info("Training stopped by user")
                break

            # Set model to training mode
            model.train()
            epoch_loss = 0.0

            # Create random permutation for batch sampling
            permutation = torch.randperm(train_X_tensor.size(0))

            # Process mini-batches
            for i in range(0, train_X_tensor.size(0), batch_size):
                # Select batch indices
                indices = permutation[i:i+batch_size]

                # Move batch data to the selected device
                batch_X = train_X_tensor[indices].to(device)
                batch_y = train_y_tensor[indices].to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(indices)

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(train_X_tensor)
            loss_history.append(avg_epoch_loss)

            # Evaluate on validation set periodically
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(test_X_tensor)
                    test_loss = criterion(test_outputs, test_y_tensor)
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, "
                                f"Test Loss: {test_loss.item():.6f}")

            # Update progress bar and status text
            progress_bar.progress((epoch + 1) / num_epochs)
            epoch_text.text(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")

        # Training completed
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")

        return model.state_dict()

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        st.error(f"Error during model training: {str(e)}")
        return None


def predict_future(
    model_state_dict: Dict[str, torch.Tensor],
    last_sequence: torch.Tensor,
    num_days: int,
    close_scaler: MinMaxScaler,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    device: torch.device
) -> Optional[np.ndarray]:
    """
    Predict future stock prices.

    Args:
        model_state_dict: Trained model state dictionary
        last_sequence: Last observed sequence
        num_days: Number of days to predict
        close_scaler: Scaler used to normalize close prices
        input_size: Number of input features
        hidden_size: Size of hidden layers
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        device: Device to run prediction on

    Returns:
        Array of predicted stock prices, or None if prediction fails
    """
    try:
        seq_length = last_sequence.shape[0]

        # Check if input dimensions match
        if last_sequence.shape[1] != input_size:
            logger.error(f"Last sequence has incorrect shape: {last_sequence.shape}. Expected ({seq_length}, {input_size})")
            st.error(f"Last sequence has incorrect shape: {last_sequence.shape}. Expected ({seq_length}, {input_size})")
            return None

        # Initialize model and load weights
        model = TransformerModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()

        logger.info(f"Predicting future prices for {num_days} days")

        # Initialize sequence for prediction
        future_predictions = []
        current_sequence = last_sequence.clone().unsqueeze(0)  # Shape: (1, seq_length, input_size)

        # Generate predictions for each future day
        with torch.no_grad():
            for day in range(num_days):
                # Get prediction for next day
                prediction = model(current_sequence)
                close_prediction = prediction[0, 0].item()
                future_predictions.append(close_prediction)

                # Create a new datapoint
                new_datapoint = torch.zeros((1, 1, input_size), device=device)  # Shape: (1, 1, input_size)
                new_datapoint[0, 0, 0] = close_prediction  # Set the 'Close' price

                # For any additional features, copy the last known values
                if input_size > 1:
                    new_datapoint[0, 0, 1:] = current_sequence[0, -1, 1:]

                # Update the sequence for the next prediction by removing the oldest time step
                # and adding the new prediction
                current_sequence = torch.cat((current_sequence[:, 1:, :], new_datapoint), dim=1)

        # Denormalize the predictions to get actual stock prices
        future_predictions = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        logger.info(f"Prediction complete: {future_predictions.flatten()}")
        return future_predictions.flatten()

    except Exception as e:
        logger.error(f"Error during future prediction: {str(e)}", exc_info=True)
        st.error(f"Failed to generate future predictions: {str(e)}")
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
