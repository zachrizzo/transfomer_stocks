"""
Live Trading Page for the Stock Prediction Application.

This page provides an interface for live trading with Alpaca using the model predictions.
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import from main modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our live trading module
from live_trading import initialize_live_trading

# Set page config
st.set_page_config(
    page_title="Live Trading | Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

# Initialize the live trading interface
initialize_live_trading()
