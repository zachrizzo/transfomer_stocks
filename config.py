# config.py
import json
from datetime import datetime
from indicators import Indicators

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'hidden_size': 512,
            'num_layers': 3,
            'num_heads': 0,
            'dropout': 0.1,
            'num_epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.001,
            'start_date': datetime(2018, 1, 1).date(),
            'end_date': datetime(2024, 1, 1).date()
        }

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)

list_of_indicators = [
    {'name': 'Volume', 'function': None, 'size': 1, 'key': 'volume', 'params': {}},
    {'name': 'News', 'function': None, 'size': 1, 'key': 'news', 'params': {}},
    {'name': 'Simple Moving Average (SMA)', 'function': Indicators.sma, 'size': 1, 'key': 'sma', 'params': {'period': 14}},
    {'name': 'Exponential Moving Average (EMA)', 'function': Indicators.ema, 'size': 1, 'key': 'ema', 'params': {'period': 14}},
    {'name': 'Moving Average Convergence Divergence (MACD)', 'function': Indicators.macd, 'size': 2, 'key': 'macd', 'params': {'period_long': 26, 'period_short': 12, 'period_signal': 9}},
    {'name': 'Relative Strength Index (RSI)', 'function': Indicators.rsi, 'size': 1, 'key': 'rsi', 'params': {'period': 14}},
    {'name': 'Bollinger Bands (BB)', 'function': Indicators.bollinger_bands, 'size': 2, 'key': 'bb', 'params': {'period': 20}},
]

def get_indicator_by_key(key):
    for indicator in list_of_indicators:
        if indicator['key'] == key:
            return indicator
    return None
