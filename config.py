"""
Configuration module for the Stock Prediction Application.

This module defines the available technical indicators and their parameters
for use in the stock prediction model.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from indicators import Indicators

# List of available technical indicators with their configuration
list_of_indicators: List[Dict[str, Any]] = [
    {
        'name': 'Volume',
        'function': None,
        'size': 1,
        'key': 'volume',
        'params': {},
        'function_input': 'none'
    },
    {
        'name': 'News',
        'function': None,
        'size': 1,
        'key': 'news',
        'params': {},
        'function_input': 'none'
    },
    {
        'name': 'Simple Moving Average (SMA)',
        'function': Indicators.sma,
        'size': 1,
        'key': 'sma',
        'params': {'period': 14},
        'function_input': 'series',
        'columns': ['SMA']
    },
    {
        'name': 'Exponential Moving Average (EMA)',
        'function': Indicators.ema,
        'size': 1,
        'key': 'ema',
        'params': {'period': 14},
        'function_input': 'series',
        'columns': ['EMA']
    },
    {
        'name': 'Moving Average Convergence Divergence (MACD)',
        'function': Indicators.macd,
        'size': 2,
        'key': 'macd',
        'params': {'period_long': 26, 'period_short': 12, 'period_signal': 9},
        'function_input': 'dataframe',
        'columns': ['MACD', 'Signal Line']
    },
    {
        'name': 'Relative Strength Index (RSI)',
        'function': Indicators.rsi,
        'size': 1,
        'key': 'rsi',
        'params': {'period': 14},
        'function_input': 'dataframe',
        'columns': ['RSI']
    },
    {
        'name': 'Bollinger Bands (BB)',
        'function': Indicators.bollinger_bands,
        'size': 2,
        'key': 'bb',
        'params': {'period': 20},
        'function_input': 'dataframe',
        'columns': ['Upper Band', 'Lower Band']
    },
    {
        'name': 'Average True Range (ATR)',
        'function': Indicators.atr,
        'size': 1,
        'key': 'atr',
        'params': {'period': 14},
        'function_input': 'dataframe',
        'columns': ['ATR']
    },
    {
        'name': 'Average Directional Index (ADX)',
        'function': Indicators.adx,
        'size': 1,
        'key': 'adx',
        'params': {'period': 14},
        'function_input': 'dataframe',
        'columns': ['ADX']
    },
]


def get_indicator_by_key(key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve an indicator configuration by its key.

    Args:
        key: The unique identifier for the indicator

    Returns:
        The indicator configuration dictionary if found, None otherwise
    """
    for indicator in list_of_indicators:
        if indicator['key'] == key:
            return indicator
    return None
