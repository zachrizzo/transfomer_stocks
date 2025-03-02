from indicators import Indicators

list_of_indicators = [
    {'name': 'Volume', 'function': None, 'size': 1, 'key': 'volume', 'params': {}, 'function_input': 'none'},
    {'name': 'News', 'function': None, 'size': 1, 'key': 'news', 'params': {}, 'function_input': 'none'},
    {'name': 'Simple Moving Average (SMA)', 'function': Indicators.sma, 'size': 1, 'key': 'sma', 'params': {'period': 14}, 'function_input': 'series', 'columns': ['SMA']},
    {'name': 'Exponential Moving Average (EMA)', 'function': Indicators.ema, 'size': 1, 'key': 'ema', 'params': {'period': 14}, 'function_input': 'series', 'columns': ['EMA']},
    {'name': 'Moving Average Convergence Divergence (MACD)', 'function': Indicators.macd, 'size': 2, 'key': 'macd', 'params': {'period_long': 26, 'period_short': 12, 'period_signal': 9}, 'function_input': 'dataframe', 'columns': ['MACD', 'Signal Line']},
    {'name': 'Relative Strength Index (RSI)', 'function': Indicators.rsi, 'size': 1, 'key': 'rsi', 'params': {'period': 14}, 'function_input': 'dataframe', 'columns': ['RSI']},
    {'name': 'Bollinger Bands (BB)', 'function': Indicators.bollinger_bands, 'size': 2, 'key': 'bb', 'params': {'period': 20}, 'function_input': 'dataframe', 'columns': ['Upper Band', 'Lower Band']},
]

def get_indicator_by_key(key):
    for indicator in list_of_indicators:
        if indicator['key'] == key:
            return indicator
    return None
