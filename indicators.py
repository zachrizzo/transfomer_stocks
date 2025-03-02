"""
Technical Indicators Module for Stock Analysis.

This module provides a collection of technical indicators commonly used
in financial analysis and algorithmic trading.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class Indicators:
    """
    A class that provides various technical indicators for stock analysis.

    All methods are implemented as static methods that take financial time series
    data and configuration parameters as inputs and return the calculated indicators.
    """

    @staticmethod
    def sma(data: Union[pd.Series, pd.DataFrame], period: int) -> pd.Series:
        """
        Calculate the Simple Moving Average (SMA).

        Args:
            data: Price data as a pandas Series
            period: The window size for the moving average

        Returns:
            A pandas Series containing the SMA values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
        return data.rolling(window=period).mean().rename('SMA')

    @staticmethod
    def ema(data: Union[pd.Series, pd.DataFrame], period: int) -> pd.Series:
        """
        Calculate the Exponential Moving Average (EMA).

        Args:
            data: Price data as a pandas Series
            period: The window size for the exponential moving average

        Returns:
            A pandas Series containing the EMA values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
        return data.ewm(span=period, adjust=False).mean().rename('EMA')

    @staticmethod
    def macd(data: pd.DataFrame, period_long: int = 26, period_short: int = 12, period_signal: int = 9) -> pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD).

        Args:
            data: DataFrame containing price data with a 'Close' column
            period_long: The window size for the long-term EMA (default: 26)
            period_short: The window size for the short-term EMA (default: 12)
            period_signal: The window size for the signal line (default: 9)

        Returns:
            A DataFrame containing the MACD and Signal Line values
        """
        if period_long <= period_short:
            raise ValueError("Long period must be greater than short period")
        if period_signal <= 0 or period_short <= 0 or period_long <= 0:
            raise ValueError("All periods must be positive integers")

        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column")

        shortEMA = Indicators.ema(data['Close'], period_short)
        longEMA = Indicators.ema(data['Close'], period_long)
        macd = shortEMA - longEMA
        signal = macd.ewm(span=period_signal, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal Line': signal})

    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            data: DataFrame containing price data with a 'Close' column
            period: The window size for the RSI calculation (default: 14)

        Returns:
            A pandas Series containing the RSI values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")

        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column")

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # First calculation with simple moving average
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.finfo(float).eps)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename('RSI')

    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate the Bollinger Bands.

        Args:
            data: DataFrame containing price data with a 'Close' column
            period: The window size for the moving average (default: 20)
            std_dev: Number of standard deviations for the bands (default: 2.0)

        Returns:
            A DataFrame containing the Upper and Lower Bollinger Bands
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")

        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column")

        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return pd.DataFrame({
            'Middle Band': sma,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        })

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR).

        Args:
            data: DataFrame containing price data with 'High', 'Low', and 'Close' columns
            period: The window size for the ATR calculation (default: 14)

        Returns:
            A pandas Series containing the ATR values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")

        required_columns = ['High', 'Low', 'Close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input DataFrame must contain a '{col}' column")

        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.rename('ATR')

    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average Directional Index (ADX).

        Args:
            data: DataFrame containing price data with 'High', 'Low', and 'Close' columns
            period: The window size for the ADX calculation (default: 14)

        Returns:
            A pandas Series containing the ADX values
        """
        if period <= 0:
            raise ValueError("Period must be a positive integer")

        required_columns = ['High', 'Low', 'Close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input DataFrame must contain a '{col}' column")

        high = data['High']
        low = data['Low']
        close = data['Close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate smoothed +DM, -DM and TR
        smoothed_plus_dm = plus_dm.rolling(window=period).sum()
        smoothed_minus_dm = minus_dm.rolling(window=period).sum()
        smoothed_tr = tr.rolling(window=period).sum()

        # Avoid division by zero
        smoothed_tr = smoothed_tr.replace(0, np.finfo(float).eps)

        # Calculate +DI and -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

        # Calculate DX
        dx_numerator = (plus_di - minus_di).abs()
        dx_denominator = (plus_di + minus_di)

        # Avoid division by zero
        dx_denominator = dx_denominator.replace(0, np.finfo(float).eps)

        dx = 100 * (dx_numerator / dx_denominator)

        # Calculate ADX
        adx = dx.rolling(window=period).mean()

        return adx.rename('ADX')
