import pandas as pd
import numpy as np

class Indicators:
    """
    A class that provides various technical indicators for stock analysis.
    """

    def __init__(self):
        """
        Constructor method for the Indicators class.
        """
        pass

    @staticmethod
    def sma(data, period):
        """
        Calculates the Simple Moving Average (SMA) for the given data and period.

        Parameters:
        - data: A pandas Series containing the stock data.
        - period: An integer representing the period for calculating the SMA.

        Returns:
        - A pandas Series representing the SMA values.
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data, period):
        """
        Calculates the Exponential Moving Average (EMA) for the given data and period.

        Parameters:
        - data: A pandas Series containing the stock data.
        - period: An integer representing the period for calculating the EMA.

        Returns:
        - A pandas Series representing the EMA values.
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(data, period_long, period_short, period_signal):
        """
        Calculates the Moving Average Convergence Divergence (MACD) for the given data and periods.

        Parameters:
        - data: A pandas DataFrame containing the stock data.
        - period_long: An integer representing the long period for calculating the MACD.
        - period_short: An integer representing the short period for calculating the MACD.
        - period_signal: An integer representing the period for calculating the signal line.

        Returns:
        - A pandas DataFrame with additional columns for MACD and signal line values.
        """
        shortEMA = Indicators.ema(data['Close'], period_short)
        longEMA = Indicators.ema(data['Close'], period_long)
        data['MACD'] = shortEMA - longEMA
        data['Signal Line'] = Indicators.ema(data['MACD'], period_signal)
        return data

    @staticmethod
    def rsi(data, period):
        """
        Calculates the Relative Strength Index (RSI) for the given data and period.

        Parameters:
        - data: A pandas DataFrame containing the stock data.
        - period: An integer representing the period for calculating the RSI.

        Returns:
        - A pandas DataFrame with an additional column for RSI values.
        """
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        return data

    @staticmethod
    def bollinger_bands(data, period):
        """
        Calculates the Bollinger Bands for the given data and period.

        Parameters:
        - data: A pandas DataFrame containing the stock data.
        - period: An integer representing the period for calculating the Bollinger Bands.

        Returns:
        - A pandas DataFrame with additional columns for SMA, standard deviation, upper band, and lower band values.
        """
        data['SMA'] = Indicators.sma(data['Close'], period)
        data['STD'] = data['Close'].rolling(window=period).std()
        data['Upper Band'] = data['SMA'] + (data['STD'] * 2)
        data['Lower Band'] = data['SMA'] - (data['STD'] * 2)
        return data

    @staticmethod
    def atr(data, period):
        """
        Calculates the Average True Range (ATR) for the given data and period.

        Parameters:
        - data: A pandas DataFrame containing the stock data.
        - period: An integer representing the period for calculating the ATR.

        Returns:
        - A pandas DataFrame with additional columns for ATR values.
        """
        data['H-L'] = abs(data['High'] - data['Low'])
        data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
        data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
        data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        data['ATR'] = data['TR'].rolling(window=period).mean()
        return data

    @staticmethod
    def adx(data, period):
        """
        Calculates the Average Directional Index (ADX) for the given data and period.

        Parameters:
        - data: A pandas DataFrame containing the stock data.
        - period: An integer representing the period for calculating the ADX.

        Returns:
        - A pandas DataFrame with additional columns for ADX values.
        """
        data['H-L'] = data['High'] - data['Low']
        data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
        data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
        data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        data['DMplus'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), data['High'] - data['High'].shift(1), 0)
        data['DMminus'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), data['Low'].shift(1) - data['Low'], 0)
        data['DIplus'] = (data['DMplus'].rolling(window=period).mean() / data['TR']) * 100
        data['DIminus'] = (data['DMminus'].rolling(window=period).mean() / data['TR']) * 100
        data['DX'] = (abs(data['DIplus'] - data['DIminus']) / abs(data['DIplus'] + data['DIminus'])) * 100
        data['ADX'] = data['DX'].rolling(window=period).mean()
        return data
