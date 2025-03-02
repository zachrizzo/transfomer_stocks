import pandas as pd
import numpy as np

class Indicators:
    """
    A class that provides various technical indicators for stock analysis.
    """

    @staticmethod
    def sma(data, period):
        """
        Calculates the Simple Moving Average (SMA) for the given data and period.
        """
        return data.rolling(window=period).mean().rename('SMA')

    @staticmethod
    def ema(data, period):
        """
        Calculates the Exponential Moving Average (EMA) for the given data and period.
        """
        return data.ewm(span=period, adjust=False).mean().rename('EMA')

    @staticmethod
    def macd(data, period_long, period_short, period_signal):
        """
        Calculates the Moving Average Convergence Divergence (MACD) for the given data and periods.
        """
        shortEMA = Indicators.ema(data['Close'], period_short)
        longEMA = Indicators.ema(data['Close'], period_long)
        macd = shortEMA - longEMA
        signal = macd.ewm(span=period_signal, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal Line': signal})

    @staticmethod
    def rsi(data, period):
        """
        Calculates the Relative Strength Index (RSI) for the given data and period.
        """
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        average_gain = gain.rolling(window=period).mean()
        average_loss = loss.rolling(window=period).mean()
        rs = average_gain / average_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename('RSI')

    @staticmethod
    def bollinger_bands(data, period):
        """
        Calculates the Bollinger Bands for the given data and period.
        """
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return pd.DataFrame({'Upper Band': upper_band, 'Lower Band': lower_band})

    @staticmethod
    def atr(data, period):
        """
        Calculates the Average True Range (ATR) for the given data and period.
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.rename('ATR')

    @staticmethod
    def adx(data, period):
        """
        Calculates the Average Directional Index (ADX) for the given data and period.
        """
        high = data['High']
        low = data['Low']
        close = data['Close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()

        return adx.rename('ADX')
