import pandas as pd
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def add_technical_indicators(df):
    """
    Given a dataframe with 'close', 'high', 'low', 'open', 'volume' columns,
    adds several technical indicators for advanced machine learning models.
    Assumes dataframe is already sorted by date.
    """
    df = df.copy()
    
    # We ensure that we have the necessary price columns
    if 'close' not in df.columns:
        return df

    close = df['close']
    import numpy as np

    # 0. Mathematical Foundations (Log Returns & Volatility)
    # Log returns are strictly more stationary than raw price changes
    df['log_return'] = np.log(close / close.shift(1))
    df['volatility_7'] = df['log_return'].rolling(window=7).std() * np.sqrt(252)
    df['volatility_30'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)

    # 1. Moving Averages
    df['sma_10'] = SMAIndicator(close, window=10).sma_indicator()
    df['sma_20'] = SMAIndicator(close, window=20).sma_indicator()
    df['sma_50'] = SMAIndicator(close, window=50).sma_indicator()
    df['ema_10'] = EMAIndicator(close, window=10).ema_indicator()
    
    # 2. RSI (Relative Strength Index)
    df['rsi_14'] = RSIIndicator(close, window=14).rsi()
    
    # 3. MACD
    macd = MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 4. Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mavg'] = bb.bollinger_mavg()
    df['bb_width'] = df['bb_high'] - df['bb_low']

    # 5. Stochastic Oscillator
    if 'high' in df.columns and 'low' in df.columns:
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=close, window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

    # 6. Average True Range (ATR)
    if 'high' in df.columns and 'low' in df.columns:
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=close, window=14)
        df['atr_14'] = atr.average_true_range()

    # 7. On-Balance Volume (OBV)
    if 'volume' in df.columns:
        obv = OnBalanceVolumeIndicator(close=close, volume=df['volume'])
        df['obv'] = obv.on_balance_volume()

    # 8. Lag/Shift Features (Past n-days of close price and returns)
    for lag in [1, 2, 3, 5, 7]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)
        
    return df

def generate_target_variable(df, target_col='close', days_ahead=1):
    """
    Generates a target variable (future price) for prediction.
    Shifts the `target_col` back by `days_ahead` to align future prices with today's features.
    """
    df = df.copy()
    df[f'target_{days_ahead}d'] = df[target_col].shift(-days_ahead)
    return df
