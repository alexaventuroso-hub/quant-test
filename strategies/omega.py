"""
OMEGA & PRO STRATEGIES - Anton Kreil Style
"""
import numpy as np
import pandas as pd
from typing import Tuple

class OmegaStrategy:
    """Best performer - trend following with statistical edge"""
    name = "Omega"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calc_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # LONG: oversold + trend up + volume
        long = (df['zscore'] < -1.0) & (df['rsi'] < 40) & (df['trend'] >= 1) & (df['vol_ratio'] > 0.8)
        # SHORT: overbought + trend down + volume  
        short = (df['zscore'] > 1.0) & (df['rsi'] > 60) & (df['trend'] <= -1) & (df['vol_ratio'] > 0.8)
        
        signals[long] = 1
        signals[short] = -1
        signals[df['adx'] < 20] = 0  # No trade in chop
        return signals
    
    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema10'] = df['close'].ewm(span=10).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['trend'] = ((df['close'] > df['ema10']).astype(int) + 
                       (df['ema10'] > df['ema20']).astype(int) + 
                       (df['ema20'] > df['ema50']).astype(int) - 1.5)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        df['zscore'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        tr = pd.concat([df['high'] - df['low'], 
                        (df['high'] - df['close'].shift()).abs(),
                        (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        plus_dm = df['high'].diff().where(lambda x: x > 0, 0)
        minus_dm = (-df['low'].diff()).where(lambda x: x > 0, 0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr14
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14
        df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 0.001)).rolling(14).mean()
        
        return df


class TrendOnlyStrategy:
    """Never fights the trend"""
    name = "TrendOnly"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        signals = pd.Series(0, index=df.index)
        
        uptrend = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
        downtrend = (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])
        
        # Pullback entries
        pullback_long = uptrend & (df['close'] < df['ema20']) & (df['close'] > df['ema50'])
        pullback_short = downtrend & (df['close'] > df['ema20']) & (df['close'] < df['ema50'])
        
        signals[pullback_long] = 1
        signals[pullback_short] = -1
        return signals


class ConservativeStrategy:
    """Low frequency, high conviction"""
    name = "Conservative"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        signals = pd.Series(0, index=df.index)
        
        # Very strict conditions
        signals[(df['close'] > df['ema200']) & (df['rsi'] < 30)] = 1
        signals[(df['close'] < df['ema200']) & (df['rsi'] > 70)] = -1
        return signals


class ProStrategy:
    """Professional multi-factor"""
    name = "Pro"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # EMAs
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema55'] = df['close'].ewm(span=55).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        signals = pd.Series(0, index=df.index)
        
        long = (df['ema9'] > df['ema21']) & (df['macd_hist'] > 0) & (df['rsi'] < 60)
        short = (df['ema9'] < df['ema21']) & (df['macd_hist'] < 0) & (df['rsi'] > 40)
        
        signals[long] = 1
        signals[short] = -1
        return signals


class SniperStrategy:
    """Precision entries at extremes"""
    name = "Sniper"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_lower'] = df['bb_mid'] - 2.5 * df['bb_std']
        df['bb_upper'] = df['bb_mid'] + 2.5 * df['bb_std']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        signals = pd.Series(0, index=df.index)
        
        # Only at 2.5 std extremes
        signals[(df['close'] < df['bb_lower']) & (df['rsi'] < 25)] = 1
        signals[(df['close'] > df['bb_upper']) & (df['rsi'] > 75)] = -1
        return signals


class AlphaStrategy:
    """Momentum + Mean reversion combo"""
    name = "Alpha"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        df['mom_5'] = df['close'] / df['close'].shift(5) - 1
        df['mom_20'] = df['close'] / df['close'].shift(20) - 1
        df['zscore'] = (df['close'] - df['close'].rolling(30).mean()) / df['close'].rolling(30).std()
        
        signals = pd.Series(0, index=df.index)
        
        # Momentum + mean reversion
        signals[(df['mom_5'] > 0) & (df['mom_20'] > 0) & (df['zscore'] < 1)] = 1
        signals[(df['mom_5'] < 0) & (df['mom_20'] < 0) & (df['zscore'] > -1)] = -1
        return signals


# Registry
STRATEGIES = {
    'omega': OmegaStrategy,
    'trendonly': TrendOnlyStrategy,
    'conservative': ConservativeStrategy,
    'pro': ProStrategy,
    'sniper': SniperStrategy,
    'alpha': AlphaStrategy,
}

def get_strategy(name: str):
    if name.lower() in STRATEGIES:
        return STRATEGIES[name.lower()]()
    raise ValueError(f"Unknown strategy: {name}")

# Base class for compatibility
class BaseStrategy:
    name = "Base"
    def generate_signals(self, df):
        return pd.Series(0, index=df.index)

class Signal:
    BUY = 1
    SELL = -1
    HOLD = 0

from .ai_pro import AIProStrategy
STRATEGIES['ai_pro'] = AIProStrategy
