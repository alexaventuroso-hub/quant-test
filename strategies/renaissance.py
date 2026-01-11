"""
RENAISSANCE-STYLE QUANT STRATEGY
================================
Combines mean reversion + momentum with statistical edge
Inspired by Medallion Fund methodology
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class RenaissanceStrategy:
    """
    Multi-factor strategy combining:
    1. Mean Reversion (Z-score) - short-term
    2. Momentum - medium-term
    3. Volatility regime filter
    4. Kelly criterion sizing
    """
    name = "Renaissance"
    
    def __init__(self):
        # Tunable parameters
        self.zscore_entry = 1.5      # Enter at 1.5 std deviation
        self.zscore_exit = 0.5       # Exit at 0.5 std deviation
        self.rsi_oversold = 35       # Slightly relaxed for more trades
        self.rsi_overbought = 65
        self.adx_min = 20            # Minimum trend strength
        self.vol_filter = True       # Use volatility regime
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calc_all_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # === REGIME DETECTION ===
        is_trending = df['adx'] > self.adx_min
        is_ranging = df['adx'] <= self.adx_min
        vol_ok = df['vol_regime'] == 'normal' if self.vol_filter else True
        
        # === MEAN REVERSION SIGNALS (for ranging markets) ===
        mr_long = (
            (df['zscore'] < -self.zscore_entry) &  # Price below mean
            (df['rsi'] < self.rsi_oversold) &       # RSI oversold
            (df['bb_pct'] < 0.1) &                  # Near lower Bollinger
            is_ranging &
            vol_ok
        )
        
        mr_short = (
            (df['zscore'] > self.zscore_entry) &   # Price above mean
            (df['rsi'] > self.rsi_overbought) &    # RSI overbought
            (df['bb_pct'] > 0.9) &                 # Near upper Bollinger
            is_ranging &
            vol_ok
        )
        
        # === MOMENTUM SIGNALS (for trending markets) ===
        mom_long = (
            (df['ema9'] > df['ema21']) &           # Fast > Slow EMA
            (df['ema21'] > df['ema50']) &          # Aligned EMAs
            (df['macd_hist'] > 0) &                # MACD bullish
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD increasing
            (df['rsi'] > 50) & (df['rsi'] < 70) &  # RSI in momentum zone
            is_trending &
            vol_ok
        )
        
        mom_short = (
            (df['ema9'] < df['ema21']) &           # Fast < Slow EMA
            (df['ema21'] < df['ema50']) &          # Aligned EMAs
            (df['macd_hist'] < 0) &                # MACD bearish
            (df['macd_hist'] < df['macd_hist'].shift(1)) &  # MACD decreasing
            (df['rsi'] < 50) & (df['rsi'] > 30) &  # RSI in momentum zone
            is_trending &
            vol_ok
        )
        
        # === COMBINED SIGNALS ===
        signals[mr_long | mom_long] = 1
        signals[mr_short | mom_short] = -1
        
        return signals
    
    def _calc_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()
        
        # Z-Score (mean reversion)
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX
        tr = self._calc_tr(df)
        df['atr'] = tr.rolling(14).mean()
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        # Volatility regime
        df['vol_ma'] = df['atr'].rolling(50).mean()
        df['vol_regime'] = 'normal'
        df.loc[df['atr'] > df['vol_ma'] * 1.5, 'vol_regime'] = 'high'
        df.loc[df['atr'] < df['vol_ma'] * 0.5, 'vol_regime'] = 'low'
        
        # Momentum
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        
        return df
    
    def _calc_tr(self, df: pd.DataFrame) -> pd.Series:
        return pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
    
    def get_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion for optimal position sizing"""
        if avg_loss == 0:
            return 0.02  # Default 2%
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use fractional Kelly (25%) for safety
        return max(0.01, min(0.10, kelly * 0.25))


class AggressiveRenaissanceStrategy(RenaissanceStrategy):
    """More aggressive version with looser parameters"""
    name = "Renaissance-Aggressive"
    
    def __init__(self):
        super().__init__()
        self.zscore_entry = 1.2      # Enter earlier
        self.zscore_exit = 0.3
        self.rsi_oversold = 40       # More relaxed
        self.rsi_overbought = 60
        self.adx_min = 15            # Trade in weaker trends too


class ConservativeRenaissanceStrategy(RenaissanceStrategy):
    """Conservative version with tighter parameters"""
    name = "Renaissance-Conservative"
    
    def __init__(self):
        super().__init__()
        self.zscore_entry = 2.0      # Only extreme deviations
        self.zscore_exit = 0.8
        self.rsi_oversold = 25       # Very oversold
        self.rsi_overbought = 75     # Very overbought
        self.adx_min = 25            # Strong trends only


# Add to registry
RENAISSANCE_STRATEGIES = {
    'renaissance': RenaissanceStrategy,
    'renaissance_agg': AggressiveRenaissanceStrategy,
    'renaissance_cons': ConservativeRenaissanceStrategy,
}
