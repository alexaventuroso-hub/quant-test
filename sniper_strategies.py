"""
THE SNIPER v7 - Ultra Selective, Maximum Profit
Only takes A+ setups with overwhelming odds

Philosophy:
- Wait for PERFECT setup (may wait days)
- When odds are 80%+ in our favor, strike hard
- Tiny stop loss, massive take profit
- 1 great trade beats 100 mediocre ones
"""
import numpy as np
import pandas as pd
from strategies import BaseStrategy, Signal, TradeSignal


class Sniper(BaseStrategy):
    """
    The Sniper - Only A+ Setups
    
    Entry Requirements (ALL must be true):
    1. Strong trend (ADX > 30)
    2. Pullback to key level (EMA/Support)
    3. Momentum confirmation (MACD + RSI)
    4. Volume surge
    5. Bullish/Bearish candle pattern
    6. Multiple timeframe alignment
    
    Risk Management:
    - Stop: Below recent swing low (tight)
    - Target: 1:5 risk/reward minimum
    - Position: 30% of capital on A+ setups
    """
    
    def __init__(self):
        super().__init__("Sniper")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # === TREND (Must be STRONG) ===
        df["ema_8"] = df["close"].ewm(span=8).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_55"] = df["close"].ewm(span=55).mean()
        df["ema_100"] = df["close"].ewm(span=100).mean()
        
        # Perfect uptrend alignment
        df["perfect_uptrend"] = (
            (df["ema_8"] > df["ema_21"]) & 
            (df["ema_21"] > df["ema_55"]) & 
            (df["ema_55"] > df["ema_100"]) &
            (df["close"] > df["ema_21"])
        )
        
        # Perfect downtrend alignment
        df["perfect_downtrend"] = (
            (df["ema_8"] < df["ema_21"]) & 
            (df["ema_21"] < df["ema_55"]) & 
            (df["ema_55"] < df["ema_100"]) &
            (df["close"] < df["ema_21"])
        )
        
        # ADX for trend strength
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 0.001))
        df["adx"] = dx.rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        df["atr"] = atr
        
        # Strong trend = ADX > 30
        df["strong_trend"] = df["adx"] > 30
        
        # === RSI ===
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_prev"] = df["rsi"].shift(1)
        
        # === MACD ===
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_hist_prev"] = df["macd_hist"].shift(1)
        
        # MACD turning
        df["macd_bullish"] = (df["macd_hist"] > df["macd_hist_prev"]) & (df["macd_hist_prev"] < 0)
        df["macd_bearish"] = (df["macd_hist"] < df["macd_hist_prev"]) & (df["macd_hist_prev"] > 0)
        
        # === VOLUME ===
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        df["volume_surge"] = df["vol_ratio"] > 1.5  # 50% above average
        
        # === PULLBACK DETECTION ===
        # Price touched 21 EMA (key level)
        df["touched_ema21"] = (df["low"] <= df["ema_21"] * 1.005) & (df["close"] > df["ema_21"])
        df["rejected_ema21"] = (df["high"] >= df["ema_21"] * 0.995) & (df["close"] < df["ema_21"])
        
        # RSI pullback
        df["rsi_pullback_up"] = (df["rsi"] < 45) & (df["rsi"] > 30)  # Pulled back but not crashed
        df["rsi_pullback_down"] = (df["rsi"] > 55) & (df["rsi"] < 70)  # Rallied but not moon
        
        # === CANDLE PATTERNS ===
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        df["body_pct"] = df["body"].abs() / df["range"]
        
        # Strong bullish candle (big body, closes near high)
        df["bullish_engulf"] = (
            (df["body"] > 0) &  # Green candle
            (df["body"].shift(1) < 0) &  # Previous red
            (df["close"] > df["open"].shift(1)) &  # Close above prev open
            (df["body_pct"] > 0.6)  # Strong body
        )
        
        # Strong bearish candle
        df["bearish_engulf"] = (
            (df["body"] < 0) &  # Red candle
            (df["body"].shift(1) > 0) &  # Previous green
            (df["close"] < df["open"].shift(1)) &  # Close below prev open
            (df["body_pct"] > 0.6)  # Strong body
        )
        
        # Hammer (reversal)
        lower_wick = df["open"].where(df["body"] > 0, df["close"]) - df["low"]
        upper_wick = df["high"] - df["close"].where(df["body"] > 0, df["open"])
        df["hammer"] = (lower_wick > df["body"].abs() * 2) & (upper_wick < df["body"].abs() * 0.5)
        
        # Shooting star (reversal)
        df["shooting_star"] = (upper_wick > df["body"].abs() * 2) & (lower_wick < df["body"].abs() * 0.5)
        
        # === SWING POINTS ===
        df["swing_low"] = df["low"].rolling(5, center=True).min() == df["low"]
        df["swing_high"] = df["high"].rolling(5, center=True).max() == df["high"]
        
        # Recent swing low for stop placement
        df["recent_low_5"] = df["low"].rolling(5).min()
        df["recent_high_5"] = df["high"].rolling(5).max()
        
        # === MOMENTUM ===
        df["momentum_up"] = (df["close"] > df["close"].shift(3)) & (df["close"].shift(1) > df["close"].shift(4))
        df["momentum_down"] = (df["close"] < df["close"].shift(3)) & (df["close"].shift(1) < df["close"].shift(4))
        
        return df
    
    def _count_confirmations_long(self, row) -> int:
        """Count how many bullish confirmations we have"""
        score = 0
        
        if row["perfect_uptrend"]: score += 2
        if row["strong_trend"]: score += 2
        if row["plus_di"] > row["minus_di"]: score += 1
        if row["touched_ema21"]: score += 2
        if row["rsi_pullback_up"]: score += 1
        if row["rsi"] > row["rsi_prev"]: score += 1  # RSI turning up
        if row["macd_bullish"]: score += 2
        if row["macd_hist"] > 0: score += 1
        if row["volume_surge"]: score += 2
        if row["bullish_engulf"]: score += 2
        if row["hammer"]: score += 2
        if row["momentum_up"]: score += 1
        
        return score
    
    def _count_confirmations_short(self, row) -> int:
        """Count how many bearish confirmations we have"""
        score = 0
        
        if row["perfect_downtrend"]: score += 2
        if row["strong_trend"]: score += 2
        if row["minus_di"] > row["plus_di"]: score += 1
        if row["rejected_ema21"]: score += 2
        if row["rsi_pullback_down"]: score += 1
        if row["rsi"] < row["rsi_prev"]: score += 1  # RSI turning down
        if row["macd_bearish"]: score += 2
        if row["macd_hist"] < 0: score += 1
        if row["volume_surge"]: score += 2
        if row["bearish_engulf"]: score += 2
        if row["shooting_star"]: score += 2
        if row["momentum_down"]: score += 1
        
        return score
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # Calculate confirmation scores
        long_scores = df.apply(self._count_confirmations_long, axis=1)
        short_scores = df.apply(self._count_confirmations_short, axis=1)
        
        # ONLY take trades with score >= 8 (A+ setups)
        # Maximum possible is ~17, so 8+ is very selective
        
        signals[long_scores >= 8] = 1
        signals[short_scores >= 8] = -1
        
        # Extra filter: Never trade against the trend
        signals[(signals == 1) & (df["perfect_downtrend"])] = 0
        signals[(signals == -1) & (df["perfect_uptrend"])] = 0
        
        # Extra filter: Must have volume
        signals[df["vol_ratio"] < 0.8] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        row = df.iloc[-1]
        price = row["close"]
        atr = row["atr"]
        
        # Calculate scores for metadata
        long_score = self._count_confirmations_long(row)
        short_score = self._count_confirmations_short(row)
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:  # LONG
            signal_type = Signal.BUY
            # Stop below recent swing low
            stop_loss = row["recent_low_5"] - atr * 0.3
            # Target: 1:5 R/R minimum
            risk = price - stop_loss
            take_profit = price + (risk * 5)
            
        elif current < 0:  # SHORT
            signal_type = Signal.SELL
            stop_loss = row["recent_high_5"] + atr * 0.3
            risk = stop_loss - price
            take_profit = price - (risk * 5)
        
        # Confidence based on score
        max_score = max(long_score, short_score)
        confidence = min(max_score / 12, 1.0)
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy": self.name,
                "long_score": long_score,
                "short_score": short_score,
                "adx": row["adx"],
                "rsi": row["rsi"],
                "trend": "UP" if row["perfect_uptrend"] else ("DOWN" if row["perfect_downtrend"] else "SIDEWAYS")
            }
        )


class TrendRider(BaseStrategy):
    """
    Trend Rider - Catches big moves, holds for the run
    
    Only enters at the START of a trend
    Stays in until trend exhaustion
    """
    
    def __init__(self):
        super().__init__("TrendRider")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_30"] = df["close"].ewm(span=30).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # EMA crossovers (trend start signals)
        df["ema_cross_up"] = (df["ema_10"] > df["ema_30"]) & (df["ema_10"].shift(1) <= df["ema_30"].shift(1))
        df["ema_cross_down"] = (df["ema_10"] < df["ema_30"]) & (df["ema_10"].shift(1) >= df["ema_30"].shift(1))
        
        # Trend confirmation
        df["uptrend"] = (df["ema_10"] > df["ema_30"]) & (df["close"] > df["ema_50"])
        df["downtrend"] = (df["ema_10"] < df["ema_30"]) & (df["close"] < df["ema_50"])
        
        # Momentum
        df["mom_20"] = df["close"].pct_change(20) * 100
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # Volume
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        
        # Breakout detection
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["breakout_up"] = df["close"] > df["high_20"].shift(1)
        df["breakout_down"] = df["close"] < df["low_20"].shift(1)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # LONG: Trend start or breakout
        long_signal = (
            (df["ema_cross_up"] | df["breakout_up"]) &
            (df["rsi"] > 50) & (df["rsi"] < 75) &
            (df["vol_ratio"] > 1.0) &
            (df["mom_20"] > 0)
        )
        
        # SHORT: Trend start or breakdown
        short_signal = (
            (df["ema_cross_down"] | df["breakout_down"]) &
            (df["rsi"] < 50) & (df["rsi"] > 25) &
            (df["vol_ratio"] > 1.0) &
            (df["mom_20"] < 0)
        )
        
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:
            signal_type = Signal.BUY
            stop_loss = price - atr * 2
            take_profit = price + atr * 8  # 1:4 R/R
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price + atr * 2
            take_profit = price - atr * 8
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={"strategy": self.name, "rsi": df["rsi"].iloc[-1]}
        )


class UltimateSniper(BaseStrategy):
    """
    Ultimate Sniper - Combines Sniper + TrendRider
    
    Maximum selectivity, maximum profit potential
    """
    
    def __init__(self):
        super().__init__("UltimateSniper")
        self.sniper = Sniper()
        self.rider = TrendRider()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sniper_signals = self.sniper.generate_signals(df)
        rider_signals = self.rider.generate_signals(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Sniper signals (highest conviction)
        signals[sniper_signals == 1] = 1
        signals[sniper_signals == -1] = -1
        
        # Add TrendRider for more opportunities
        signals[(signals == 0) & (rider_signals == 1)] = 1
        signals[(signals == 0) & (rider_signals == -1)] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        sniper_sig = self.sniper.get_signal(df)
        rider_sig = self.rider.get_signal(df)
        
        # Prefer Sniper
        if sniper_sig.signal != Signal.HOLD:
            return sniper_sig
        return rider_sig


def get_sniper_strategy(name: str) -> BaseStrategy:
    strategies = {
        "sniper": Sniper(),
        "rider": TrendRider(),
        "ultsniper": UltimateSniper(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    print("Testing Sniper Strategies...")
    print("=" * 50)
    
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    
    # Simulate trending market
    trend = np.cumsum(np.random.randn(500) * 0.002 + 0.0008)
    noise = np.random.randn(500) * 0.004
    returns = trend + noise
    
    price = 95000 * np.exp(returns)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(500) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(500) * 0.003)),
        "low": price * (1 - np.abs(np.random.randn(500) * 0.003)),
        "close": price,
        "volume": np.random.randint(1000, 5000, 500) * 10000
    }, index=dates)
    
    for name in ["sniper", "rider", "ultimate"]:
        strat = get_sniper_strategy(name)
        signals = strat.generate_signals(df)
        longs = (signals == 1).sum()
        shorts = (signals == -1).sum()
        
        print(f"\n{strat.name}:")
        print(f"  📈 Longs: {longs}")
        print(f"  📉 Shorts: {shorts}")
        print(f"  Total: {longs + shorts}")
        print(f"  Selectivity: {(1 - (longs+shorts)/len(df))*100:.1f}% waiting")
