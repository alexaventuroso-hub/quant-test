"""
ALPHA TREND STRATEGY v7
Pure Trend Following - Never Fight the Market

Key Principles:
1. ONLY trade in direction of trend (never counter-trend)
2. Wait for pullbacks in trend to enter (better prices)
3. Trailing stops to lock in profits
4. Pyramid into winners (add to winning positions)
5. Cut losers FAST, let winners RUN

Goal: Big profits, minimal drawdown
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from strategies import BaseStrategy, Signal, TradeSignal


@dataclass
class TrendState:
    """Current market trend state"""
    direction: int  # 1=UP, -1=DOWN, 0=SIDEWAYS
    strength: float  # 0-100
    duration: int  # Bars in current trend
    pullback: bool  # In pullback?


class AlphaTrend(BaseStrategy):
    """
    Alpha Trend - Pure Trend Following
    
    RULES:
    1. Identify strong trend (ADX > 25, EMAs aligned)
    2. Wait for pullback to key level (EMA, Fib, Support)
    3. Enter on reversal signal (hammer, engulfing, RSI bounce)
    4. Trail stop using ATR
    5. Add to winners on breakouts
    6. NEVER trade against the trend
    
    This strategy will:
    - Miss some moves (waits for confirmation)
    - Have high win rate (only takes A+ setups)
    - Let winners run big
    - Cut losers immediately
    """
    
    def __init__(self):
        super().__init__("AlphaTrend")
        self.atr_multiplier = 2.0  # For stops
        self.trend_strength_min = 20  # ADX threshold (lowered from 25)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index) for trend strength"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # === TREND IDENTIFICATION ===
        # EMAs for trend direction
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_100"] = df["close"].ewm(span=100).mean()
        
        # ADX for trend strength
        df["adx"], df["plus_di"], df["minus_di"] = self._calculate_adx(df)
        
        # Trend direction
        df["uptrend"] = (
            (df["ema_10"] > df["ema_20"]) & 
            (df["ema_20"] > df["ema_50"]) &
            (df["close"] > df["ema_50"])
        )
        df["downtrend"] = (
            (df["ema_10"] < df["ema_20"]) & 
            (df["ema_20"] < df["ema_50"]) &
            (df["close"] < df["ema_50"])
        )
        
        # Strong trend (ADX > 25)
        df["strong_trend"] = df["adx"] > self.trend_strength_min
        
        # === PULLBACK DETECTION ===
        # RSI for oversold/overbought in trend
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Pullback in uptrend = RSI < 40 while still in uptrend
        df["pullback_up"] = df["uptrend"] & (df["rsi"] < 45) & (df["close"] > df["ema_50"])
        # Pullback in downtrend = RSI > 60 while still in downtrend  
        df["pullback_down"] = df["downtrend"] & (df["rsi"] > 55) & (df["close"] < df["ema_50"])
        
        # === ENTRY TRIGGERS ===
        # MACD for momentum confirmation
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Momentum shift (MACD histogram turning)
        df["macd_turning_up"] = (df["macd_hist"] > df["macd_hist"].shift(1)) & (df["macd_hist"].shift(1) < 0)
        df["macd_turning_down"] = (df["macd_hist"] < df["macd_hist"].shift(1)) & (df["macd_hist"].shift(1) > 0)
        
        # Price action
        df["higher_low"] = df["low"] > df["low"].shift(1)
        df["lower_high"] = df["high"] < df["high"].shift(1)
        
        # Candle patterns
        df["body"] = df["close"] - df["open"]
        df["bullish_candle"] = df["body"] > 0
        df["bearish_candle"] = df["body"] < 0
        
        # Strong bullish candle (closes in upper 25% of range)
        df["range"] = df["high"] - df["low"]
        df["close_position"] = (df["close"] - df["low"]) / df["range"]
        df["strong_bullish"] = df["close_position"] > 0.75
        df["strong_bearish"] = df["close_position"] < 0.25
        
        # === VOLATILITY ===
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # === VOLUME ===
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        df["high_volume"] = df["vol_ratio"] > 1.2
        
        # === BREAKOUT DETECTION ===
        df["highest_20"] = df["high"].rolling(20).max()
        df["lowest_20"] = df["low"].rolling(20).min()
        df["breakout_up"] = df["close"] > df["highest_20"].shift(1)
        df["breakout_down"] = df["close"] < df["lowest_20"].shift(1)
        
        return df
    
    def _get_trend_state(self, df: pd.DataFrame) -> TrendState:
        """Analyze current trend state"""
        row = df.iloc[-1]
        
        direction = 0
        if row["uptrend"]:
            direction = 1
        elif row["downtrend"]:
            direction = -1
        
        strength = row["adx"] if not pd.isna(row["adx"]) else 0
        
        # Count bars in trend
        duration = 0
        for i in range(len(df) - 1, max(0, len(df) - 50), -1):
            if direction == 1 and df.iloc[i]["uptrend"]:
                duration += 1
            elif direction == -1 and df.iloc[i]["downtrend"]:
                duration += 1
            else:
                break
        
        pullback = row["pullback_up"] if direction == 1 else row["pullback_down"] if direction == -1 else False
        
        return TrendState(direction, strength, duration, pullback)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # =========================================
        # LONG SIGNALS (Only in UPTREND)
        # =========================================
        
        # Entry 1: Pullback buy in strong uptrend
        long_pullback = (
            df["uptrend"] &  # Must be in uptrend
            df["strong_trend"] &  # ADX > 25
            df["pullback_up"] &  # RSI pulled back
            (df["rsi"] > df["rsi"].shift(1)) &  # RSI turning up
            df["higher_low"] &  # Making higher lows
            (df["macd_hist"] > df["macd_hist"].shift(1)) &  # MACD improving
            df["bullish_candle"]  # Bullish candle
        )
        
        # Entry 2: Breakout continuation in uptrend
        long_breakout = (
            df["uptrend"] &  # In uptrend
            df["breakout_up"] &  # New 20-bar high
            df["strong_bullish"] &  # Strong close
            df["high_volume"] &  # Volume confirms
            (df["adx"] > 20)  # Some trend strength
        )
        
        # Entry 3: Trend resumption (EMA bounce)
        long_ema_bounce = (
            df["uptrend"] &
            (df["low"] <= df["ema_20"]) &  # Touched EMA
            (df["close"] > df["ema_20"]) &  # Closed above
            df["bullish_candle"] &
            (df["rsi"] < 50)  # Not overbought
        )
        
        # =========================================
        # SHORT SIGNALS (Only in DOWNTREND)
        # =========================================
        
        # Entry 1: Pullback sell in strong downtrend
        short_pullback = (
            df["downtrend"] &  # Must be in downtrend
            df["strong_trend"] &  # ADX > 25
            df["pullback_down"] &  # RSI pulled back up
            (df["rsi"] < df["rsi"].shift(1)) &  # RSI turning down
            df["lower_high"] &  # Making lower highs
            (df["macd_hist"] < df["macd_hist"].shift(1)) &  # MACD weakening
            df["bearish_candle"]  # Bearish candle
        )
        
        # Entry 2: Breakdown continuation in downtrend
        short_breakdown = (
            df["downtrend"] &  # In downtrend
            df["breakout_down"] &  # New 20-bar low
            df["strong_bearish"] &  # Strong close
            df["high_volume"] &  # Volume confirms
            (df["adx"] > 20)  # Some trend strength
        )
        
        # Entry 3: Trend resumption (EMA rejection)
        short_ema_rejection = (
            df["downtrend"] &
            (df["high"] >= df["ema_20"]) &  # Touched EMA
            (df["close"] < df["ema_20"]) &  # Closed below
            df["bearish_candle"] &
            (df["rsi"] > 50)  # Not oversold
        )
        
        # =========================================
        # COMBINE SIGNALS
        # =========================================
        
        signals[long_pullback | long_breakout | long_ema_bounce] = 1
        signals[short_pullback | short_breakdown | short_ema_rejection] = -1
        
        # =========================================
        # FILTERS - Only high probability setups
        # =========================================
        
        # NO signals in sideways market (ADX < 20)
        sideways = df["adx"] < 20
        signals[sideways] = 0
        
        # NO signals on low volume
        low_vol = df["vol_ratio"] < 0.7
        signals[low_vol] = 0
        
        # NO counter-trend trades (extra safety)
        # If uptrend but signal is -1, ignore
        signals[(df["uptrend"]) & (signals == -1)] = 0
        # If downtrend but signal is 1, ignore
        signals[(df["downtrend"]) & (signals == 1)] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        trend = self._get_trend_state(df)
        
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:  # LONG
            signal_type = Signal.BUY
            # Tight stop below recent low or 1.5 ATR
            recent_low = df["low"].tail(3).min()
            stop_loss = max(recent_low - atr * 0.5, price - atr * 1.5)
            # Let winners run - 3x ATR minimum
            take_profit = price + (atr * 4)
            
        elif current < 0:  # SHORT
            signal_type = Signal.SELL
            recent_high = df["high"].tail(3).max()
            stop_loss = min(recent_high + atr * 0.5, price + atr * 1.5)
            take_profit = price - (atr * 4)
        
        # Calculate confidence based on trend strength
        confidence = min(trend.strength / 50, 1.0) if trend.strength > 0 else 0.5
        
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
                "trend": "UP" if trend.direction == 1 else ("DOWN" if trend.direction == -1 else "SIDEWAYS"),
                "trend_strength": trend.strength,
                "trend_duration": trend.duration,
                "in_pullback": trend.pullback,
                "rsi": df["rsi"].iloc[-1],
                "adx": df["adx"].iloc[-1]
            }
        )


class MomentumSurfer(BaseStrategy):
    """
    Momentum Surfer - Ride strong momentum moves
    
    Only enters when there's strong momentum in a clear direction.
    Uses trailing stops to ride the wave.
    """
    
    def __init__(self):
        super().__init__("MomentumSurfer")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Momentum
        df["mom_5"] = df["close"].pct_change(5) * 100
        df["mom_10"] = df["close"].pct_change(10) * 100
        df["mom_20"] = df["close"].pct_change(20) * 100
        
        # Rate of change acceleration
        df["mom_accel"] = df["mom_5"] - df["mom_5"].shift(3)
        
        # EMAs
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        
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
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # LONG: Strong upward momentum
        long_signal = (
            (df["mom_5"] > 1.5) &  # Good 5-bar momentum
            (df["mom_10"] > 2) &  # Confirmed by 10-bar
            (df["ema_9"] > df["ema_21"]) &  # EMAs aligned
            (df["rsi"] > 45) & (df["rsi"] < 75)  # Not extreme
        )
        
        # Also: Momentum acceleration
        long_accel = (
            (df["mom_accel"] > 0.5) &  # Strong acceleration
            (df["mom_5"] > 0.5) &  # Positive momentum
            (df["rsi"] > 40) & (df["rsi"] < 70)
        )
        
        # SHORT: Strong downward momentum
        short_signal = (
            (df["mom_5"] < -1.5) &  # Strong down momentum
            (df["mom_10"] < -2) &  # Confirmed
            (df["ema_9"] < df["ema_21"]) &  # EMAs aligned
            (df["rsi"] < 55) & (df["rsi"] > 25)  # Not extreme
        )
        
        # Also: Momentum deceleration
        short_decel = (
            (df["mom_accel"] < -0.5) &  # Strong deceleration
            (df["mom_5"] < -0.5) &  # Negative momentum
            (df["rsi"] < 60) & (df["rsi"] > 30)
        )
        
        signals[long_signal | long_accel] = 1
        signals[short_signal | short_decel] = -1
        
        # Volume filter
        signals[df["vol_ratio"] < 0.6] = 0
        
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
            stop_loss = price - atr * 1.5
            take_profit = price + atr * 4.5  # 1:3 R/R
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price + atr * 1.5
            take_profit = price - atr * 4.5
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy": self.name,
                "momentum_5": df["mom_5"].iloc[-1],
                "rsi": df["rsi"].iloc[-1]
            }
        )


class AlphaCombo(BaseStrategy):
    """
    Alpha Combo - Best of trend + momentum
    
    Uses AlphaTrend for direction, MomentumSurfer for confirmation
    """
    
    def __init__(self):
        super().__init__("AlphaCombo")
        self.alpha = AlphaTrend()
        self.momentum = MomentumSurfer()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        alpha_signals = self.alpha.generate_signals(df)
        mom_signals = self.momentum.generate_signals(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Either strategy triggers a trade
        signals[alpha_signals == 1] = 1
        signals[alpha_signals == -1] = -1
        signals[mom_signals == 1] = 1
        signals[mom_signals == -1] = -1
        
        # Extra confidence when both agree
        # (could add larger position size here)
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        alpha_sig = self.alpha.get_signal(df)
        mom_sig = self.momentum.get_signal(df)
        
        # Prefer AlphaTrend if it has signal
        if alpha_sig.signal != Signal.HOLD:
            return alpha_sig
        return mom_sig


def get_alpha_strategy(name: str) -> BaseStrategy:
    strategies = {
        "alpha": AlphaTrend(),
        "surfer": MomentumSurfer(),
        "alphacombo": AlphaCombo(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    print("Testing Alpha Trend Strategies...")
    print("=" * 50)
    
    import numpy as np
    np.random.seed(42)
    
    # Create trending market data
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    
    # Simulate trending market (not random walk)
    trend = np.cumsum(np.random.randn(500) * 0.002 + 0.0005)  # Slight upward bias
    noise = np.random.randn(500) * 0.005
    returns = trend + noise
    
    price = 40000 * np.exp(returns)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(500) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(500) * 0.003)),
        "low": price * (1 - np.abs(np.random.randn(500) * 0.003)),
        "close": price,
        "volume": np.random.randint(1000, 5000, 500) * 10000
    }, index=dates)
    
    for name in ["alpha", "surfer", "alphacombo"]:
        strat = get_alpha_strategy(name)
        signals = strat.generate_signals(df)
        longs = (signals == 1).sum()
        shorts = (signals == -1).sum()
        
        sig = strat.get_signal(df)
        
        print(f"\n{strat.name}:")
        print(f"  📈 Longs: {longs}")
        print(f"  📉 Shorts: {shorts}")
        print(f"  Current: {sig.signal.name}")
        if "trend" in sig.metadata:
            print(f"  Trend: {sig.metadata['trend']}")
            print(f"  ADX: {sig.metadata.get('adx', 'N/A')}")
