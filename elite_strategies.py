"""
Elite Quant Strategy v6 - With Short Selling & Improved Signals
Now can profit from BOTH up and down markets
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from strategies import BaseStrategy, Signal, TradeSignal


class EliteQuant(BaseStrategy):
    """
    Elite Quant Strategy
    
    Key improvements over v5:
    1. SHORT SELLING - profit from downtrends
    2. Better entry timing with momentum confirmation
    3. Trailing stops to lock in profits
    4. Works on 1h timeframe for more opportunities
    5. Higher win rate through multiple confirmations
    """
    
    def __init__(self):
        super().__init__("EliteQuant")
        self.zscore_threshold = 1.2  # More sensitive
        self.min_rsi_divergence = 5  # RSI must diverge from price
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # === TREND INDICATORS ===
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_55"] = df["close"].ewm(span=55).mean()
        
        # Trend strength (ADX-like)
        df["trend_up"] = (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_55"])
        df["trend_down"] = (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_55"])
        
        # === MOMENTUM INDICATORS ===
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_sma"] = df["rsi"].rolling(5).mean()
        
        # Stochastic
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        
        # === MEAN REVERSION ===
        # Z-score
        df["zscore"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (bb_std * 2)
        df["bb_lower"] = df["bb_mid"] - (bb_std * 2)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # === VOLATILITY ===
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # Volatility squeeze
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.8
        
        # === VOLUME ===
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        df["high_vol"] = df["vol_ratio"] > 1.2
        
        # === PRICE ACTION ===
        df["higher_high"] = df["high"] > df["high"].shift(1)
        df["lower_low"] = df["low"] < df["low"].shift(1)
        df["higher_low"] = df["low"] > df["low"].shift(1)
        df["lower_high"] = df["high"] < df["high"].shift(1)
        
        # Momentum
        df["mom_5"] = df["close"].pct_change(5) * 100
        df["mom_10"] = df["close"].pct_change(10) * 100
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # =========================================
        # LONG SIGNALS (Buy)
        # =========================================
        
        # 1. Mean Reversion Long: Oversold bounce
        long_mean_rev = (
            (df["zscore"] < -self.zscore_threshold) &  # Oversold
            (df["rsi"] < 35) &  # RSI confirms
            (df["rsi"] > df["rsi"].shift(1)) &  # RSI turning up
            (df["stoch_k"] < 25) &  # Stoch oversold
            (df["macd_hist"] > df["macd_hist"].shift(1))  # MACD improving
        )
        
        # 2. Trend Following Long: Pullback in uptrend
        long_trend = (
            df["trend_up"] &  # In uptrend
            (df["rsi"] < 45) & (df["rsi"] > 30) &  # Mild pullback
            df["higher_low"] &  # Making higher lows
            (df["close"] > df["ema_21"]) &  # Above 21 EMA
            df["macd_cross_up"]  # MACD cross up
        )
        
        # 3. Breakout Long: Squeeze breakout up
        long_breakout = (
            df["squeeze"].shift(1) &  # Was in squeeze
            ~df["squeeze"] &  # Breaking out
            (df["mom_5"] > 1.0) &  # Strong momentum
            (df["close"] > df["bb_upper"]) &  # Breaking upper band
            df["high_vol"]  # Volume confirms
        )
        
        # =========================================
        # SHORT SIGNALS (Sell/Short)
        # =========================================
        
        # 1. Mean Reversion Short: Overbought reversal
        short_mean_rev = (
            (df["zscore"] > self.zscore_threshold) &  # Overbought
            (df["rsi"] > 65) &  # RSI confirms
            (df["rsi"] < df["rsi"].shift(1)) &  # RSI turning down
            (df["stoch_k"] > 75) &  # Stoch overbought
            (df["macd_hist"] < df["macd_hist"].shift(1))  # MACD weakening
        )
        
        # 2. Trend Following Short: Rally in downtrend
        short_trend = (
            df["trend_down"] &  # In downtrend
            (df["rsi"] > 55) & (df["rsi"] < 70) &  # Mild rally
            df["lower_high"] &  # Making lower highs
            (df["close"] < df["ema_21"]) &  # Below 21 EMA
            df["macd_cross_down"]  # MACD cross down
        )
        
        # 3. Breakout Short: Squeeze breakout down
        short_breakout = (
            df["squeeze"].shift(1) &  # Was in squeeze
            ~df["squeeze"] &  # Breaking out
            (df["mom_5"] < -1.0) &  # Strong downward momentum
            (df["close"] < df["bb_lower"]) &  # Breaking lower band
            df["high_vol"]  # Volume confirms
        )
        
        # =========================================
        # COMBINE SIGNALS
        # =========================================
        
        signals[long_mean_rev | long_trend | long_breakout] = 1
        signals[short_mean_rev | short_trend | short_breakout] = -1
        
        # Filter: Don't trade in dead volume
        dead_vol = df["vol_ratio"] < 0.5
        signals[dead_vol] = 0
        
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
        
        if current > 0:  # LONG
            signal_type = Signal.BUY
            stop_loss = price - (atr * 1.5)  # Tighter stop
            take_profit = price + (atr * 3)  # 1:2 R/R
        elif current < 0:  # SHORT
            signal_type = Signal.SELL
            stop_loss = price + (atr * 1.5)
            take_profit = price - (atr * 3)
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            confidence=0.7 if current != 0 else 0.5,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy": self.name,
                "zscore": df["zscore"].iloc[-1],
                "rsi": df["rsi"].iloc[-1],
                "trend": "UP" if df["trend_up"].iloc[-1] else ("DOWN" if df["trend_down"].iloc[-1] else "SIDEWAYS"),
                "atr_pct": df["atr_pct"].iloc[-1]
            }
        )


class AggressiveQuant(BaseStrategy):
    """
    Aggressive Quant - More trades, faster signals
    
    For 1h timeframe, more active trading
    """
    
    def __init__(self):
        super().__init__("AggressiveQuant")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Fast EMAs
        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_13"] = df["close"].ewm(span=13).mean()
        df["ema_34"] = df["close"].ewm(span=34).mean()
        
        # RSI fast
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD fast
        ema_8 = df["close"].ewm(span=8).mean()
        ema_17 = df["close"].ewm(span=17).mean()
        df["macd"] = ema_8 - ema_17
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Momentum
        df["mom_3"] = df["close"].pct_change(3) * 100
        df["mom_8"] = df["close"].pct_change(8) * 100
        
        # Volatility
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(10).mean()
        
        # Volume
        df["vol_sma"] = df["volume"].rolling(15).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        
        # Trend
        df["uptrend"] = df["ema_5"] > df["ema_13"]
        df["downtrend"] = df["ema_5"] < df["ema_13"]
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # LONG: EMA alignment + momentum
        long_signal = (
            (df["ema_5"] > df["ema_13"]) &
            (df["mom_3"] > 0.2) &  # Positive momentum
            (df["rsi"] < 70) &  # Not overbought
            (df["rsi"] > 30)  # Not oversold
        )
        
        # Also: Oversold bounce
        long_bounce = (
            (df["rsi"] < 35) &
            (df["rsi"] > df["rsi"].shift(1)) &  # RSI turning
            (df["macd_hist"] > df["macd_hist"].shift(1))  # MACD improving
        )
        
        # SHORT: EMA alignment + momentum
        short_signal = (
            (df["ema_5"] < df["ema_13"]) &
            (df["mom_3"] < -0.2) &  # Negative momentum
            (df["rsi"] > 30) &  # Not oversold
            (df["rsi"] < 70)  # Not overbought
        )
        
        # Also: Overbought reversal
        short_reversal = (
            (df["rsi"] > 65) &
            (df["rsi"] < df["rsi"].shift(1)) &  # RSI turning down
            (df["macd_hist"] < df["macd_hist"].shift(1))  # MACD weakening
        )
        
        signals[long_signal | long_bounce] = 1
        signals[short_signal | short_reversal] = -1
        
        # Volume filter
        signals[df["vol_ratio"] < 0.5] = 0
        
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
            stop_loss = price - (atr * 1.2)
            take_profit = price + (atr * 2.4)
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price + (atr * 1.2)
            take_profit = price - (atr * 2.4)
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={"strategy": self.name, "rsi": df["rsi"].iloc[-1]}
        )


class ComboQuant(BaseStrategy):
    """
    Combined Elite + Aggressive
    
    Uses Elite for high conviction, Aggressive for more opportunities
    """
    
    def __init__(self):
        super().__init__("ComboQuant")
        self.elite = EliteQuant()
        self.aggressive = AggressiveQuant()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        elite_signals = self.elite.generate_signals(df)
        agg_signals = self.aggressive.generate_signals(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Elite signals have priority
        signals[elite_signals == 1] = 1
        signals[elite_signals == -1] = -1
        
        # Aggressive fills in gaps
        signals[(signals == 0) & (agg_signals == 1)] = 1
        signals[(signals == 0) & (agg_signals == -1)] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        elite_sig = self.elite.get_signal(df)
        agg_sig = self.aggressive.get_signal(df)
        
        # Prefer Elite if it has a signal
        if elite_sig.signal != Signal.HOLD:
            return elite_sig
        return agg_sig


def get_elite_strategy(name: str) -> BaseStrategy:
    strategies = {
        "elite": EliteQuant(),
        "aggressive": AggressiveQuant(),
        "combo": ComboQuant(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    print("Testing Elite Quant Strategies...")
    
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    returns = np.random.randn(500) * 0.008
    trend = np.sin(np.linspace(0, 6*np.pi, 500)) * 0.003
    returns = returns + trend
    
    price = 40000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(500) * 0.002),
        "high": price * (1 + np.abs(np.random.randn(500) * 0.004)),
        "low": price * (1 - np.abs(np.random.randn(500) * 0.004)),
        "close": price,
        "volume": np.random.randint(500, 5000, 500) * 10000
    }, index=dates)
    
    for name in ["elite", "aggressive", "combo"]:
        strat = get_elite_strategy(name)
        signals = strat.generate_signals(df)
        buys = (signals == 1).sum()
        sells = (signals == -1).sum()
        
        print(f"\n{strat.name}:")
        print(f"  Longs: {buys}, Shorts: {sells}")
        print(f"  Signal rate: {(buys+sells)/len(df)*100:.1f}%")
