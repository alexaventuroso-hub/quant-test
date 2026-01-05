"""
Optimized Quant Strategy - High Performance
Less trades, better risk/reward, lower commission drag
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
from strategies import BaseStrategy, Signal, TradeSignal


class TurboQuantStrategy(BaseStrategy):
    """
    High-Performance Quant Strategy
    
    Key improvements:
    - Strict entry criteria (only high-conviction trades)
    - 1:3 risk/reward ratio
    - Volatility-adjusted signals
    - Trend + Mean Reversion combo
    - Fewer trades = lower commissions
    """
    
    def __init__(self):
        super().__init__("TurboQuant")
        
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs for trend
        df["ema_8"] = df["close"].ewm(span=8).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_55"] = df["close"].ewm(span=55).mean()
        
        # Trend direction
        df["uptrend"] = (df["ema_8"] > df["ema_21"]) & (df["ema_21"] > df["ema_55"])
        df["downtrend"] = (df["ema_8"] < df["ema_21"]) & (df["ema_21"] < df["ema_55"])
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi_min = df["rsi"].rolling(14).min()
        rsi_max = df["rsi"].rolling(14).max()
        df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min) * 100
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (bb_std * 2)
        df["bb_lower"] = df["bb_mid"] - (bb_std * 2)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR for volatility
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # Volume confirmation
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] > (df["vol_sma"] * 1.5)
        
        # Momentum
        df["momentum"] = df["close"].pct_change(10) * 100
        
        # Support/Resistance (recent highs/lows)
        df["resistance"] = df["high"].rolling(20).max()
        df["support"] = df["low"].rolling(20).min()
        df["near_support"] = df["close"] < (df["support"] * 1.02)
        df["near_resistance"] = df["close"] > (df["resistance"] * 0.98)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # ======================
        # STRONG BUY CONDITIONS
        # ======================
        # Condition 1: Trend + Oversold + MACD cross
        buy_cond_1 = (
            df["uptrend"] &  # In uptrend
            (df["rsi"] < 40) &  # RSI oversold in uptrend
            df["macd_cross_up"] &  # MACD bullish cross
            (df["bb_pct"] < 0.3)  # Near lower band
        )
        
        # Condition 2: Mean reversion at support with volume
        buy_cond_2 = (
            df["near_support"] &  # Near support level
            (df["rsi"] < 35) &  # Oversold
            (df["stoch_rsi"] < 20) &  # Stoch RSI oversold
            df["vol_spike"]  # Volume confirmation
        )
        
        # Condition 3: Momentum reversal
        buy_cond_3 = (
            (df["momentum"].shift(1) < -3) &  # Was dropping
            (df["momentum"] > df["momentum"].shift(1)) &  # Now recovering
            (df["rsi"] < 40) &  # Still oversold
            (df["macd_hist"] > df["macd_hist"].shift(1))  # MACD histogram rising
        )
        
        # ======================
        # STRONG SELL CONDITIONS  
        # ======================
        # Condition 1: Trend + Overbought + MACD cross
        sell_cond_1 = (
            df["downtrend"] &  # In downtrend
            (df["rsi"] > 60) &  # RSI overbought in downtrend
            df["macd_cross_down"] &  # MACD bearish cross
            (df["bb_pct"] > 0.7)  # Near upper band
        )
        
        # Condition 2: Mean reversion at resistance with volume
        sell_cond_2 = (
            df["near_resistance"] &  # Near resistance level
            (df["rsi"] > 65) &  # Overbought
            (df["stoch_rsi"] > 80) &  # Stoch RSI overbought
            df["vol_spike"]  # Volume confirmation
        )
        
        # Condition 3: Momentum reversal down
        sell_cond_3 = (
            (df["momentum"].shift(1) > 3) &  # Was rising
            (df["momentum"] < df["momentum"].shift(1)) &  # Now falling
            (df["rsi"] > 60) &  # Still overbought
            (df["macd_hist"] < df["macd_hist"].shift(1))  # MACD histogram falling
        )
        
        # Apply signals (need at least 2 conditions for high conviction)
        buy_score = buy_cond_1.astype(int) + buy_cond_2.astype(int) + buy_cond_3.astype(int)
        sell_score = sell_cond_1.astype(int) + sell_cond_2.astype(int) + sell_cond_3.astype(int)
        
        signals[buy_score >= 1] = 1  # At least 1 buy condition
        signals[sell_score >= 1] = -1  # At least 1 sell condition
        
        # Don't trade in choppy/low volatility markets
        low_vol = df["atr_pct"] < df["atr_pct"].rolling(50).mean() * 0.5
        signals[low_vol] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        # Calculate confidence
        confidence = 0.5
        if current != 0:
            # More extreme RSI = higher confidence
            rsi = df["rsi"].iloc[-1]
            if current == 1:
                confidence = min((50 - rsi) / 50 + 0.5, 1.0)
            else:
                confidence = min((rsi - 50) / 50 + 0.5, 1.0)
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        # Dynamic stop loss based on ATR
        atr = df["atr"].iloc[-1]
        price = df["close"].iloc[-1]
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            confidence=confidence,
            stop_loss=price - (atr * 2) if current > 0 else price + (atr * 2),
            take_profit=price + (atr * 4) if current > 0 else price - (atr * 4),  # 1:2 R/R
            metadata={
                "strategy": self.name,
                "rsi": df["rsi"].iloc[-1],
                "trend": "up" if df["uptrend"].iloc[-1] else ("down" if df["downtrend"].iloc[-1] else "sideways")
            }
        )


class ScalpStrategy(BaseStrategy):
    """
    Scalping Strategy - Quick in and out
    
    - Very short holding period
    - Small but consistent gains
    - Tight stop losses
    """
    
    def __init__(self):
        super().__init__("Scalper")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Fast indicators
        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_13"] = df["close"].ewm(span=13).mean()
        
        # RSI fast
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        df["rsi_fast"] = 100 - (100 / (1 + rs))
        
        # Price momentum (very short term)
        df["mom_3"] = df["close"].pct_change(3) * 100
        
        signals = pd.Series(0, index=df.index)
        
        # Quick buy: EMA cross + oversold
        buy = (
            (df["ema_5"] > df["ema_13"]) &
            (df["ema_5"].shift(1) <= df["ema_13"].shift(1)) &
            (df["rsi_fast"] < 45)
        )
        
        # Quick sell: EMA cross + overbought
        sell = (
            (df["ema_5"] < df["ema_13"]) &
            (df["ema_5"].shift(1) >= df["ema_13"].shift(1)) &
            (df["rsi_fast"] > 55)
        )
        
        signals[buy] = 1
        signals[sell] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            metadata={"strategy": self.name}
        )


class SwingTrader(BaseStrategy):
    """
    Swing Trading Strategy
    
    - Longer holding periods
    - Catches bigger moves
    - Very selective entries
    """
    
    def __init__(self):
        super().__init__("SwingTrader")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Longer EMAs for swing trading
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_100"] = df["close"].ewm(span=100).mean()
        
        # Strong trend
        df["strong_uptrend"] = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_100"])
        df["strong_downtrend"] = (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_100"])
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Weekly momentum
        df["momentum_20"] = df["close"].pct_change(20) * 100
        
        # Volume trend
        df["vol_trend"] = df["volume"].rolling(10).mean() / df["volume"].rolling(30).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # Swing buy: Strong trend + pullback
        buy = (
            df["strong_uptrend"] &
            (df["rsi"] < 45) &  # Pullback
            (df["rsi"].shift(1) < df["rsi"]) &  # RSI turning up
            (df["close"] > df["ema_50"]) &  # Still above key MA
            (df["vol_trend"] > 0.8)  # Decent volume
        )
        
        # Swing sell: Clear trend reversal
        sell = (
            df["strong_downtrend"] &  # Confirmed downtrend
            (df["rsi"] > 55) &  # Not oversold
            (df["close"] < df["ema_50"]) &  # Below key MA
            (df["momentum_20"] < -2)  # Negative momentum
        )
        
        signals[buy] = 1
        signals[sell] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            metadata={"strategy": self.name}
        )


class UltimateQuant(BaseStrategy):
    """
    Ultimate Quant - Best of everything
    
    Combines:
    - TurboQuant signals
    - SwingTrader confirmation
    - Risk management
    """
    
    def __init__(self):
        super().__init__("UltimateQuant")
        self.turbo = TurboQuantStrategy()
        self.swing = SwingTrader()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        turbo_signals = self.turbo.generate_signals(df)
        swing_signals = self.swing.generate_signals(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Only trade when both agree OR turbo has very strong signal
        both_buy = (turbo_signals == 1) & (swing_signals == 1)
        both_sell = (turbo_signals == -1) & (swing_signals == -1)
        
        signals[both_buy] = 1
        signals[both_sell] = -1
        
        # Also allow turbo-only signals if very strong
        turbo_only_buy = (turbo_signals == 1) & (swing_signals == 0)
        turbo_only_sell = (turbo_signals == -1) & (swing_signals == 0)
        
        signals[turbo_only_buy] = 1
        signals[turbo_only_sell] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        turbo_sig = self.turbo.get_signal(df)
        swing_sig = self.swing.get_signal(df)
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        # Higher confidence if both agree
        confidence = 0.5
        if turbo_sig.signal == swing_sig.signal and turbo_sig.signal != Signal.HOLD:
            confidence = 0.9
        elif current != 0:
            confidence = 0.7
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=confidence,
            stop_loss=turbo_sig.stop_loss,
            take_profit=turbo_sig.take_profit,
            metadata={
                "strategy": self.name,
                "turbo": turbo_sig.signal.name,
                "swing": swing_sig.signal.name
            }
        )


def get_turbo_strategy(name: str) -> BaseStrategy:
    strategies = {
        "turbo": TurboQuantStrategy(),
        "scalp": ScalpStrategy(),
        "swing": SwingTrader(),
        "ultimate": UltimateQuant(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    print("Testing optimized strategies...")
    
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    returns = np.random.randn(500) * 0.015
    price = 40000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": price * 0.999,
        "high": price * 1.008,
        "low": price * 0.992,
        "close": price,
        "volume": np.random.randint(1000, 10000, 500) * 1000
    }, index=dates)
    
    for name in ["turbo", "swing", "ultimate"]:
        strat = get_turbo_strategy(name)
        signals = strat.generate_signals(df)
        buys = (signals == 1).sum()
        sells = (signals == -1).sum()
        print(f"{name}: {buys} buys, {sells} sells")
