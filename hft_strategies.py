"""
High-Frequency Quant Strategy - 5 Minute Timeframe
Fast execution, small profits, high win rate
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
from strategies import BaseStrategy, Signal, TradeSignal


class HFTScalper(BaseStrategy):
    """
    High-Frequency Scalping Strategy
    
    Designed for 5-minute candles:
    - Quick entries and exits
    - Small but consistent profits (0.3-0.5%)
    - Very tight stop losses (0.2%)
    - High win rate target (60%+)
    - Uses order flow and micro-structure signals
    """
    
    def __init__(self):
        super().__init__("HFT_Scalper")
        
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Ultra-fast EMAs
        df["ema_3"] = df["close"].ewm(span=3).mean()
        df["ema_8"] = df["close"].ewm(span=8).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        
        # VWAP approximation (using typical price * volume)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
        
        # Price vs VWAP
        df["above_vwap"] = df["close"] > df["vwap"]
        
        # RSI (fast)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        # Volume analysis
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        df["high_volume"] = df["vol_ratio"] > 1.3
        
        # Candle patterns
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"] * 100
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["bullish_candle"] = df["body"] > 0
        df["bearish_candle"] = df["body"] < 0
        
        # Momentum micro
        df["mom_1"] = df["close"].pct_change(1) * 100
        df["mom_3"] = df["close"].pct_change(3) * 100
        df["mom_5"] = df["close"].pct_change(5) * 100
        
        # Acceleration (momentum of momentum)
        df["acceleration"] = df["mom_3"] - df["mom_3"].shift(1)
        
        # Bollinger squeeze (low volatility = coming breakout)
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_mid"] - (df["bb_std"] * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
        df["squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.8
        
        # Support/Resistance micro levels
        df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["s1"] = 2 * df["pivot"] - df["high"].shift(1)
        
        # Price position
        df["near_support"] = (df["close"] - df["s1"]).abs() / df["close"] < 0.002
        df["near_resistance"] = (df["close"] - df["r1"]).abs() / df["close"] < 0.002
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # ============================
        # SCALP BUY CONDITIONS
        # ============================
        
        # 1. EMA Momentum Buy: Fast EMA cross with volume
        buy_ema = (
            (df["ema_3"] > df["ema_8"]) &
            (df["ema_3"].shift(1) <= df["ema_8"].shift(1)) &
            (df["ema_8"] > df["ema_21"]) &  # Trend filter
            df["high_volume"]
        )
        
        # 2. Oversold bounce with momentum shift
        buy_oversold = (
            (df["rsi"] < 35) &
            (df["rsi"] > df["rsi"].shift(1)) &  # RSI turning up
            (df["stoch_k"] < 25) &
            (df["stoch_k"] > df["stoch_d"]) &  # Stoch cross up
            (df["acceleration"] > 0)  # Momentum accelerating
        )
        
        # 3. VWAP bounce in uptrend
        buy_vwap = (
            (df["close"] > df["vwap"]) &
            (df["close"].shift(1) <= df["vwap"].shift(1)) &  # Just crossed above
            (df["ema_8"] > df["ema_21"]) &  # Uptrend
            df["bullish_candle"]
        )
        
        # 4. Squeeze breakout up
        buy_squeeze = (
            df["squeeze"].shift(1) &  # Was in squeeze
            ~df["squeeze"] &  # Breaking out
            (df["mom_3"] > 0.3) &  # Momentum up
            df["bullish_candle"] &
            df["high_volume"]
        )
        
        # 5. Support bounce
        buy_support = (
            df["near_support"] &
            df["bullish_candle"] &
            (df["lower_wick"] > df["body"].abs()) &  # Long lower wick = rejection
            (df["rsi"] < 45)
        )
        
        # ============================
        # SCALP SELL CONDITIONS
        # ============================
        
        # 1. EMA Momentum Sell
        sell_ema = (
            (df["ema_3"] < df["ema_8"]) &
            (df["ema_3"].shift(1) >= df["ema_8"].shift(1)) &
            (df["ema_8"] < df["ema_21"]) &  # Downtrend
            df["high_volume"]
        )
        
        # 2. Overbought reversal
        sell_overbought = (
            (df["rsi"] > 65) &
            (df["rsi"] < df["rsi"].shift(1)) &  # RSI turning down
            (df["stoch_k"] > 75) &
            (df["stoch_k"] < df["stoch_d"]) &  # Stoch cross down
            (df["acceleration"] < 0)
        )
        
        # 3. VWAP rejection in downtrend
        sell_vwap = (
            (df["close"] < df["vwap"]) &
            (df["close"].shift(1) >= df["vwap"].shift(1)) &
            (df["ema_8"] < df["ema_21"]) &
            df["bearish_candle"]
        )
        
        # 4. Squeeze breakout down
        sell_squeeze = (
            df["squeeze"].shift(1) &
            ~df["squeeze"] &
            (df["mom_3"] < -0.3) &
            df["bearish_candle"] &
            df["high_volume"]
        )
        
        # 5. Resistance rejection
        sell_resistance = (
            df["near_resistance"] &
            df["bearish_candle"] &
            (df["upper_wick"] > df["body"].abs()) &  # Long upper wick
            (df["rsi"] > 55)
        )
        
        # Combine signals (any condition triggers)
        buy_signals = buy_ema | buy_oversold | buy_vwap | buy_squeeze | buy_support
        sell_signals = sell_ema | sell_overbought | sell_vwap | sell_squeeze | sell_resistance
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        # Don't trade during very low volume (likely manipulation)
        low_vol = df["vol_ratio"] < 0.5
        signals[low_vol] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        price = df["close"].iloc[-1]
        
        # Tight scalp stops
        stop_pct = 0.003  # 0.3% stop
        take_pct = 0.006  # 0.6% take profit (1:2 R/R)
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:
            signal_type = Signal.BUY
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + take_pct)
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - take_pct)
        
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
                "rsi": df["rsi"].iloc[-1],
                "vol_ratio": df["vol_ratio"].iloc[-1]
            }
        )


class MicroMomentum(BaseStrategy):
    """
    Micro-Momentum Strategy
    
    Captures small momentum bursts:
    - Very short holding period
    - Rides quick momentum waves
    - Exits at first sign of reversal
    """
    
    def __init__(self):
        super().__init__("MicroMomentum")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Momentum indicators
        df["mom_2"] = df["close"].pct_change(2) * 100
        df["mom_5"] = df["close"].pct_change(5) * 100
        df["mom_10"] = df["close"].pct_change(10) * 100
        
        # Momentum consistency
        df["mom_consistent_up"] = (df["mom_2"] > 0) & (df["mom_5"] > 0)
        df["mom_consistent_down"] = (df["mom_2"] < 0) & (df["mom_5"] < 0)
        
        # Rate of change acceleration
        df["roc"] = df["close"].pct_change(1) * 100
        df["roc_accel"] = df["roc"] - df["roc"].shift(1)
        
        # Volume surge
        df["vol_sma"] = df["volume"].rolling(10).mean()
        df["vol_surge"] = df["volume"] > df["vol_sma"] * 1.5
        
        # Fast RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
        rs = gain / loss
        df["rsi_fast"] = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        
        # Buy: Momentum burst starting
        buy = (
            df["mom_consistent_up"] &
            (df["roc_accel"] > 0.05) &  # Accelerating
            (df["mom_5"] > 0.2) &  # Decent momentum
            (df["mom_5"] < 2.0) &  # Not exhausted
            (df["rsi_fast"] < 70) &  # Room to run
            df["vol_surge"]
        )
        
        # Sell: Momentum fading
        sell = (
            df["mom_consistent_down"] &
            (df["roc_accel"] < -0.05) &
            (df["mom_5"] < -0.2) &
            (df["mom_5"] > -2.0) &
            (df["rsi_fast"] > 30) &
            df["vol_surge"]
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
        
        price = df["close"].iloc[-1]
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            stop_loss=price * 0.997 if current > 0 else price * 1.003,
            take_profit=price * 1.005 if current > 0 else price * 0.995,
            metadata={"strategy": self.name}
        )


class HFTCombo(BaseStrategy):
    """
    Combined HFT Strategy
    
    Uses multiple HFT signals for higher conviction
    """
    
    def __init__(self):
        super().__init__("HFT_Combo")
        self.scalper = HFTScalper()
        self.momentum = MicroMomentum()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        scalp_signals = self.scalper.generate_signals(df)
        mom_signals = self.momentum.generate_signals(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Trade when either fires (more opportunities)
        signals[scalp_signals == 1] = 1
        signals[mom_signals == 1] = 1
        signals[scalp_signals == -1] = -1
        signals[mom_signals == -1] = -1
        
        # But stronger when both agree
        both_buy = (scalp_signals == 1) & (mom_signals == 1)
        both_sell = (scalp_signals == -1) & (mom_signals == -1)
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        price = df["close"].iloc[-1]
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            stop_loss=price * 0.997 if current > 0 else price * 1.003,
            take_profit=price * 1.006 if current > 0 else price * 0.994,
            metadata={"strategy": self.name}
        )


def get_hft_strategy(name: str) -> BaseStrategy:
    strategies = {
        "hft": HFTScalper(),
        "micro": MicroMomentum(),
        "hftcombo": HFTCombo(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    print("Testing HFT Strategies (5m timeframe simulation)...")
    
    import numpy as np
    np.random.seed(42)
    
    # Simulate 5m data (more volatile, more data points)
    dates = pd.date_range(start="2024-01-01", periods=2000, freq="5min")
    returns = np.random.randn(2000) * 0.003  # Smaller moves for 5m
    price = 40000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(2000) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(2000) * 0.002)),
        "low": price * (1 - np.abs(np.random.randn(2000) * 0.002)),
        "close": price,
        "volume": np.random.randint(100, 1000, 2000) * 10000
    }, index=dates)
    
    for name in ["hft", "micro", "hftcombo"]:
        strat = get_hft_strategy(name)
        signals = strat.generate_signals(df)
        buys = (signals == 1).sum()
        sells = (signals == -1).sum()
        print(f"{name}: {buys} buys, {sells} sells (over {len(df)} candles)")
