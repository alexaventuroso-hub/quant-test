"""
OMEGA STRATEGY v8 - All Indicators Combined
The Ultimate Multi-Indicator Trend Following System

Uses ALL major indicators for maximum confirmation:
- Bollinger Bands (volatility + squeeze)
- RSI (momentum)
- MACD (trend momentum)  
- Stochastic (overbought/oversold)
- ADX (trend strength)
- Multiple EMAs (trend direction)
- Volume (confirmation)
- ATR (volatility-based stops)
- Ichimoku Cloud (support/resistance)
- OBV (volume trend)

Key Fix: WIDER STOPS to avoid getting shaken out
"""
import numpy as np
import pandas as pd
from strategies import BaseStrategy, Signal, TradeSignal


class OmegaStrategy(BaseStrategy):
    """
    Omega Strategy - Every Indicator Combined
    
    Entry: Wait for ALL indicators to align
    Exit: Trail stop using ATR, let winners run
    
    The key is PATIENCE - wait for perfect alignment
    """
    
    def __init__(self):
        super().__init__("Omega")
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # =============================================
        # TREND INDICATORS
        # =============================================
        
        # Multiple EMAs
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_100"] = df["close"].ewm(span=100).mean()
        df["ema_200"] = df["close"].ewm(span=200).mean()
        
        # EMA Alignment Score (0-5)
        df["ema_bull_score"] = (
            (df["close"] > df["ema_9"]).astype(int) +
            (df["ema_9"] > df["ema_21"]).astype(int) +
            (df["ema_21"] > df["ema_50"]).astype(int) +
            (df["ema_50"] > df["ema_100"]).astype(int) +
            (df["ema_100"] > df["ema_200"]).astype(int)
        )
        df["ema_bear_score"] = (
            (df["close"] < df["ema_9"]).astype(int) +
            (df["ema_9"] < df["ema_21"]).astype(int) +
            (df["ema_21"] < df["ema_50"]).astype(int) +
            (df["ema_50"] < df["ema_100"]).astype(int) +
            (df["ema_100"] < df["ema_200"]).astype(int)
        )
        
        # ADX (Trend Strength)
        high, low, close = df["high"], df["low"], df["close"]
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 0.001))
        df["adx"] = dx.rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        # =============================================
        # MOMENTUM INDICATORS
        # =============================================
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 0.001)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # =============================================
        # VOLATILITY INDICATORS
        # =============================================
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (bb_std * 2)
        df["bb_lower"] = df["bb_mid"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 0.001)
        
        # Bollinger Squeeze (low volatility = breakout coming)
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.8
        
        # ATR
        df["atr"] = atr14
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        # =============================================
        # VOLUME INDICATORS
        # =============================================
        
        # Volume SMA
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        
        # OBV (On Balance Volume)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_sma"] = df["obv"].rolling(20).mean()
        df["obv_trend"] = df["obv"] > df["obv_sma"]
        
        # =============================================
        # ICHIMOKU CLOUD (Simplified)
        # =============================================
        
        # Tenkan-sen (Conversion Line)
        high_9 = df["high"].rolling(9).max()
        low_9 = df["low"].rolling(9).min()
        df["tenkan"] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df["high"].rolling(26).max()
        low_26 = df["low"].rolling(26).min()
        df["kijun"] = (high_26 + low_26) / 2
        
        # Cloud
        df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
        high_52 = df["high"].rolling(52).max()
        low_52 = df["low"].rolling(52).min()
        df["senkou_b"] = ((high_52 + low_52) / 2).shift(26)
        
        # Price above cloud = bullish
        df["above_cloud"] = df["close"] > df[["senkou_a", "senkou_b"]].max(axis=1)
        df["below_cloud"] = df["close"] < df[["senkou_a", "senkou_b"]].min(axis=1)
        
        # =============================================
        # PRICE ACTION
        # =============================================
        
        # Candle patterns
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
        
        # Strong candles
        df["strong_bull"] = (df["body"] > 0) & (df["body"] > df["range"] * 0.6)
        df["strong_bear"] = (df["body"] < 0) & (df["body"].abs() > df["range"] * 0.6)
        
        # Swing points
        df["swing_low"] = df["low"] == df["low"].rolling(5, center=True).min()
        df["swing_high"] = df["high"] == df["high"].rolling(5, center=True).max()
        
        # Support/Resistance
        df["near_support"] = df["close"] <= df["low"].rolling(20).min() * 1.02
        df["near_resistance"] = df["close"] >= df["high"].rolling(20).max() * 0.98
        
        # =============================================
        # MOMENTUM SHIFTS
        # =============================================
        
        df["rsi_oversold"] = df["rsi"] < 35
        df["rsi_overbought"] = df["rsi"] > 65
        df["rsi_turning_up"] = (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"].shift(1) < 40)
        df["rsi_turning_down"] = (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"].shift(1) > 60)
        
        df["stoch_oversold"] = df["stoch_k"] < 25
        df["stoch_overbought"] = df["stoch_k"] > 75
        df["stoch_cross_up"] = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        df["stoch_cross_down"] = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
        
        df["macd_cross_up"] = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        df["macd_cross_down"] = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        df["macd_positive"] = df["macd_hist"] > 0
        df["macd_improving"] = df["macd_hist"] > df["macd_hist"].shift(1)
        
        return df
    
    def _calculate_bull_score(self, row) -> int:
        """Calculate bullish score (0-20)"""
        score = 0
        
        # Trend (max 5)
        score += min(row["ema_bull_score"], 5)
        
        # ADX trend strength (max 2)
        if row["adx"] > 25: score += 1
        if row["adx"] > 35: score += 1
        
        # DI (max 1)
        if row["plus_di"] > row["minus_di"]: score += 1
        
        # RSI (max 2)
        if 40 < row["rsi"] < 70: score += 1
        if row["rsi_turning_up"]: score += 1
        
        # Stochastic (max 2)
        if row["stoch_cross_up"]: score += 1
        if row["stoch_oversold"]: score += 1
        
        # MACD (max 3)
        if row["macd_positive"]: score += 1
        if row["macd_improving"]: score += 1
        if row["macd_cross_up"]: score += 1
        
        # Bollinger (max 2)
        if row["bb_pct"] < 0.3: score += 1  # Near lower band
        if row["bb_squeeze"]: score += 1  # Breakout potential
        
        # Volume (max 2)
        if row["vol_ratio"] > 1.2: score += 1
        if row["obv_trend"]: score += 1
        
        # Ichimoku (max 1)
        if row["above_cloud"]: score += 1
        
        # Price action (max 1)
        if row["strong_bull"]: score += 1
        
        return score
    
    def _calculate_bear_score(self, row) -> int:
        """Calculate bearish score (0-20)"""
        score = 0
        
        # Trend (max 5)
        score += min(row["ema_bear_score"], 5)
        
        # ADX trend strength (max 2)
        if row["adx"] > 25: score += 1
        if row["adx"] > 35: score += 1
        
        # DI (max 1)
        if row["minus_di"] > row["plus_di"]: score += 1
        
        # RSI (max 2)
        if 30 < row["rsi"] < 60: score += 1
        if row["rsi_turning_down"]: score += 1
        
        # Stochastic (max 2)
        if row["stoch_cross_down"]: score += 1
        if row["stoch_overbought"]: score += 1
        
        # MACD (max 3)
        if not row["macd_positive"]: score += 1
        if not row["macd_improving"]: score += 1
        if row["macd_cross_down"]: score += 1
        
        # Bollinger (max 2)
        if row["bb_pct"] > 0.7: score += 1  # Near upper band
        if row["bb_squeeze"]: score += 1  # Breakout potential
        
        # Volume (max 2)
        if row["vol_ratio"] > 1.2: score += 1
        if not row["obv_trend"]: score += 1
        
        # Ichimoku (max 1)
        if row["below_cloud"]: score += 1
        
        # Price action (max 1)
        if row["strong_bear"]: score += 1
        
        return score
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # --- Patched: entry-only + long-only + cooldown + regime filter ---
        df = self._calculate_all_indicators(df)
        signals = pd.Series(0, index=df.index, name="signal")

        bull_scores = df.apply(self._calculate_bull_score, axis=1)

        min_score = 12
        base_long = (bull_scores >= min_score) & (df["ema_bull_score"] >= 3)

        regime_long = (
            (df["close"] > df["ema_200"]) &
            (df["ema_200"] > df["ema_200"].shift(10)) &
            (df["adx"] >= 22) &
            (df["vol_ratio"] >= 0.9)
        )

        long_ok = base_long & regime_long
        enter = long_ok & (~long_ok.shift(1).fillna(False))

        cooldown = 20
        last_entry = -10**9
        enter_idx = []
        for i, is_enter in enumerate(enter.values):
            if bool(is_enter) and (i - last_entry) > cooldown:
                enter_idx.append(i)
                last_entry = i

        if enter_idx:
            signals.iloc[enter_idx] = 1

        return signals

    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_all_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        row = df.iloc[-1]
        price = row["close"]
        atr = row["atr"]
        
        bull_score = self._calculate_bull_score(row)
        bear_score = self._calculate_bear_score(row)
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:
            signal_type = Signal.BUY
            # WIDER stop - 2.5 ATR (won't get shaken out easily)
            stop_loss = price - atr * 2.5
            # Target 2x the stop distance
            take_profit = price + atr * 5
            
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price + atr * 2.5
            take_profit = price - atr * 5
        
        confidence = max(bull_score, bear_score) / 20
        
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
                "bull_score": bull_score,
                "bear_score": bear_score,
                "adx": row["adx"],
                "rsi": row["rsi"],
                "bb_pct": row["bb_pct"],
                "trend": "BULL" if row["ema_bull_score"] >= 4 else ("BEAR" if row["ema_bear_score"] >= 4 else "NEUTRAL")
            }
        )


class ConservativeOmega(BaseStrategy):
    """
    Conservative Omega - Even more selective
    
    Only trades when score >= 15 (75% confirmation)
    Wider stops, bigger targets
    """
    
    def __init__(self):
        super().__init__("ConservativeOmega")
        self.omega = OmegaStrategy()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self.omega._calculate_all_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        bull_scores = df.apply(self.omega._calculate_bull_score, axis=1)
        bear_scores = df.apply(self.omega._calculate_bear_score, axis=1)
        
        # Need score >= 15 for entry (very selective)
        min_score = 15
        
        long_signal = (bull_scores >= min_score) & (df["ema_bull_score"] >= 4)
        short_signal = (bear_scores >= min_score) & (df["ema_bear_score"] >= 4)
        
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        # Must have strong trend
        signals[df["adx"] < 25] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        sig = self.omega.get_signal(df)
        signals = self.generate_signals(df)
        
        if signals.iloc[-1] == 0:
            return TradeSignal(
                signal=Signal.HOLD,
                symbol=sig.symbol,
                price=sig.price,
                timestamp=sig.timestamp,
                metadata=sig.metadata
            )
        
        return sig


'''

'''



if __name__ == "__main__":
    print("Testing Omega Strategies...")
    print("=" * 50)
    
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    trend = np.cumsum(np.random.randn(500) * 0.002 + 0.0005)
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
    
    for name in ["omega", "conservative", "trendonly"]:
        strat = get_omega_strategy(name)
        signals = strat.generate_signals(df)
        longs = (signals == 1).sum()
        shorts = (signals == -1).sum()
        
        print(f"\n{strat.name}:")
        print(f"  📈 Longs: {longs}")
        print(f"  📉 Shorts: {shorts}")
        print(f"  Total: {longs + shorts}")
'''
'''



strategies = {
    "omega": OmegaStrategy(),
    "conservative": ConservativeOmega(),
#     "trendonlyomega": TrendOnlyOmega(),   # ← THIS IS THE NEW LINE
}


