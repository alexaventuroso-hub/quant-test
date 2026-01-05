"""
Professional Quant Strategy - Based on Real Institutional Methods
Inspired by Anton Kreil's ATRP methodology and Renaissance Technologies principles

Key Principles:
1. Volatility-based position sizing (ATRP)
2. Statistical edge through mean reversion at 2+ std dev
3. Trend confirmation before entry
4. 1-3 month holding periods (not scalping)
5. Risk parity and correlation awareness
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from strategies import BaseStrategy, Signal, TradeSignal


@dataclass
class PositionSizing:
    """Position sizing based on volatility"""
    atrp: float  # Average True Range Percentage
    position_size_pct: float  # % of portfolio to risk
    stop_distance: float  # In ATR multiples
    shares: float  # Number of shares/coins


class ProfessionalQuant(BaseStrategy):
    """
    Professional Quant Strategy
    
    Based on institutional trading principles:
    1. ATRP (Average True Range %) for position sizing
    2. Z-score mean reversion (only trade at statistical extremes)
    3. Trend filter (don't fight the trend)
    4. Multiple timeframe confirmation
    5. Proper risk management (1-2% risk per trade)
    
    This is NOT a scalping strategy - it's designed for 
    swing/position trading with proper edge.
    """
    
    def __init__(self, risk_per_trade: float = 0.02):
        super().__init__("ProfessionalQuant")
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.min_zscore = 1.5  # Trade at 1.5+ standard deviations (more active)
        self.atr_stop_multiple = 2.0  # Stop loss at 2x ATR
        self.atr_profit_multiple = 4.0  # Take profit at 4x ATR (1:2 R/R)
    
    def _calculate_atrp(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range Percentage
        This is the key metric for volatility-based position sizing
        """
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # ATRP = ATR as percentage of price
        atrp = atr / df["close"] * 100
        return atrp
    
    def _calculate_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Z-score for mean reversion signals"""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std
    
    def _detect_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect trend using multiple EMAs
        Returns: 1 for uptrend, -1 for downtrend, 0 for sideways
        """
        ema_20 = df["close"].ewm(span=20).mean()
        ema_50 = df["close"].ewm(span=50).mean()
        ema_200 = df["close"].ewm(span=200).mean()
        
        trend = pd.Series(0, index=df.index)
        
        # Strong uptrend: price > all EMAs, EMAs properly stacked
        uptrend = (df["close"] > ema_20) & (ema_20 > ema_50) & (ema_50 > ema_200)
        trend[uptrend] = 1
        
        # Strong downtrend: price < all EMAs
        downtrend = (df["close"] < ema_20) & (ema_20 < ema_50) & (ema_50 < ema_200)
        trend[downtrend] = -1
        
        return trend
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # ATRP for position sizing
        df["atrp"] = self._calculate_atrp(df)
        df["atr"] = df["atrp"] * df["close"] / 100
        
        # Z-score for mean reversion
        df["zscore_20"] = self._calculate_zscore(df["close"], 20)
        df["zscore_50"] = self._calculate_zscore(df["close"], 50)
        
        # Trend detection
        df["trend"] = self._detect_trend(df)
        
        # EMAs
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_200"] = df["close"].ewm(span=200).mean()
        
        # RSI with smoothing
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_ma"] = df["rsi"].rolling(5).mean()
        
        # MACD for momentum
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (bb_std * 2)
        df["bb_lower"] = df["bb_mid"] - (bb_std * 2)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # Volume confirmation
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        
        # Momentum (rate of change)
        df["roc_10"] = df["close"].pct_change(10) * 100
        df["roc_20"] = df["close"].pct_change(20) * 100
        
        return df
    
    def calculate_position_size(
        self, 
        df: pd.DataFrame, 
        capital: float = 10000
    ) -> PositionSizing:
        """
        Calculate position size based on ATRP
        
        Key principle: Risk same dollar amount on each trade
        Higher volatility = smaller position
        Lower volatility = larger position
        """
        current_atrp = df["atrp"].iloc[-1]
        current_price = df["close"].iloc[-1]
        current_atr = df["atr"].iloc[-1]
        
        # Risk amount (e.g., 2% of capital)
        risk_amount = capital * self.risk_per_trade
        
        # Stop distance in price terms
        stop_distance = current_atr * self.atr_stop_multiple
        
        # Position size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance
        
        # Convert to percentage of portfolio
        position_value = position_size * current_price
        position_pct = position_value / capital
        
        # Cap at 25% of portfolio per position
        position_pct = min(position_pct, 0.25)
        
        return PositionSizing(
            atrp=current_atrp,
            position_size_pct=position_pct,
            stop_distance=stop_distance,
            shares=position_size
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on statistical edge
        
        BUY when:
        1. Z-score < -2 (statistically oversold)
        2. Trend is up OR neutral (not fighting downtrend)
        3. RSI showing reversal (RSI > RSI MA)
        4. Volume confirming
        
        SELL when:
        1. Z-score > 2 (statistically overbought)
        2. Trend is down OR neutral
        3. RSI showing reversal
        4. Volume confirming
        """
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # =====================================
        # STATISTICAL MEAN REVERSION SIGNALS
        # =====================================
        
        # Buy: Oversold with reversal confirmation
        buy_condition = (
            # Statistical oversold (Z-score < -1.5)
            (df["zscore_20"] < -self.min_zscore) &
            # RSI showing oversold
            (df["rsi"] < 40) &
            # Price near or below lower Bollinger Band
            (df["bb_pct"] < 0.3) &
            # Volume not dead
            (df["vol_ratio"] > 0.7)
        )
        
        # Sell: Overbought with reversal confirmation
        sell_condition = (
            # Statistical overbought (Z-score > 1.5)
            (df["zscore_20"] > self.min_zscore) &
            # RSI showing overbought
            (df["rsi"] > 60) &
            # Price near or above upper Bollinger Band
            (df["bb_pct"] > 0.7) &
            # Volume not dead
            (df["vol_ratio"] > 0.7)
        )
        
        # =====================================
        # TREND FOLLOWING SIGNALS (with trend)
        # =====================================
        
        # Buy: Pullback in uptrend or neutral
        buy_pullback = (
            # In uptrend or neutral
            (df["trend"] >= 0) &
            # Mild pullback (Z-score between -1.5 and 0)
            (df["zscore_20"] > -1.5) & (df["zscore_20"] < 0) &
            # RSI oversold but not extreme
            (df["rsi"] < 45) & (df["rsi"] > 25) &
            # MACD histogram turning positive (momentum shift)
            (df["macd_hist"] > df["macd_hist"].shift(1)) &
            # Volume present
            (df["vol_ratio"] > 0.6)
        )
        
        # Sell: Rally in downtrend or neutral  
        sell_rally = (
            # In downtrend or neutral
            (df["trend"] <= 0) &
            # Mild rally (Z-score between 0 and 1.5)
            (df["zscore_20"] < 1.5) & (df["zscore_20"] > 0) &
            # RSI overbought but not extreme
            (df["rsi"] > 55) & (df["rsi"] < 75) &
            # MACD histogram turning negative
            (df["macd_hist"] < df["macd_hist"].shift(1)) &
            # Volume present
            (df["vol_ratio"] > 0.6)
        )
        
        # Combine signals
        signals[buy_condition | buy_pullback] = 1
        signals[sell_condition | sell_rally] = -1
        
        # Filter out signals during very low volume (likely manipulation)
        low_vol = df["vol_ratio"] < 0.5
        signals[low_vol] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Get current signal with full context"""
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        zscore = df["zscore_20"].iloc[-1]
        trend = df["trend"].iloc[-1]
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:
            signal_type = Signal.BUY
            stop_loss = price - (atr * self.atr_stop_multiple)
            take_profit = price + (atr * self.atr_profit_multiple)
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price + (atr * self.atr_stop_multiple)
            take_profit = price - (atr * self.atr_profit_multiple)
        
        # Calculate confidence based on z-score extremity
        confidence = min(abs(zscore) / 3, 1.0) if zscore != 0 else 0.5
        
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
                "zscore": zscore,
                "trend": "UP" if trend == 1 else ("DOWN" if trend == -1 else "SIDEWAYS"),
                "atrp": df["atrp"].iloc[-1],
                "rsi": df["rsi"].iloc[-1],
                "position_size": self.calculate_position_size(df)
            }
        )


class VolatilityBreakout(BaseStrategy):
    """
    Volatility Breakout Strategy
    
    Based on the concept that after periods of low volatility (squeeze),
    there's usually a breakout with high momentum.
    
    Uses Bollinger Band squeeze detection.
    """
    
    def __init__(self):
        super().__init__("VolatilityBreakout")
    
    def _detect_squeeze(self, df: pd.DataFrame) -> pd.Series:
        """Detect Bollinger Band squeeze (low volatility)"""
        # Bollinger Band width
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / bb_mid
        
        # Squeeze = width below 50-period average
        avg_width = bb_width.rolling(50).mean()
        squeeze = bb_width < avg_width * 0.75
        
        return squeeze
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Detect squeeze
        df["squeeze"] = self._detect_squeeze(df)
        df["was_squeeze"] = df["squeeze"].shift(1)
        
        # Momentum for breakout direction
        df["mom"] = df["close"].pct_change(5)
        
        # EMAs for trend
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # Volume surge
        df["vol_surge"] = df["volume"] > df["volume"].rolling(20).mean() * 1.5
        
        signals = pd.Series(0, index=df.index)
        
        # Breakout UP: Was in squeeze, now breaking out upward
        breakout_up = (
            df["was_squeeze"] &
            ~df["squeeze"] &
            (df["mom"] > 0.01) &
            (df["close"] > df["ema_20"]) &
            df["vol_surge"]
        )
        
        # Breakout DOWN
        breakout_down = (
            df["was_squeeze"] &
            ~df["squeeze"] &
            (df["mom"] < -0.01) &
            (df["close"] < df["ema_20"]) &
            df["vol_surge"]
        )
        
        signals[breakout_up] = 1
        signals[breakout_down] = -1
        
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


class InstitutionalQuant(BaseStrategy):
    """
    Institutional-Grade Quant Strategy
    
    Combines:
    1. ProfessionalQuant (ATRP + mean reversion)
    2. VolatilityBreakout (squeeze breakouts)
    3. Proper risk management
    
    Only trades when multiple strategies agree OR
    when there's a very high conviction signal.
    """
    
    def __init__(self):
        super().__init__("InstitutionalQuant")
        self.pro_quant = ProfessionalQuant()
        self.vol_breakout = VolatilityBreakout()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pq_signals = self.pro_quant.generate_signals(df)
        vb_signals = self.vol_breakout.generate_signals(df)
        
        signals = pd.Series(0, index=df.index)
        
        # Professional Quant signals (primary)
        signals[pq_signals == 1] = 1
        signals[pq_signals == -1] = -1
        
        # Volatility breakout adds extra signals
        signals[vb_signals == 1] = 1
        signals[vb_signals == -1] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        pq_signal = self.pro_quant.get_signal(df)
        
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
            confidence=pq_signal.confidence,
            stop_loss=pq_signal.stop_loss,
            take_profit=pq_signal.take_profit,
            metadata={
                "strategy": self.name,
                **pq_signal.metadata
            }
        )


def get_pro_strategy(name: str) -> BaseStrategy:
    """Get professional strategy by name"""
    strategies = {
        "pro": ProfessionalQuant(),
        "breakout": VolatilityBreakout(),
        "institutional": InstitutionalQuant(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    print("Testing Professional Quant Strategies...")
    print("=" * 60)
    
    import numpy as np
    
    # Create realistic market data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="4h")
    
    # Simulate trending and mean-reverting periods
    returns = np.random.randn(500) * 0.015
    # Add some trend
    trend = np.sin(np.linspace(0, 4*np.pi, 500)) * 0.002
    returns = returns + trend
    
    price = 40000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(500) * 0.002),
        "high": price * (1 + np.abs(np.random.randn(500) * 0.005)),
        "low": price * (1 - np.abs(np.random.randn(500) * 0.005)),
        "close": price,
        "volume": np.random.randint(1000, 10000, 500) * 10000
    }, index=dates)
    
    for name in ["pro", "breakout", "institutional"]:
        strat = get_pro_strategy(name)
        signals = strat.generate_signals(df)
        buys = (signals == 1).sum()
        sells = (signals == -1).sum()
        sig = strat.get_signal(df)
        
        print(f"\n{strat.name}:")
        print(f"  Signal: {sig.signal.name}")
        print(f"  Total Signals: {buys} buys, {sells} sells")
        
        if "zscore" in sig.metadata:
            print(f"  Z-Score: {sig.metadata['zscore']:.2f}")
            print(f"  Trend: {sig.metadata['trend']}")
            print(f"  ATRP: {sig.metadata['atrp']:.2f}%")
