"""
Strategy Engine - Quantitative Trading Strategies
Implements various algorithmic trading strategies
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum


class Signal(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """Trade signal with metadata"""
    signal: Signal
    symbol: str
    price: float
    timestamp: pd.Timestamp
    confidence: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the given data"""
        pass
    
    @abstractmethod
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Get current trading signal"""
        pass


class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Classic trend-following strategy:
    - BUY when fast MA crosses above slow MA
    - SELL when fast MA crosses below slow MA
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        super().__init__(f"MA_Crossover_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Calculate MAs
        df["fast_ma"] = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        df["slow_ma"] = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Crossover detection
        df["cross"] = df["fast_ma"] - df["slow_ma"]
        df["cross_prev"] = df["cross"].shift(1)
        
        # Buy when crossing up
        signals[df["cross"] > 0] = 1
        signals[(df["cross"] > 0) & (df["cross_prev"] <= 0)] = 2  # Strong buy on crossover
        
        # Sell when crossing down
        signals[df["cross"] < 0] = -1
        signals[(df["cross"] < 0) & (df["cross_prev"] >= 0)] = -2  # Strong sell on crossover
        
        return signals.clip(-1, 1)
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current_signal = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        if current_signal > 0:
            signal_type = Signal.BUY
        elif current_signal < 0:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=abs(current_signal),
            metadata={"strategy": self.name}
        )


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy
    
    Contrarian strategy based on oversold/overbought conditions:
    - BUY when RSI < oversold threshold (default 30)
    - SELL when RSI > overbought threshold (default 70)
    """
    
    def __init__(
        self, 
        rsi_period: int = 14, 
        oversold: float = 30, 
        overbought: float = 70
    ):
        super().__init__(f"RSI_MeanReversion_{rsi_period}")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df["rsi"] = self._calculate_rsi(df["close"])
        
        signals = pd.Series(0, index=df.index)
        
        # Buy when oversold
        signals[df["rsi"] < self.oversold] = 1
        
        # Sell when overbought
        signals[df["rsi"] > self.overbought] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current_rsi = self._calculate_rsi(df["close"]).iloc[-1]
        current_signal = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        confidence = 0.5
        
        if current_signal > 0:
            signal_type = Signal.BUY
            confidence = (self.oversold - current_rsi) / self.oversold
        elif current_signal < 0:
            signal_type = Signal.SELL
            confidence = (current_rsi - self.overbought) / (100 - self.overbought)
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=max(0, min(1, confidence)),
            metadata={"strategy": self.name, "rsi": current_rsi}
        )


class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Band Mean Reversion Strategy
    
    - BUY when price touches/crosses below lower band
    - SELL when price touches/crosses above upper band
    - Additional confirmation with squeeze detection
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(f"BollingerBand_{period}_{num_std}")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=self.period).mean()
        bb_std = df["close"].rolling(window=self.period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * self.num_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std * self.num_std)
        
        # Position within bands (0 to 1)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        signals = pd.Series(0, index=df.index)
        
        # Buy when price is below lower band
        signals[df["close"] < df["bb_lower"]] = 1
        
        # Sell when price is above upper band
        signals[df["close"] > df["bb_upper"]] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current_signal = signals.iloc[-1]
        
        df = df.copy()
        df["bb_middle"] = df["close"].rolling(window=self.period).mean()
        bb_std = df["close"].rolling(window=self.period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * self.num_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std * self.num_std)
        
        signal_type = Signal.HOLD
        if current_signal > 0:
            signal_type = Signal.BUY
        elif current_signal < 0:
            signal_type = Signal.SELL
        
        current_price = df["close"].iloc[-1]
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=current_price,
            timestamp=df.index[-1],
            stop_loss=df["bb_lower"].iloc[-1] * 0.99 if signal_type == Signal.BUY else None,
            take_profit=df["bb_middle"].iloc[-1] if signal_type == Signal.BUY else None,
            metadata={
                "strategy": self.name,
                "bb_upper": df["bb_upper"].iloc[-1],
                "bb_lower": df["bb_lower"].iloc[-1],
                "bb_middle": df["bb_middle"].iloc[-1]
            }
        )


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    
    Trend-following based on price momentum:
    - BUY when momentum is positive and increasing
    - SELL when momentum is negative and decreasing
    """
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        super().__init__(f"Momentum_{lookback}")
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Calculate momentum
        df["momentum"] = df["close"] / df["close"].shift(self.lookback) - 1
        df["momentum_ma"] = df["momentum"].rolling(window=5).mean()
        
        # Rate of change of momentum
        df["momentum_roc"] = df["momentum"] - df["momentum"].shift(1)
        
        signals = pd.Series(0, index=df.index)
        
        # Strong upward momentum
        signals[(df["momentum"] > self.threshold) & (df["momentum_roc"] > 0)] = 1
        
        # Strong downward momentum
        signals[(df["momentum"] < -self.threshold) & (df["momentum_roc"] < 0)] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current_signal = signals.iloc[-1]
        
        df = df.copy()
        momentum = df["close"] / df["close"].shift(self.lookback) - 1
        current_momentum = momentum.iloc[-1]
        
        signal_type = Signal.HOLD
        if current_signal > 0:
            signal_type = Signal.BUY
        elif current_signal < 0:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=min(abs(current_momentum) / self.threshold, 1.0),
            metadata={"strategy": self.name, "momentum": current_momentum}
        )


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy
    
    - BUY when MACD crosses above signal line
    - SELL when MACD crosses below signal line
    - Histogram divergence for confirmation
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Calculate MACD
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.signal_period, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        signals = pd.Series(0, index=df.index)
        
        # Detect crossovers
        df["cross"] = df["macd"] - df["macd_signal"]
        df["cross_prev"] = df["cross"].shift(1)
        
        # Bullish crossover
        signals[(df["cross"] > 0) & (df["cross_prev"] <= 0)] = 1
        
        # Bearish crossover
        signals[(df["cross"] < 0) & (df["cross_prev"] >= 0)] = -1
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current_signal = signals.iloc[-1]
        
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - macd_signal
        
        signal_type = Signal.HOLD
        if current_signal > 0:
            signal_type = Signal.BUY
        elif current_signal < 0:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            metadata={
                "strategy": self.name,
                "macd": macd.iloc[-1],
                "signal_line": macd_signal.iloc[-1],
                "histogram": histogram.iloc[-1]
            }
        )


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble Strategy - Combines multiple strategies
    
    Uses voting or weighted average of multiple strategies
    to generate more robust signals
    """
    
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None):
        names = "_".join([s.name[:10] for s in strategies])
        super().__init__(f"Ensemble_{names}")
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        all_signals = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(df) * weight
            all_signals.append(signals)
        
        # Weighted average
        combined = pd.concat(all_signals, axis=1).sum(axis=1)
        
        # Threshold for final signal
        final_signals = pd.Series(0, index=df.index)
        final_signals[combined > 0.3] = 1
        final_signals[combined < -0.3] = -1
        
        return final_signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current_signal = signals.iloc[-1]
        
        # Collect all strategy signals for metadata
        strategy_signals = {}
        for strategy in self.strategies:
            sig = strategy.get_signal(df)
            strategy_signals[strategy.name] = sig.signal.value
        
        signal_type = Signal.HOLD
        if current_signal > 0:
            signal_type = Signal.BUY
        elif current_signal < 0:
            signal_type = Signal.SELL
        
        # Calculate confidence based on agreement
        agreement = sum(1 for v in strategy_signals.values() if v == current_signal)
        confidence = agreement / len(self.strategies)
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=confidence,
            metadata={"strategy": self.name, "sub_signals": strategy_signals}
        )


# Factory function to get strategy by name
def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Factory function to create strategies"""
    strategies = {
        "ma_crossover": MACrossoverStrategy,
        "rsi": RSIMeanReversionStrategy,
        "bollinger": BollingerBandStrategy,
        "momentum": MomentumStrategy,
        "macd": MACDStrategy,
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()](**kwargs)


if __name__ == "__main__":
    # Test strategies with sample data
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    price = 40000 + np.cumsum(np.random.randn(500) * 100)
    
    df = pd.DataFrame({
        "open": price * 0.999,
        "high": price * 1.005,
        "low": price * 0.995,
        "close": price,
        "volume": np.random.randint(100, 1000, 500)
    }, index=dates)
    
    # Test each strategy
    strategies = [
        MACrossoverStrategy(),
        RSIMeanReversionStrategy(),
        BollingerBandStrategy(),
        MomentumStrategy(),
        MACDStrategy()
    ]
    
    print("Strategy Signals Test")
    print("=" * 50)
    
    for strategy in strategies:
        signal = strategy.get_signal(df)
        signals = strategy.generate_signals(df)
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        
        print(f"\n{strategy.name}:")
        print(f"  Current Signal: {signal.signal.name}")
        print(f"  Buy Signals: {buy_count}")
        print(f"  Sell Signals: {sell_count}")
    
    # Test ensemble
    ensemble = EnsembleStrategy(strategies)
    ensemble_signal = ensemble.get_signal(df)
    print(f"\n{ensemble.name}:")
    print(f"  Current Signal: {ensemble_signal.signal.name}")
    print(f"  Confidence: {ensemble_signal.confidence:.2%}")
    print(f"  Sub-signals: {ensemble_signal.metadata['sub_signals']}")
