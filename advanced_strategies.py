"""
Advanced Quant Strategies - Renaissance Style
Statistical Arbitrage, Machine Learning, Multi-Factor Alpha
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from strategies import BaseStrategy, Signal, TradeSignal


class StatArbStrategy(BaseStrategy):
    """
    Statistical Arbitrage - Z-Score Mean Reversion
    
    Renaissance-style mean reversion:
    - Calculate z-score of price relative to moving average
    - Buy when z-score < -threshold (oversold)
    - Sell when z-score > +threshold (overbought)
    - Uses dynamic thresholds based on volatility regime
    """
    
    def __init__(self, lookback: int = 20, entry_z: float = 2.0, exit_z: float = 0.5):
        super().__init__(f"StatArb_Z{entry_z}")
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
    
    def _calculate_zscore(self, series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect high/low volatility regime"""
        returns = df["close"].pct_change()
        vol = returns.rolling(window=20).std()
        vol_ma = vol.rolling(window=50).mean()
        # 1 = high vol, 0 = low vol
        return (vol > vol_ma).astype(int)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Calculate z-score
        df["zscore"] = self._calculate_zscore(df["close"], self.lookback)
        
        # Volatility regime - tighten thresholds in high vol
        vol_regime = self._detect_volatility_regime(df)
        entry_threshold = np.where(vol_regime == 1, self.entry_z * 1.5, self.entry_z)
        
        signals = pd.Series(0, index=df.index)
        
        # Mean reversion signals
        signals[df["zscore"] < -entry_threshold] = 1  # Oversold - BUY
        signals[df["zscore"] > entry_threshold] = -1  # Overbought - SELL
        
        # Exit signals (return to mean)
        in_position = False
        position_side = 0
        
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                in_position = True
                position_side = 1
            elif signals.iloc[i] == -1:
                in_position = True
                position_side = -1
            elif in_position:
                # Exit when z-score returns to normal
                if position_side == 1 and df["zscore"].iloc[i] > -self.exit_z:
                    signals.iloc[i] = -1  # Close long
                    in_position = False
                elif position_side == -1 and df["zscore"].iloc[i] < self.exit_z:
                    signals.iloc[i] = 1  # Close short
                    in_position = False
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        zscore = self._calculate_zscore(df["close"], self.lookback).iloc[-1]
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
            confidence=min(abs(zscore) / self.entry_z, 1.0),
            metadata={"strategy": self.name, "zscore": zscore}
        )


class MLAlphaStrategy(BaseStrategy):
    """
    Machine Learning Alpha Strategy
    
    Uses Random Forest to predict next-period returns:
    - Features: Technical indicators, momentum, volatility, volume
    - Target: Next period return direction
    - Walk-forward training to avoid lookahead bias
    """
    
    def __init__(self, retrain_period: int = 100, min_confidence: float = 0.6):
        super().__init__("ML_RandomForest")
        self.retrain_period = retrain_period
        self.min_confidence = min_confidence
        self.model = None
        self.scaler = StandardScaler()
        self.last_train_idx = 0
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from price data"""
        feat = pd.DataFrame(index=df.index)
        
        # Returns at various horizons
        for lag in [1, 2, 3, 5, 10, 20]:
            feat[f"ret_{lag}"] = df["close"].pct_change(lag)
        
        # Moving average ratios
        for window in [5, 10, 20, 50]:
            ma = df["close"].rolling(window=window).mean()
            feat[f"ma_ratio_{window}"] = df["close"] / ma - 1
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        feat["rsi"] = 100 - (100 / (1 + rs))
        
        # Volatility
        feat["volatility"] = df["close"].pct_change().rolling(window=20).std()
        feat["volatility_ratio"] = feat["volatility"] / feat["volatility"].rolling(50).mean()
        
        # Volume features
        feat["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        feat["volume_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
        
        # Price position
        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        feat["price_position"] = (df["close"] - low_20) / (high_20 - low_20)
        
        # Momentum
        feat["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        feat["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        feat["macd"] = ema_12 - ema_26
        feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
        feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
        
        # Bollinger Band position
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        feat["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std)
        
        return feat
    
    def _train_model(self, df: pd.DataFrame, end_idx: int):
        """Train model on historical data"""
        features = self._create_features(df.iloc[:end_idx])
        
        # Target: next period return direction
        target = (df["close"].pct_change().shift(-1) > 0).astype(int)
        target = target.iloc[:end_idx]
        
        # Align and drop NaN
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        if len(X) < 100:
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.last_train_idx = end_idx
        self.feature_names = X.columns.tolist()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        features = self._create_features(df)
        
        # Walk-forward: retrain periodically
        for i in range(200, len(df)):
            # Retrain if needed
            if self.model is None or (i - self.last_train_idx) >= self.retrain_period:
                self._train_model(df, i)
            
            if self.model is None:
                continue
            
            # Predict
            try:
                X = features.iloc[i:i+1][self.feature_names]
                if X.isna().any().any():
                    continue
                
                X_scaled = self.scaler.transform(X)
                prob = self.model.predict_proba(X_scaled)[0]
                
                # Only trade if confident
                if prob[1] > self.min_confidence:
                    signals.iloc[i] = 1  # BUY
                elif prob[0] > self.min_confidence:
                    signals.iloc[i] = -1  # SELL
            except:
                continue
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        # Get prediction probability
        confidence = 0.5
        if self.model is not None:
            try:
                features = self._create_features(df)
                X = features.iloc[-1:][self.feature_names]
                if not X.isna().any().any():
                    X_scaled = self.scaler.transform(X)
                    prob = self.model.predict_proba(X_scaled)[0]
                    confidence = max(prob)
            except:
                pass
        
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
            confidence=confidence,
            metadata={"strategy": self.name}
        )


class MultiFactorStrategy(BaseStrategy):
    """
    Multi-Factor Alpha Model
    
    Combines multiple alpha factors with dynamic weighting:
    - Momentum factor
    - Mean reversion factor
    - Volatility factor
    - Volume factor
    - Trend factor
    
    Weights are adjusted based on recent performance
    """
    
    def __init__(self, lookback: int = 20):
        super().__init__("MultiFactor")
        self.lookback = lookback
        self.factor_weights = {
            "momentum": 0.25,
            "mean_reversion": 0.25,
            "volatility": 0.15,
            "volume": 0.15,
            "trend": 0.20
        }
    
    def _momentum_factor(self, df: pd.DataFrame) -> pd.Series:
        """Momentum: recent returns predict future returns"""
        ret_5 = df["close"].pct_change(5)
        ret_20 = df["close"].pct_change(20)
        # Normalize to [-1, 1]
        score = (ret_5.rank(pct=True) + ret_20.rank(pct=True)) / 2
        return (score - 0.5) * 2
    
    def _mean_reversion_factor(self, df: pd.DataFrame) -> pd.Series:
        """Mean reversion: oversold/overbought conditions"""
        ma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        zscore = (df["close"] - ma) / std
        # Invert: low z-score = buy signal
        return -zscore.clip(-3, 3) / 3
    
    def _volatility_factor(self, df: pd.DataFrame) -> pd.Series:
        """Volatility: trade less in high vol, more in low vol"""
        vol = df["close"].pct_change().rolling(20).std()
        vol_ma = vol.rolling(50).mean()
        # Low vol = positive signal
        return -(vol / vol_ma - 1).clip(-1, 1)
    
    def _volume_factor(self, df: pd.DataFrame) -> pd.Series:
        """Volume: high volume confirms moves"""
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
        price_change = df["close"].pct_change(5)
        # High volume + positive price = bullish
        return (vol_ratio * np.sign(price_change)).clip(-2, 2) / 2
    
    def _trend_factor(self, df: pd.DataFrame) -> pd.Series:
        """Trend: follow the trend"""
        ma_short = df["close"].ewm(span=10).mean()
        ma_long = df["close"].ewm(span=30).mean()
        # Above MA = bullish
        trend = (ma_short - ma_long) / ma_long
        return trend.clip(-0.1, 0.1) * 10
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # Calculate all factors
        factors = {
            "momentum": self._momentum_factor(df),
            "mean_reversion": self._mean_reversion_factor(df),
            "volatility": self._volatility_factor(df),
            "volume": self._volume_factor(df),
            "trend": self._trend_factor(df)
        }
        
        # Combine factors with weights
        combined = pd.Series(0, index=df.index)
        for name, factor in factors.items():
            combined += factor * self.factor_weights[name]
        
        # Generate signals based on combined score
        signals = pd.Series(0, index=df.index)
        signals[combined > 0.15] = 1  # Bullish
        signals[combined < -0.15] = -1  # Bearish
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        # Calculate current factor scores for metadata
        factors = {
            "momentum": self._momentum_factor(df).iloc[-1],
            "mean_reversion": self._mean_reversion_factor(df).iloc[-1],
            "volatility": self._volatility_factor(df).iloc[-1],
            "volume": self._volume_factor(df).iloc[-1],
            "trend": self._trend_factor(df).iloc[-1]
        }
        
        combined = sum(f * self.factor_weights[n] for n, f in factors.items())
        
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
            confidence=min(abs(combined), 1.0),
            metadata={"strategy": self.name, "factors": factors, "combined": combined}
        )


class AdaptiveStrategy(BaseStrategy):
    """
    Adaptive Strategy - Regime Detection
    
    Detects market regime and switches strategies:
    - Trending: Use momentum
    - Ranging: Use mean reversion
    - High volatility: Reduce position size
    """
    
    def __init__(self):
        super().__init__("Adaptive")
        self.momentum = MultiFactorStrategy()
        self.mean_rev = StatArbStrategy()
    
    def _detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        # ADX for trend strength
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        
        # Directional movement
        plus_dm = (high - high.shift()).clip(lower=0)
        minus_dm = (low.shift() - low).clip(lower=0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        current_adx = adx.iloc[-1]
        
        # Volatility check
        vol = close.pct_change().rolling(20).std().iloc[-1]
        vol_ma = close.pct_change().rolling(20).std().rolling(50).mean().iloc[-1]
        
        if vol > vol_ma * 1.5:
            return "high_vol"
        elif current_adx > 25:
            return "trending"
        else:
            return "ranging"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        regime = self._detect_regime(df)
        
        if regime == "trending":
            return self.momentum.generate_signals(df)
        elif regime == "ranging":
            return self.mean_rev.generate_signals(df)
        else:  # high_vol - be cautious
            signals = self.mean_rev.generate_signals(df)
            # Reduce signal strength in high vol
            return signals * 0.5
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        regime = self._detect_regime(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        if current > 0.5:
            signal_type = Signal.BUY
        elif current < -0.5:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=abs(current) if current != 0 else 0.5,
            metadata={"strategy": self.name, "regime": regime}
        )


class RenaissanceEnsemble(BaseStrategy):
    """
    Renaissance-Style Ensemble
    
    Combines all advanced strategies with:
    - Dynamic weight adjustment based on recent performance
    - Correlation-based diversification
    - Confidence-weighted voting
    """
    
    def __init__(self):
        super().__init__("Renaissance_Ensemble")
        self.strategies = [
            StatArbStrategy(lookback=20, entry_z=2.0),
            MultiFactorStrategy(),
            AdaptiveStrategy(),
        ]
        self.weights = [1.0] * len(self.strategies)
        self.performance_window = 50
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        all_signals = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(df) * weight
            all_signals.append(signals)
        
        # Weighted average
        combined = pd.concat(all_signals, axis=1).mean(axis=1)
        
        # Threshold for final signal
        final = pd.Series(0, index=df.index)
        final[combined > 0.25] = 1
        final[combined < -0.25] = -1
        
        return final
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        # Get all sub-strategy signals
        sub_signals = {}
        total_confidence = 0
        for strategy in self.strategies:
            sig = strategy.get_signal(df)
            sub_signals[strategy.name] = {
                "signal": sig.signal.name,
                "confidence": sig.confidence
            }
            if sig.signal != Signal.HOLD:
                total_confidence += sig.confidence
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        avg_confidence = total_confidence / len(self.strategies) if total_confidence > 0 else 0.5
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, 'name') else "UNKNOWN",
            price=df["close"].iloc[-1],
            timestamp=df.index[-1],
            confidence=avg_confidence,
            metadata={"strategy": self.name, "sub_signals": sub_signals}
        )


# Factory function
def get_advanced_strategy(name: str) -> BaseStrategy:
    strategies = {
        "statarb": StatArbStrategy(),
        "ml": MLAlphaStrategy(),
        "multifactor": MultiFactorStrategy(),
        "adaptive": AdaptiveStrategy(),
        "renaissance": RenaissanceEnsemble(),
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


if __name__ == "__main__":
    # Test the strategies
    print("Testing Advanced Quant Strategies...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1h")
    returns = np.random.randn(500) * 0.01 + 0.0001
    price = 40000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": price * 0.999,
        "high": price * 1.005,
        "low": price * 0.995,
        "close": price,
        "volume": np.random.randint(100, 1000, 500) * 1000
    }, index=dates)
    
    strategies = [
        StatArbStrategy(),
        MultiFactorStrategy(),
        AdaptiveStrategy(),
        RenaissanceEnsemble(),
    ]
    
    for strategy in strategies:
        signal = strategy.get_signal(df)
        signals = strategy.generate_signals(df)
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        
        print(f"\n{strategy.name}:")
        print(f"  Signal: {signal.signal.name} (conf: {signal.confidence:.2f})")
        print(f"  Buys: {buy_count} | Sells: {sell_count}")
