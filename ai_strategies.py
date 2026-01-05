"""
AI QUANT STRATEGY v9
Uses Local Ollama (CodeLlama) OR Groq Cloud for intelligent trading decisions

Supports:
1. LOCAL: Ollama + codellama:7b (FREE, runs on M1 Mac)
2. CLOUD: Groq API (FREE tier, faster)

Requirements for LOCAL:
- Install Ollama: brew install ollama
- Pull model: ollama pull codellama:7b
- Start server: ollama serve

Requirements for CLOUD (Groq):
- Sign up: console.groq.com
- Get free API key
- pip install groq
"""
import numpy as np
import pandas as pd
import requests
import json
import os
from typing import Optional, Dict, Tuple
from strategies import BaseStrategy, Signal, TradeSignal


class AIAnalyzer:
    """
    AI analyzer supporting multiple backends:
    1. Ollama (local, free)
    2. Groq (cloud, free tier)
    """
    
    def __init__(self, 
                 backend: str = "ollama",  # "ollama" or "groq"
                 model: str = None,
                 api_key: str = None):
        
        self.backend = backend
        self.enabled = True
        
        if backend == "ollama":
            self.model = model or "codellama:7b"
            self.base_url = "http://localhost:11434"
        elif backend == "groq":
            self.model = model or "llama3-8b-8192"  # Free on Groq
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1"
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _query_ollama(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Query local Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.2,
                    }
                },
                timeout=60  # Longer timeout for local model
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return None
            
        except requests.exceptions.ConnectionError:
            print("⚠️ Ollama not running. Start with: ollama serve")
            self.enabled = False
            return None
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def _query_groq(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Query Groq cloud API"""
        if not self.api_key:
            print("⚠️ No GROQ_API_KEY set. Get free key at console.groq.com")
            self.enabled = False
            return None
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Groq error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Groq error: {e}")
            return None
    
    def query(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Query the selected backend"""
        if not self.enabled:
            return None
        
        if self.backend == "ollama":
            return self._query_ollama(prompt, max_tokens)
        elif self.backend == "groq":
            return self._query_groq(prompt, max_tokens)
        return None
    
    def analyze_market(self, indicators: Dict) -> Dict:
        """
        Analyze market using AI
        
        Returns: {bias, confidence, action, reasoning}
        """
        # Simple prompt that works with CodeLlama
        prompt = f"""You are a crypto trading analyst. Analyze this data:

RSI: {indicators.get('rsi', 50):.1f}
MACD: {'positive' if indicators.get('macd_hist', 0) > 0 else 'negative'}
Trend: {'up' if indicators.get('trend', 0) > 0 else 'down' if indicators.get('trend', 0) < 0 else 'sideways'}
ADX: {indicators.get('adx', 20):.1f}
BB Position: {indicators.get('bb_pct', 0.5):.1%}

Reply with ONLY one word: BUY, SELL, or HOLD"""

        response = self.query(prompt, max_tokens=10)
        
        if response:
            response = response.strip().upper()
            if "BUY" in response:
                return {"bias": "BULLISH", "confidence": 70, "action": "BUY", "reasoning": "AI bullish"}
            elif "SELL" in response:
                return {"bias": "BEARISH", "confidence": 70, "action": "SELL", "reasoning": "AI bearish"}
        
        # Fallback to rule-based
        return self._fallback_analysis(indicators)
    
    def _fallback_analysis(self, indicators: Dict) -> Dict:
        """Rule-based fallback"""
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd_hist", 0)
        trend = indicators.get("trend", 0)
        adx = indicators.get("adx", 20)
        
        score = 0
        if rsi < 35: score += 2
        elif rsi > 65: score -= 2
        if macd > 0: score += 1
        else: score -= 1
        score += trend
        
        if adx < 20:
            score = int(score * 0.5)
        
        if score >= 2:
            return {"bias": "BULLISH", "confidence": 65, "action": "BUY", "reasoning": "Indicators bullish"}
        elif score <= -2:
            return {"bias": "BEARISH", "confidence": 65, "action": "SELL", "reasoning": "Indicators bearish"}
        return {"bias": "NEUTRAL", "confidence": 50, "action": "HOLD", "reasoning": "Mixed signals"}


class AIQuantStrategy(BaseStrategy):
    """
    AI-Enhanced Quant Strategy
    
    Combines technical analysis with AI interpretation
    """
    
    def __init__(self, use_ai: bool = True):
        super().__init__("AIQuant")
        self.ai = AIAnalyzer() if use_ai else None
        self.min_confidence = 60  # Only trade when AI is confident
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # Trend score
        df["trend"] = (
            (df["close"] > df["ema_9"]).astype(int) +
            (df["ema_9"] > df["ema_21"]).astype(int) +
            (df["ema_21"] > df["ema_50"]).astype(int) - 1.5
        )
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # ADX
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
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 0.001))
        df["adx"] = dx.rolling(14).mean()
        df["atr"] = atr
        
        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + (bb_std * 2)
        df["bb_lower"] = df["bb_mid"] - (bb_std * 2)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 0.001)
        
        # Volume
        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma"]
        
        return df
    
    def _get_ai_signal(self, df: pd.DataFrame) -> Tuple[int, int, str]:
        """Get AI-enhanced signal"""
        if self.ai is None or not self.ai.enabled:
            return 0, 50, "AI disabled"
        
        row = df.iloc[-1]
        indicators = {
            "trend": row["trend"],
            "rsi": row["rsi"],
            "macd_hist": row["macd_hist"],
            "adx": row["adx"],
            "bb_pct": row["bb_pct"],
            "vol_ratio": row["vol_ratio"]
        }
        
        analysis = self.ai.analyze_market(indicators)
        
        signal = 0
        if analysis["action"] == "BUY" and analysis["confidence"] >= self.min_confidence:
            signal = 1
        elif analysis["action"] == "SELL" and analysis["confidence"] >= self.min_confidence:
            signal = -1
        
        return signal, analysis["confidence"], analysis["reasoning"]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # For backtesting, use rule-based (AI too slow for every bar)
        # LONG conditions
        long_signal = (
            (df["trend"] >= 1) &  # Uptrend
            (df["rsi"] < 60) & (df["rsi"] > 30) &  # Not overbought
            (df["macd_hist"] > 0) &  # Positive MACD
            (df["adx"] > 20) &  # Trending
            (df["vol_ratio"] > 0.8)  # Decent volume
        )
        
        # SHORT conditions
        short_signal = (
            (df["trend"] <= -1) &  # Downtrend
            (df["rsi"] > 40) & (df["rsi"] < 70) &  # Not oversold
            (df["macd_hist"] < 0) &  # Negative MACD
            (df["adx"] > 20) &  # Trending
            (df["vol_ratio"] > 0.8)  # Decent volume
        )
        
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        # Extra filter: Don't trade in choppy market
        signals[df["adx"] < 20] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Get signal with AI enhancement for live trading"""
        df = self._calculate_indicators(df)
        
        # Get AI analysis
        ai_signal, confidence, reasoning = self._get_ai_signal(df)
        
        # Also get rule-based signal
        rule_signals = self.generate_signals(df)
        rule_signal = rule_signals.iloc[-1]
        
        # Combine: AI must agree with rules, or AI must be very confident
        final_signal = 0
        if ai_signal != 0 and rule_signal == ai_signal:
            # Both agree
            final_signal = ai_signal
        elif ai_signal != 0 and confidence >= 75:
            # AI very confident
            final_signal = ai_signal
        elif rule_signal != 0 and confidence < 40:
            # Rules say trade but AI says no - skip
            final_signal = 0
        else:
            final_signal = rule_signal
        
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if final_signal > 0:
            signal_type = Signal.BUY
            stop_loss = price - atr * 2.5
            take_profit = price + atr * 5
        elif final_signal < 0:
            signal_type = Signal.SELL
            stop_loss = price + atr * 2.5
            take_profit = price - atr * 5
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, "name") else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            confidence=confidence / 100,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy": self.name,
                "ai_confidence": confidence,
                "ai_reasoning": reasoning,
                "rule_signal": rule_signal,
                "trend": df["trend"].iloc[-1],
                "rsi": df["rsi"].iloc[-1],
                "adx": df["adx"].iloc[-1]
            }
        )


class AITrendFollower(BaseStrategy):
    """
    AI-Enhanced Trend Follower
    
    Uses AI to confirm trend direction and strength
    Only trades with strong trends
    """
    
    def __init__(self):
        super().__init__("AITrendFollower")
        self.ai = AIAnalyzer()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Multiple EMAs for trend
        for period in [10, 20, 50, 100, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        
        # Trend score (0-5)
        df["trend_score"] = (
            (df["close"] > df["ema_10"]).astype(int) +
            (df["ema_10"] > df["ema_20"]).astype(int) +
            (df["ema_20"] > df["ema_50"]).astype(int) +
            (df["ema_50"] > df["ema_100"]).astype(int) +
            (df["ema_100"] > df["ema_200"]).astype(int)
        )
        
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
        
        # Momentum
        df["momentum"] = df["close"].pct_change(10) * 100
        
        # Volume
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # STRONG UPTREND: 4-5 EMAs aligned
        strong_up = (df["trend_score"] >= 4) & (df["momentum"] > 0) & (df["rsi"] < 70)
        
        # STRONG DOWNTREND: 0-1 EMAs aligned (inverse)
        strong_down = (df["trend_score"] <= 1) & (df["momentum"] < 0) & (df["rsi"] > 30)
        
        signals[strong_up] = 1
        signals[strong_down] = -1
        
        # Volume filter
        signals[df["vol_ratio"] < 0.7] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        df = self._calculate_indicators(df)
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        trend_score = df["trend_score"].iloc[-1]
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if current > 0:
            signal_type = Signal.BUY
            stop_loss = price - atr * 3  # Wide stop for trend following
            take_profit = price + atr * 6  # 1:2 R/R
        elif current < 0:
            signal_type = Signal.SELL
            stop_loss = price + atr * 3
            take_profit = price - atr * 6
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, "name") else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy": self.name,
                "trend_score": f"{trend_score}/5",
                "rsi": df["rsi"].iloc[-1],
                "momentum": df["momentum"].iloc[-1]
            }
        )


def get_ai_strategy(name: str) -> BaseStrategy:
    strategies = {
        "aiquant": AIQuantStrategy(use_ai=True),
        "aitrend": AITrendFollower(),
        "aiquant_noai": AIQuantStrategy(use_ai=False),  # For testing without Ollama
    }
    
    if name.lower() not in strategies:
        raise ValueError(f"Unknown: {name}. Available: {list(strategies.keys())}")
    
    return strategies[name.lower()]


# Test Ollama connection
def test_ollama():
    """Test if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama is running!")
            print(f"   Available models: {[m['name'] for m in models]}")
            return True
    except:
        pass
    
    print("❌ Ollama not running. Start with: ollama serve")
    return False


if __name__ == "__main__":
    print("🤖 AI QUANT STRATEGY v9")
    print("=" * 50)
    
    # Test Ollama
    ollama_ok = test_ollama()
    
    # Test strategy
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start="2024-01-01", periods=300, freq="1h")
    trend = np.cumsum(np.random.randn(300) * 0.002 + 0.0003)
    price = 95000 * np.exp(trend)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(300) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(300) * 0.003)),
        "low": price * (1 - np.abs(np.random.randn(300) * 0.003)),
        "close": price,
        "volume": np.random.randint(1000, 5000, 300) * 10000
    }, index=dates)
    
    for name in ["aiquant_noai", "aitrend"]:
        strat = get_ai_strategy(name)
        signals = strat.generate_signals(df)
        longs = (signals == 1).sum()
        shorts = (signals == -1).sum()
        
        print(f"\n{strat.name}:")
        print(f"  📈 Longs: {longs}")
        print(f"  📉 Shorts: {shorts}")
    
    if ollama_ok:
        print("\n🧠 Testing AI analysis...")
        ai = AIAnalyzer()
        result = ai.analyze_market({
            "trend": 1.5,
            "rsi": 45,
            "macd_hist": 0.002,
            "adx": 28,
            "bb_pct": 0.4,
            "vol_ratio": 1.2
        })
        print(f"   AI says: {result['action']} ({result['confidence']}%)")
        print(f"   Reason: {result['reasoning']}")
