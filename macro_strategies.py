"""
MACRO QUANT STRATEGY v10 - Anton Kreil Style
============================================

This is how REAL institutional traders work:

1. MACRO FIRST - Check the economic environment before trading
   - VIX (fear index) - High VIX = don't trade or trade small
   - DXY (dollar strength) - Strong dollar = risk off
   - Interest rates - Rising rates = risk off
   - Risk-on vs Risk-off regime

2. ATRP POSITION SIZING - Risk the same $ amount on every trade
   - Higher volatility = smaller position
   - Lower volatility = larger position

3. DISTRIBUTION OF RETURNS - Only trade when odds favor you
   - 2+ standard deviations from mean = opportunity
   - Trade mean reversion at extremes
   - Trade trend continuation in trends

4. CORRELATION - Don't fight correlated assets
   - BTC correlates with NASDAQ, SPY
   - If stocks dumping, don't long crypto

5. AI (GROK) - Analyze macro data and make decisions

This is NOT a get-rich-quick scheme. This is how professionals trade.
"""

import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from strategies import BaseStrategy, Signal, TradeSignal


class MacroDataFetcher:
    """
    Fetches macro economic data for analysis
    
    Free data sources:
    - Yahoo Finance (VIX, DXY, SPY)
    - FRED API (interest rates, GDP)
    - Fear & Greed Index
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour
    
    def get_vix(self) -> Optional[float]:
        """Get VIX (volatility index) - measures market fear"""
        try:
            # Using Yahoo Finance
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            pass
        return None
    
    def get_dxy(self) -> Optional[float]:
        """Get DXY (US Dollar Index) - strong dollar = risk off"""
        try:
            import yfinance as yf
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            pass
        return None
    
    def get_spy_trend(self) -> Optional[str]:
        """Get SPY trend - crypto correlates with stocks"""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="5d")
            if len(hist) >= 2:
                change = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                if change > 1:
                    return "BULLISH"
                elif change < -1:
                    return "BEARISH"
                return "NEUTRAL"
        except:
            pass
        return None
    
    def get_fear_greed(self) -> Optional[int]:
        """Get Crypto Fear & Greed Index (0-100)"""
        try:
            response = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return int(data['data'][0]['value'])
        except:
            pass
        return None
    
    def get_btc_dominance(self) -> Optional[float]:
        """Get BTC dominance - high = flight to safety in crypto"""
        try:
            response = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data['data']['market_cap_percentage']['btc']
        except:
            pass
        return None
    
    def get_macro_summary(self) -> Dict:
        """Get all macro data"""
        return {
            "vix": self.get_vix(),
            "dxy": self.get_dxy(),
            "spy_trend": self.get_spy_trend(),
            "fear_greed": self.get_fear_greed(),
            "btc_dominance": self.get_btc_dominance(),
            "timestamp": datetime.now().isoformat()
        }


class GrokAnalyzer:
    """
    Uses Grok AI to analyze macro conditions and make trading decisions
    
    This is your AI quant - it interprets data like a human analyst would
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            print("⚠️ No XAI_API_KEY - Grok disabled. Get free key at console.x.ai")
    
    def _query(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Query Grok API"""
        if not self.enabled:
            return None
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-2-1212",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional quant trader. Be concise. Give clear BUY/SELL/HOLD recommendations with confidence levels."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Grok error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Grok error: {e}")
            return None
    
    def analyze_macro(self, macro_data: Dict) -> Dict:
        """
        Have Grok analyze macro conditions
        
        Returns: {
            "regime": "RISK_ON" | "RISK_OFF" | "NEUTRAL",
            "trade_size": "FULL" | "HALF" | "NONE",
            "bias": "LONG" | "SHORT" | "FLAT",
            "reasoning": str
        }
        """
        prompt = f"""Analyze this macro data for crypto trading:

VIX (fear index): {macro_data.get('vix', 'N/A')} (>20 = elevated fear, >30 = high fear)
DXY (dollar): {macro_data.get('dxy', 'N/A')} (strong dollar = risk off)
SPY trend: {macro_data.get('spy_trend', 'N/A')} (crypto correlates with stocks)
Fear & Greed: {macro_data.get('fear_greed', 'N/A')} (0-25 extreme fear, 75-100 extreme greed)
BTC Dominance: {macro_data.get('btc_dominance', 'N/A')}%

Based on Anton Kreil's methodology:
1. Is this RISK_ON or RISK_OFF environment?
2. Should we trade FULL size, HALF size, or NOT trade?
3. Should we be LONG, SHORT, or FLAT bias?

Reply with ONLY this format:
REGIME: RISK_ON/RISK_OFF/NEUTRAL
SIZE: FULL/HALF/NONE
BIAS: LONG/SHORT/FLAT
REASON: one sentence"""

        response = self._query(prompt)
        
        if response:
            result = {
                "regime": "NEUTRAL",
                "trade_size": "HALF",
                "bias": "FLAT",
                "reasoning": response
            }
            
            response_upper = response.upper()
            
            if "RISK_ON" in response_upper:
                result["regime"] = "RISK_ON"
            elif "RISK_OFF" in response_upper:
                result["regime"] = "RISK_OFF"
            
            if "SIZE: FULL" in response_upper or "FULL SIZE" in response_upper:
                result["trade_size"] = "FULL"
            elif "SIZE: NONE" in response_upper or "DON'T TRADE" in response_upper or "NO TRADE" in response_upper:
                result["trade_size"] = "NONE"
            
            if "BIAS: LONG" in response_upper:
                result["bias"] = "LONG"
            elif "BIAS: SHORT" in response_upper:
                result["bias"] = "SHORT"
            
            return result
        
        # Fallback based on data
        return self._fallback_macro_analysis(macro_data)
    
    def _fallback_macro_analysis(self, macro_data: Dict) -> Dict:
        """Rule-based fallback if Grok unavailable"""
        regime = "NEUTRAL"
        trade_size = "HALF"
        bias = "FLAT"
        
        vix = macro_data.get("vix")
        fear_greed = macro_data.get("fear_greed")
        spy_trend = macro_data.get("spy_trend")
        
        # VIX analysis
        if vix:
            if vix > 30:
                regime = "RISK_OFF"
                trade_size = "NONE"
            elif vix > 25:
                regime = "RISK_OFF"
                trade_size = "HALF"
            elif vix < 15:
                regime = "RISK_ON"
                trade_size = "FULL"
        
        # Fear & Greed
        if fear_greed:
            if fear_greed < 25:  # Extreme fear = buy opportunity
                bias = "LONG"
            elif fear_greed > 75:  # Extreme greed = sell opportunity
                bias = "SHORT"
        
        # SPY correlation
        if spy_trend == "BULLISH":
            if bias == "FLAT":
                bias = "LONG"
        elif spy_trend == "BEARISH":
            if bias == "FLAT":
                bias = "SHORT"
        
        return {
            "regime": regime,
            "trade_size": trade_size,
            "bias": bias,
            "reasoning": "Fallback analysis based on VIX, Fear&Greed, SPY"
        }
    
    def analyze_trade(self, symbol: str, indicators: Dict, macro_regime: Dict) -> Dict:
        """
        Analyze a specific trade setup
        
        Returns: {
            "action": "BUY" | "SELL" | "HOLD",
            "confidence": 0-100,
            "size_pct": 0-100,
            "stop_atr_mult": float,
            "target_atr_mult": float,
            "reasoning": str
        }
        """
        prompt = f"""You are analyzing a {symbol} trade setup.

MACRO CONTEXT:
- Regime: {macro_regime.get('regime', 'NEUTRAL')}
- Suggested size: {macro_regime.get('trade_size', 'HALF')}
- Bias: {macro_regime.get('bias', 'FLAT')}

TECHNICAL DATA:
- Trend: {indicators.get('trend', 'NEUTRAL')}
- RSI: {indicators.get('rsi', 50):.1f}
- MACD: {'Bullish' if indicators.get('macd_hist', 0) > 0 else 'Bearish'}
- ADX (trend strength): {indicators.get('adx', 20):.1f}
- ATRP (volatility): {indicators.get('atrp', 2):.2f}%
- Z-score: {indicators.get('zscore', 0):.2f} (>2 overbought, <-2 oversold)

Anton Kreil rules:
1. Never fight the macro regime
2. Use ATRP for position sizing (higher vol = smaller position)
3. Only trade at statistical extremes (2+ std dev)
4. Let winners run, cut losers fast

Should we BUY, SELL, or HOLD? Reply:
ACTION: BUY/SELL/HOLD
CONFIDENCE: 0-100
STOP: ATR multiplier (e.g., 2.5)
TARGET: ATR multiplier (e.g., 5)
REASON: one sentence"""

        response = self._query(prompt)
        
        if response:
            result = {
                "action": "HOLD",
                "confidence": 50,
                "size_pct": 25,
                "stop_atr_mult": 2.5,
                "target_atr_mult": 5.0,
                "reasoning": response
            }
            
            response_upper = response.upper()
            
            if "ACTION: BUY" in response_upper:
                result["action"] = "BUY"
            elif "ACTION: SELL" in response_upper:
                result["action"] = "SELL"
            
            # Extract confidence
            import re
            conf_match = re.search(r"CONFIDENCE:\s*(\d+)", response_upper)
            if conf_match:
                result["confidence"] = int(conf_match.group(1))
            
            # Extract stop
            stop_match = re.search(r"STOP:\s*([\d.]+)", response_upper)
            if stop_match:
                result["stop_atr_mult"] = float(stop_match.group(1))
            
            # Extract target
            target_match = re.search(r"TARGET:\s*([\d.]+)", response_upper)
            if target_match:
                result["target_atr_mult"] = float(target_match.group(1))
            
            # Adjust size based on macro
            if macro_regime.get("trade_size") == "NONE":
                result["action"] = "HOLD"
                result["size_pct"] = 0
            elif macro_regime.get("trade_size") == "HALF":
                result["size_pct"] = 15
            else:
                result["size_pct"] = 25
            
            return result
        
        return self._fallback_trade_analysis(indicators, macro_regime)
    
    def _fallback_trade_analysis(self, indicators: Dict, macro_regime: Dict) -> Dict:
        """Rule-based fallback"""
        action = "HOLD"
        confidence = 50
        
        rsi = indicators.get("rsi", 50)
        zscore = indicators.get("zscore", 0)
        adx = indicators.get("adx", 20)
        trend = indicators.get("trend", 0)
        
        # Only trade if macro allows
        if macro_regime.get("trade_size") == "NONE":
            return {
                "action": "HOLD",
                "confidence": 0,
                "size_pct": 0,
                "stop_atr_mult": 2.5,
                "target_atr_mult": 5.0,
                "reasoning": "Macro regime says no trading"
            }
        
        # Check for statistical extremes
        if zscore < -2 and rsi < 30 and macro_regime.get("bias") != "SHORT":
            action = "BUY"
            confidence = 70
        elif zscore > 2 and rsi > 70 and macro_regime.get("bias") != "LONG":
            action = "SELL"
            confidence = 70
        # Trend following
        elif adx > 25 and trend > 1 and macro_regime.get("bias") == "LONG":
            action = "BUY"
            confidence = 60
        elif adx > 25 and trend < -1 and macro_regime.get("bias") == "SHORT":
            action = "SELL"
            confidence = 60
        
        size_pct = 25 if macro_regime.get("trade_size") == "FULL" else 15
        
        return {
            "action": action,
            "confidence": confidence,
            "size_pct": size_pct,
            "stop_atr_mult": 2.5,
            "target_atr_mult": 5.0,
            "reasoning": "Fallback rule-based analysis"
        }


class MacroQuantStrategy(BaseStrategy):
    """
    Anton Kreil Style Macro Quant Strategy
    
    Key principles:
    1. MACRO FIRST - Check VIX, DXY, SPY before trading
    2. ATRP SIZING - Position size based on volatility
    3. STATISTICAL EDGE - Only trade at extremes
    4. AI ANALYSIS - Use Grok to interpret data
    """
    
    def __init__(self):
        super().__init__("MacroQuant")
        self.macro_fetcher = MacroDataFetcher()
        self.grok = GrokAnalyzer()
        self.last_macro_check = None
        self.macro_cache = None
        self.macro_analysis = None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMAs for trend
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_200"] = df["close"].ewm(span=200).mean()
        
        # Trend score
        df["trend"] = (
            (df["close"] > df["ema_20"]).astype(int) +
            (df["ema_20"] > df["ema_50"]).astype(int) +
            (df["ema_50"] > df["ema_200"]).astype(int) - 1.5
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
        
        # ATR and ATRP (key for position sizing)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atrp"] = (df["atr"] / df["close"]) * 100  # ATR as percentage
        
        # Z-score (for mean reversion)
        df["zscore"] = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
        
        # ADX
        plus_dm = df["high"].diff()
        minus_dm = df["low"].diff().abs() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 0.001))
        df["adx"] = dx.rolling(14).mean()
        
        # Bollinger for squeeze detection
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + bb_std * 2
        df["bb_lower"] = df["bb_mid"] - bb_std * 2
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        
        # Volume
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        return df
    
    def _get_macro_analysis(self) -> Dict:
        """Get macro analysis (cached for 1 hour)"""
        now = datetime.now()
        
        # Check cache
        if self.last_macro_check and self.macro_analysis:
            age = (now - self.last_macro_check).seconds
            if age < 3600:  # 1 hour cache
                return self.macro_analysis
        
        # Fetch new macro data
        macro_data = self.macro_fetcher.get_macro_summary()
        self.macro_cache = macro_data
        
        # Analyze with Grok
        self.macro_analysis = self.grok.analyze_macro(macro_data)
        self.last_macro_check = now
        
        return self.macro_analysis
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals (rule-based for backtesting speed)"""
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        
        # Get macro regime (use default for backtesting)
        # In live trading, this would call Grok
        
        # LONG signals
        long_signal = (
            # Statistical extreme (oversold)
            ((df["zscore"] < -1.5) & (df["rsi"] < 35)) |
            # Trend pullback
            ((df["trend"] >= 1) & (df["rsi"] < 45) & (df["adx"] > 25))
        ) & (df["vol_ratio"] > 0.7)
        
        # SHORT signals
        short_signal = (
            # Statistical extreme (overbought)
            ((df["zscore"] > 1.5) & (df["rsi"] > 65)) |
            # Trend pullback
            ((df["trend"] <= -1) & (df["rsi"] > 55) & (df["adx"] > 25))
        ) & (df["vol_ratio"] > 0.7)
        
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        # Filter: No signals in sideways market
        signals[df["adx"] < 20] = 0
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        """Get signal with full Grok analysis (for live trading)"""
        df = self._calculate_indicators(df)
        
        row = df.iloc[-1]
        price = row["close"]
        atr = row["atr"]
        atrp = row["atrp"]
        
        # Get indicators for Grok
        indicators = {
            "trend": row["trend"],
            "rsi": row["rsi"],
            "macd_hist": row["macd_hist"],
            "adx": row["adx"],
            "atrp": atrp,
            "zscore": row["zscore"]
        }
        
        # Get macro analysis
        macro_analysis = self._get_macro_analysis()
        
        # Get trade analysis from Grok
        trade_analysis = self.grok.analyze_trade(
            symbol=df.name if hasattr(df, "name") else "BTCUSDT",
            indicators=indicators,
            macro_regime=macro_analysis
        )
        
        signal_type = Signal.HOLD
        stop_loss = None
        take_profit = None
        
        if trade_analysis["action"] == "BUY" and trade_analysis["confidence"] >= 60:
            signal_type = Signal.BUY
            stop_loss = price - (atr * trade_analysis["stop_atr_mult"])
            take_profit = price + (atr * trade_analysis["target_atr_mult"])
            
        elif trade_analysis["action"] == "SELL" and trade_analysis["confidence"] >= 60:
            signal_type = Signal.SELL
            stop_loss = price + (atr * trade_analysis["stop_atr_mult"])
            take_profit = price - (atr * trade_analysis["target_atr_mult"])
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name if hasattr(df, "name") else "UNKNOWN",
            price=price,
            timestamp=df.index[-1],
            confidence=trade_analysis["confidence"] / 100,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "strategy": self.name,
                "macro_regime": macro_analysis.get("regime"),
                "macro_bias": macro_analysis.get("bias"),
                "trade_size": trade_analysis.get("size_pct"),
                "ai_reasoning": trade_analysis.get("reasoning", "")[:200],
                "rsi": row["rsi"],
                "zscore": row["zscore"],
                "atrp": atrp
            }
        )


def get_macro_strategy(name: str) -> BaseStrategy:
    strategies = {
        "macro": MacroQuantStrategy(),
    }
    return strategies.get(name.lower(), MacroQuantStrategy())


if __name__ == "__main__":
    print("=" * 60)
    print("🏦 MACRO QUANT v10 - Anton Kreil Style")
    print("=" * 60)
    
    # Test macro data
    print("\n📊 Fetching macro data...")
    fetcher = MacroDataFetcher()
    macro = fetcher.get_macro_summary()
    
    print(f"  VIX: {macro.get('vix', 'N/A')}")
    print(f"  Fear & Greed: {macro.get('fear_greed', 'N/A')}")
    print(f"  SPY Trend: {macro.get('spy_trend', 'N/A')}")
    print(f"  BTC Dominance: {macro.get('btc_dominance', 'N/A')}")
    
    # Test Grok
    print("\n🤖 Testing Grok AI...")
    grok = GrokAnalyzer()
    
    if grok.enabled:
        analysis = grok.analyze_macro(macro)
        print(f"  Regime: {analysis['regime']}")
        print(f"  Trade Size: {analysis['trade_size']}")
        print(f"  Bias: {analysis['bias']}")
        print(f"  Reasoning: {analysis['reasoning'][:100]}...")
    else:
        print("  Grok disabled - set XAI_API_KEY")
    
    print("\n✅ Macro Quant ready!")
    print("\nUsage:")
    print("  export XAI_API_KEY=your_key")
    print("  python main.py --strategy macro --mode paper")
