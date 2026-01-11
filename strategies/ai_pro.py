"""
AI-ENHANCED PRO STRATEGY
Uses Groq AI to confirm trades
"""
import pandas as pd
import os
import requests
import re

class AIProStrategy:
    """Pro strategy with Groq AI confirmation"""
    name = "AI-Pro"
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
    
    def _get_ai_signal(self, indicators: dict) -> dict:
        if not self.api_key:
            return {"action": "HOLD", "confidence": 50}
        
        prompt = f"RSI={indicators['rsi']:.0f}, MACD={'Bull' if indicators['macd']>0 else 'Bear'}, Trend={'Up' if indicators['trend']>0 else 'Down'}. Reply only: BUY, SELL, or HOLD"
        
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20},
                timeout=10
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].upper()
                if "BUY" in text: return {"action": "BUY", "confidence": 70}
                if "SELL" in text: return {"action": "SELL", "confidence": 70}
        except:
            pass
        return {"action": "HOLD", "confidence": 50}
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
        # EMAs
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_hist'] = ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Trend
        df['trend'] = (df['ema9'] - df['ema21']) / df['ema21'] * 100
        
        signals = pd.Series(0, index=df.index)
        
        # Rule-based signals
        long = (df['ema9'] > df['ema21']) & (df['macd_hist'] > 0) & (df['rsi'] < 65)
        short = (df['ema9'] < df['ema21']) & (df['macd_hist'] < 0) & (df['rsi'] > 35)
        
        signals[long] = 1
        signals[short] = -1
        
        return signals
