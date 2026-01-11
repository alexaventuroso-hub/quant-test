import os
import requests
import re
from typing import Dict, Optional

class AIAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("No GROQ_API_KEY set")
    
    @property
    def enabled(self):
        return bool(self.api_key)
    
    def analyze_trade(self, symbol, indicators):
        if not self.enabled:
            return {"action": "HOLD", "confidence": 0, "reasoning": "No API key"}
        
        prompt = f"Analyze {symbol}: RSI={indicators.get('rsi',50):.1f}, MACD={'Bull' if indicators.get('macd_hist',0)>0 else 'Bear'}, ADX={indicators.get('adx',20):.1f}. Reply: ACTION: BUY/SELL/HOLD, CONFIDENCE: 0-100"
        
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100},
                timeout=30
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                action = "BUY" if "BUY" in text.upper() else "SELL" if "SELL" in text.upper() else "HOLD"
                conf = re.search(r"CONFIDENCE:\s*(\d+)", text.upper())
                return {"action": action, "confidence": int(conf.group(1)) if conf else 50, "reasoning": text}
            print(f"Error: {r.status_code} - {r.text[:100]}")
        except Exception as e:
            print(f"Error: {e}")
        return {"action": "HOLD", "confidence": 0, "reasoning": "Error"}

if __name__ == "__main__":
    print("Testing Groq AI (FREE)...")
    ai = AIAnalyzer()
    if ai.enabled:
        result = ai.analyze_trade("BTCUSDT", {"rsi": 35, "macd_hist": 0.5, "adx": 28})
        print(f"Action: {result['action']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning']}")
    else:
        print("Get FREE key at console.groq.com")
