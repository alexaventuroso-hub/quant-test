#!/usr/bin/env python3
"""
PAPER TRADER - AI-FIRST with Groq
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime
from data_fetcher import BinanceDataFetcher
from config import APIConfig

class GroqAI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        
    def analyze(self, symbol, rsi, macd, trend, adx, price):
        if not self.api_key:
            return {"action": "HOLD", "confidence": 0}
        
        prompt = f"""Analyze {symbol} at ${price:.2f}:
- RSI: {rsi:.1f} (oversold<30, overbought>70)  
- MACD: {'Bullish' if macd > 0 else 'Bearish'}
- Trend: {'Up' if trend > 0 else 'Down' if trend < 0 else 'Flat'}
- ADX: {adx:.1f} (>25 = strong)

Be aggressive! Reply ONLY:
ACTION: BUY or SELL or HOLD
CONFIDENCE: 60-100"""

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 50, "temperature": 0.3},
                timeout=15
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                action = "BUY" if "BUY" in text.upper() else "SELL" if "SELL" in text.upper() else "HOLD"
                import re
                conf = re.search(r"(\d+)", text)
                confidence = int(conf.group(1)) if conf else 70
                return {"action": action, "confidence": min(confidence, 100)}
        except Exception as e:
            print(f"AI Error: {e}")
        return {"action": "HOLD", "confidence": 0}


class PaperTrader:
    def __init__(self, symbol="BTCUSDT", capital=10000, timeframe="5m"):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.timeframe = timeframe
        self.position = None
        self.trades = []
        self.ai = GroqAI()
        self.config = APIConfig()
        self.fetcher = BinanceDataFetcher(self.config)
        
    def calc_indicators(self, df):
        df = df.copy()
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_hist'] = ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        tr = pd.concat([df['high'] - df['low'], 
                        (df['high'] - df['close'].shift()).abs(),
                        (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        plus_dm = df['high'].diff().where(lambda x: x > 0, 0)
        minus_dm = (-df['low'].diff()).where(lambda x: x > 0, 0)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr14
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14
        df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 0.001)).rolling(14).mean()
        
        df['trend'] = ((df['close'] > df['ema9']).astype(int) + 
                       (df['ema9'] > df['ema21']).astype(int) - 1)
        return df
        
    def fetch_data(self):
        return self.fetcher.get_historical_data(self.symbol, self.timeframe, 3)
    
    def run_cycle(self):
        print(f"\n{'='*50}")
        print(f"📊 {datetime.now().strftime('%H:%M:%S')} | {self.symbol} {self.timeframe}")
        print(f"{'='*50}")
        
        df = self.fetch_data()
        if df is None or len(df) < 30:
            print("❌ No data")
            return
        
        df = self.calc_indicators(df)
        row = df.iloc[-1]
        price = row['close']
        
        print(f"💰 ${price:.2f} | RSI {row['rsi']:.0f} | MACD {'🟢' if row['macd_hist'] > 0 else '🔴'} | ADX {row['adx']:.0f}")
        
        # AI decides!
        ai = self.ai.analyze(self.symbol, row['rsi'], row['macd_hist'], row['trend'], row['adx'], price)
        print(f"🤖 AI: {ai['action']} ({ai['confidence']}%)")
        
        # Execute if confidence >= 65
        if ai['confidence'] >= 65:
            self._execute(ai['action'], price, row['atr'])
        else:
            print("⏸️ Low confidence - holding")
            
        self._check_stops(price)
        self._print_status(price)
    
    def _execute(self, signal, price, atr):
        if self.position is None:
            if signal in ["BUY", "SELL"]:
                size = (self.capital * 0.20) / price  # 20% position
                side = "LONG" if signal == "BUY" else "SHORT"
                
                if side == "LONG":
                    stop = price - 2 * atr
                    target = price + 4 * atr
                else:
                    stop = price + 2 * atr
                    target = price - 4 * atr
                
                self.position = {"side": side, "size": size, "entry": price, "stop": stop, "target": target}
                emoji = "🟢" if side == "LONG" else "🔴"
                print(f"{emoji} OPEN {side}: {size:.4f} @ ${price:.2f}")
                print(f"   SL: ${stop:.2f} | TP: ${target:.2f}")
        else:
            # Reverse signal?
            if (self.position['side'] == "LONG" and signal == "SELL") or \
               (self.position['side'] == "SHORT" and signal == "BUY"):
                self._close("REVERSE", price)
    
    def _check_stops(self, price):
        if not self.position:
            return
        
        if self.position['side'] == "LONG":
            if price <= self.position['stop']:
                self._close("STOP", price)
            elif price >= self.position['target']:
                self._close("TARGET", price)
        else:
            if price >= self.position['stop']:
                self._close("STOP", price)
            elif price <= self.position['target']:
                self._close("TARGET", price)
    
    def _close(self, reason, price):
        if self.position['side'] == "LONG":
            pnl = (price - self.position['entry']) * self.position['size']
        else:
            pnl = (self.position['entry'] - price) * self.position['size']
        
        self.capital += pnl
        emoji = "✅" if pnl > 0 else "❌"
        print(f"{emoji} CLOSE {self.position['side']} ({reason}): ${pnl:+.2f}")
        self.trades.append({"pnl": pnl, "reason": reason})
        self.position = None
    
    def _print_status(self, price):
        pnl_pct = (self.capital / self.initial_capital - 1) * 100
        print(f"\n💼 ${self.capital:.2f} ({pnl_pct:+.2f}%) | Trades: {len(self.trades)}")
        if self.position:
            if self.position['side'] == "LONG":
                unreal = (price - self.position['entry']) * self.position['size']
            else:
                unreal = (self.position['entry'] - price) * self.position['size']
            print(f"📍 {self.position['side']} @ ${self.position['entry']:.2f} | P&L: ${unreal:+.2f}")
    
    def run(self, interval=30):
        print("🚀 AI PAPER TRADER")
        print(f"Symbol: {self.symbol} | TF: {self.timeframe}")
        print("Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\n🛑 Final: ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
            print(f"Trades: {len(self.trades)} | Wins: {sum(1 for t in self.trades if t['pnl'] > 0)}")


if __name__ == "__main__":
    trader = PaperTrader(symbol="BTCUSDT", capital=10000, timeframe="5m")
    trader.run(interval=30)
