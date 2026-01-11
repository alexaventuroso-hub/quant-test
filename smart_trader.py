#!/usr/bin/env python3
"""
SMART AI TRADER - ETHUSDT 4h
"""
import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from data_fetcher import BinanceDataFetcher
from config import APIConfig

class GroqAI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        
    def analyze(self, symbol, rsi, macd, adx, zscore, price):
        if not self.api_key:
            return {"signal": 0, "confidence": 0, "pattern": "none"}
        
        prompt = f"""Goldman Sachs quant analysis for {symbol} at ${price:.2f}:
RSI: {rsi:.0f} | MACD: {'Bull' if macd>0 else 'Bear'} | ADX: {adx:.0f} | Z-score: {zscore:.2f}

Reply EXACTLY:
SIGNAL: BUY or SELL or HOLD
CONFIDENCE: 0-100
PATTERN: detected pattern name"""

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 80, "temperature": 0.1},
                timeout=15
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                signal = 1 if "BUY" in text.upper() else -1 if "SELL" in text.upper() else 0
                import re
                conf = re.search(r"CONFIDENCE:\s*(\d+)", text.upper())
                pat = re.search(r"PATTERN:\s*([^\n]+)", text)
                return {
                    "signal": signal,
                    "confidence": int(conf.group(1)) if conf else 50,
                    "pattern": pat.group(1).strip() if pat else "unknown"
                }
        except:
            pass
        return {"signal": 0, "confidence": 0, "pattern": "error"}


class SmartTrader:
    def __init__(self, symbol="ETHUSDT", capital=10000, timeframe="4h"):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.timeframe = timeframe
        self.position = None
        self.trades = []
        self.best_price = None
        
        self.ai = GroqAI()
        self.config = APIConfig()
        self.fetcher = BinanceDataFetcher(self.config)
        
    def calc_indicators(self, df):
        df = df.copy()
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_hist'] = ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()
        
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
        return df
        
    def fetch_data(self):
        return self.fetcher.get_historical_data(self.symbol, self.timeframe, 30)
    
    def get_signal(self, df):
        row = df.iloc[-1]
        
        # Rule-based
        rule = 0
        if row['ema8'] > row['ema21'] and row['macd_hist'] > 0 and row['rsi'] < 65:
            rule = 1
        elif row['ema8'] < row['ema21'] and row['macd_hist'] < 0 and row['rsi'] > 35:
            rule = -1
        
        # AI signal
        ai = self.ai.analyze(self.symbol, row['rsi'], row['macd_hist'], row['adx'], row['zscore'], row['close'])
        
        # Combine
        final = 0
        conf = 0
        if rule == ai['signal'] and ai['confidence'] >= 60:
            final = rule
            conf = ai['confidence'] + 20
        elif ai['confidence'] >= 75:
            final = ai['signal']
            conf = ai['confidence']
        elif rule != 0:
            final = rule
            conf = 55
        
        return {
            "signal": final, "confidence": min(conf, 100),
            "rule": rule, "ai": ai['signal'], "ai_conf": ai['confidence'],
            "pattern": ai['pattern'], "price": row['close'], "atr": row['atr'],
            "rsi": row['rsi'], "adx": row['adx']
        }
    
    def run_cycle(self):
        print(f"\n{'='*60}")
        print(f"🤖 {datetime.utcnow().strftime('%H:%M:%S')} UTC | {self.symbol} {self.timeframe}")
        print(f"{'='*60}")
        
        df = self.fetch_data()
        if df is None or len(df) < 50:
            print("❌ No data")
            return
        
        df = self.calc_indicators(df)
        sig = self.get_signal(df)
        
        print(f"💰 ${sig['price']:.2f} | RSI {sig['rsi']:.0f} | ADX {sig['adx']:.0f}")
        print(f"📋 Rule: {'BUY' if sig['rule']==1 else 'SELL' if sig['rule']==-1 else 'HOLD'}")
        print(f"🤖 AI: {'BUY' if sig['ai']==1 else 'SELL' if sig['ai']==-1 else 'HOLD'} ({sig['ai_conf']}%) - {sig['pattern']}")
        print(f"✅ Final: {'BUY' if sig['signal']==1 else 'SELL' if sig['signal']==-1 else 'HOLD'} ({sig['confidence']}%)")
        
        # Trailing stop update
        if self.position and self.best_price:
            if self.position['side'] == "LONG" and sig['price'] > self.best_price:
                self.best_price = sig['price']
                new_stop = self.best_price - 1.5 * sig['atr']
                if new_stop > self.position['stop']:
                    print(f"📈 Trail: ${self.position['stop']:.2f} → ${new_stop:.2f}")
                    self.position['stop'] = new_stop
            elif self.position['side'] == "SHORT" and sig['price'] < self.best_price:
                self.best_price = sig['price']
                new_stop = self.best_price + 1.5 * sig['atr']
                if new_stop < self.position['stop']:
                    print(f"📉 Trail: ${self.position['stop']:.2f} → ${new_stop:.2f}")
                    self.position['stop'] = new_stop
        
        # Execute
        if sig['confidence'] >= 60:
            self._execute(sig)
        
        self._print_status(sig['price'])
    
    def _execute(self, sig):
        price, atr, signal = sig['price'], sig['atr'], sig['signal']
        
        if self.position:
            side = self.position['side']
            if (side == "LONG" and price <= self.position['stop']) or \
               (side == "SHORT" and price >= self.position['stop']):
                self._close("STOP", price)
            elif (side == "LONG" and price >= self.position['target']) or \
                 (side == "SHORT" and price <= self.position['target']):
                self._close("TARGET", price)
            elif (side == "LONG" and signal == -1) or (side == "SHORT" and signal == 1):
                self._close("REVERSE", price)
        
        if self.position is None and signal != 0:
            size = (self.capital * 0.20) / price
            if signal == 1:
                stop, target = price - 2*atr, price + 4*atr
                self.position = {"side": "LONG", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"🟢 LONG {size:.4f} @ ${price:.2f} | SL ${stop:.2f} | TP ${target:.2f}")
            else:
                stop, target = price + 2*atr, price - 4*atr
                self.position = {"side": "SHORT", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"🔴 SHORT {size:.4f} @ ${price:.2f} | SL ${stop:.2f} | TP ${target:.2f}")
    
    def _close(self, reason, price):
        pnl = (price - self.position['entry']) * self.position['size'] if self.position['side'] == "LONG" else (self.position['entry'] - price) * self.position['size']
        pnl -= price * self.position['size'] * 0.0008  # Commission
        self.capital += pnl
        self.trades.append({"pnl": pnl, "reason": reason})
        print(f"{'✅' if pnl > 0 else '❌'} CLOSE {self.position['side']} ({reason}): ${pnl:+.2f}")
        self.position = None
        self.best_price = None
    
    def _print_status(self, price):
        pnl_pct = (self.capital / self.initial_capital - 1) * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        print(f"\n💼 ${self.capital:.2f} ({pnl_pct:+.2f}%) | Trades: {len(self.trades)} | Wins: {wins}")
        if self.position:
            unreal = (price - self.position['entry']) * self.position['size'] if self.position['side'] == "LONG" else (self.position['entry'] - price) * self.position['size']
            print(f"📍 {self.position['side']} @ ${self.position['entry']:.2f} | P&L: ${unreal:+.2f}")
    
    def run(self, interval=60):
        print(f"🚀 SMART TRADER | {self.symbol} {self.timeframe}")
        print("Ctrl+C to stop\n")
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    print(f"⚠️ {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\n🛑 Final: ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
            print(f"Trades: {len(self.trades)} | Wins: {sum(1 for t in self.trades if t['pnl'] > 0)}")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "4h"
    SmartTrader(symbol=symbol, timeframe=tf).run(interval=60)
