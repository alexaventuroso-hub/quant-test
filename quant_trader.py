#!/usr/bin/env python3
"""QUANT TRADER V2"""
import os, sys, time, requests, pandas as pd
from datetime import datetime, timezone

class QuantAI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.last_signal = None
        self.signal_count = 0
        
    def analyze(self, data):
        if not self.api_key:
            return {"signal": 0, "confidence": 50, "reason": "no api", "consistent": 0}
        
        prompt = f"""Crypto analysis for {data['symbol']} at ${data['price']:.2f}:
EMA5>EMA21: {'YES' if data['ema5'] > data['ema21'] else 'NO'}
RSI: {data['rsi']:.0f}, MACD: {'POS' if data['macd']>0 else 'NEG'}, ADX: {data['adx']:.0f}
Mom5: {data['mom5']:.2f}%

Reply EXACTLY: ACTION: BUY/SELL/HOLD | CONFIDENCE: 60-90 | REASON: brief"""

        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 60, "temperature": 0.2},
                timeout=15)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].upper()
                signal = 1 if "BUY" in text else -1 if "SELL" in text else 0
                import re
                conf = int(re.search(r'CONFIDENCE[:\s]*(\d+)', text).group(1)) if re.search(r'CONFIDENCE[:\s]*(\d+)', text) else 60
                if signal == self.last_signal and signal != 0:
                    self.signal_count += 1
                else:
                    self.signal_count = 1
                self.last_signal = signal
                return {"signal": signal, "confidence": min(90, conf), "reason": "AI", "consistent": self.signal_count}
        except:
            pass
        return {"signal": 0, "confidence": 50, "reason": "error", "consistent": 0}

class QuantTrader:
    def __init__(self, symbol="ETHUSDT", capital=10000, timeframe="1h"):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.timeframe = timeframe
        self.position = None
        self.trades = []
        self.best_price = None
        self.ai = QuantAI()
        self.min_confidence = 70
        self.min_consistency = 2
        
    def fetch_data(self):
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines", 
                params={"symbol": self.symbol, "interval": self.timeframe, "limit": 200}, timeout=10)
            df = pd.DataFrame(r.json(), columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            return df
        except:
            return None
    
    def calc_indicators(self, df):
        df = df.copy()
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        ema12, ema26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
        df['macd_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        plus_dm, minus_dm = df['high'].diff().clip(lower=0), (-df['low'].diff()).clip(lower=0)
        plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        df['mom5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        return df
    
    def get_signal(self, df):
        r = df.iloc[-1]
        data = {'symbol': self.symbol, 'price': r['close'], 'ema5': r['ema5'], 'ema21': r['ema21'],
                'rsi': r['rsi'], 'macd': r['macd_hist'], 'adx': r['adx'], 'mom5': r['mom5']}
        
        # Rule-based (need 4/6)
        long_score = sum([r['ema5']>r['ema21'], r['macd_hist']>0, 40<r['rsi']<70, r['adx']>20, r['mom5']>0, r['close']>r['ema21']])
        short_score = sum([r['ema5']<r['ema21'], r['macd_hist']<0, 30<r['rsi']<60, r['adx']>20, r['mom5']<0, r['close']<r['ema21']])
        rule = 1 if long_score >= 4 else -1 if short_score >= 4 else 0
        
        ai = self.ai.analyze(data)
        final, conf = 0, 0
        if rule == ai['signal'] and ai['signal'] != 0 and ai['confidence'] >= self.min_confidence and ai['consistent'] >= self.min_consistency:
            final, conf = rule, ai['confidence']
        elif ai['confidence'] >= 85 and ai['consistent'] >= 3:
            final, conf = ai['signal'], ai['confidence'] - 5
            
        return {"signal": final, "confidence": conf, "rule": rule, "ai": ai['signal'], "ai_conf": ai['confidence'],
                "consistent": ai['consistent'], "price": r['close'], "atr": r['atr'], "rsi": r['rsi'], "adx": r['adx']}
    
    def run_cycle(self):
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}\n{now.strftime('%H:%M:%S')} | {self.symbol} {self.timeframe}\n{'='*60}")
        
        df = self.fetch_data()
        if df is None or len(df) < 50: return print("No data")
        df = self.calc_indicators(df)
        sig = self.get_signal(df)
        
        print(f"${sig['price']:.2f} | RSI {sig['rsi']:.0f} | ADX {sig['adx']:.0f}")
        print(f"Rule: {'BUY' if sig['rule']==1 else 'SELL' if sig['rule']==-1 else 'WAIT'}")
        print(f"AI: {'BUY' if sig['ai']==1 else 'SELL' if sig['ai']==-1 else 'WAIT'} ({sig['ai_conf']}%) [{sig['consistent']}x]")
        print(f"Signal: {'BUY' if sig['signal']==1 else 'SELL' if sig['signal']==-1 else 'WAIT'} ({sig['confidence']}%)")
        
        # Trailing stop
        if self.position and self.best_price:
            p, atr = sig['price'], sig['atr']
            if self.position['side'] == "LONG" and p > self.best_price:
                self.best_price = p
                ns = self.best_price - 1.5 * atr
                if ns > self.position['stop']: self.position['stop'] = ns; print(f"Trail -> ${ns:.2f}")
            elif self.position['side'] == "SHORT" and p < self.best_price:
                self.best_price = p
                ns = self.best_price + 1.5 * atr
                if ns < self.position['stop']: self.position['stop'] = ns; print(f"Trail -> ${ns:.2f}")
        
        self._execute(sig)
        self._status(sig['price'])
    
    def _execute(self, sig):
        p, atr, s, c = sig['price'], sig['atr'], sig['signal'], sig['confidence']
        if self.position:
            side = self.position['side']
            if (side=="LONG" and p<=self.position['stop']) or (side=="SHORT" and p>=self.position['stop']): self._close("STOP", p)
            elif (side=="LONG" and p>=self.position['target']) or (side=="SHORT" and p<=self.position['target']): self._close("TARGET", p)
            elif (side=="LONG" and s==-1 and c>=75) or (side=="SHORT" and s==1 and c>=75): self._close("REVERSE", p)
        
        if self.position is None and s != 0 and c >= self.min_confidence:
            size = (self.capital * 0.25) / p
            if s == 1:
                self.position = {"side": "LONG", "size": size, "entry": p, "stop": p - 1.8*atr, "target": p + 3*atr}
                print(f"OPEN LONG {size:.4f} @ ${p:.2f}")
            else:
                self.position = {"side": "SHORT", "size": size, "entry": p, "stop": p + 1.8*atr, "target": p - 3*atr}
                print(f"OPEN SHORT {size:.4f} @ ${p:.2f}")
            self.best_price = p
    
    def _close(self, reason, p):
        pnl = (p - self.position['entry']) * self.position['size'] if self.position['side']=="LONG" else (self.position['entry'] - p) * self.position['size']
        pnl -= p * self.position['size'] * 0.0008
        self.capital += pnl
        self.trades.append({"pnl": pnl})
        print(f"{'WIN' if pnl>0 else 'LOSS'} {self.position['side']} ({reason}): ${pnl:+.2f}")
        self.position = None
        self.best_price = None
    
    def _status(self, p):
        pct = (self.capital / self.initial_capital - 1) * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        print(f"\nCapital: ${self.capital:.2f} ({pct:+.2f}%) | Trades: {len(self.trades)} | Wins: {wins}")
        if self.position:
            u = (p - self.position['entry']) * self.position['size'] if self.position['side']=="LONG" else (self.position['entry'] - p) * self.position['size']
            print(f"Position: {self.position['side']} @ ${self.position['entry']:.2f} | P&L: ${u:+.2f}")
    
    def run(self, interval=45):
        print(f"QUANT TRADER | {self.symbol} {self.timeframe}\nCtrl+C to stop\n")
        try:
            while True:
                try: self.run_cycle()
                except Exception as e: print(f"Error: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\nFINAL: ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
            if self.trades:
                wins = [t for t in self.trades if t['pnl'] > 0]
                print(f"Trades: {len(self.trades)} | Wins: {len(wins)} ({len(wins)/len(self.trades)*100:.0f}%)")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1h"
    QuantTrader(symbol=symbol, timeframe=tf).run(45)
