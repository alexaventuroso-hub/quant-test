#!/usr/bin/env python3
"""
FAST TRADER - Actually trades!
Simpler rules, trust AI more, lower thresholds
"""
import os, sys, time, requests, pandas as pd
from datetime import datetime, timezone

class FastTrader:
    def __init__(self, symbol="ETHUSDT", capital=10000, timeframe="15m"):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.timeframe = timeframe
        self.position = None
        self.trades = []
        self.best_price = None
        self.api_key = os.getenv("GROQ_API_KEY")
        
    def fetch_data(self):
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines", 
                params={"symbol": self.symbol, "interval": self.timeframe, "limit": 100}, timeout=10)
            df = pd.DataFrame(r.json(), columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            return df
        except:
            return None
    
    def calc_indicators(self, df):
        df = df.copy()
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        ema12, ema26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['mom'] = (df['close'] / df['close'].shift(3) - 1) * 100
        return df
    
    def get_ai_signal(self, price, rsi, macd, mom, trend):
        if not self.api_key:
            return 0, 50
        prompt = f"""{self.symbol} ${price:.0f} RSI:{rsi:.0f} MACD:{'+'if macd>0 else'-'} Mom:{mom:.1f}% Trend:{trend}
Trade now? Reply: BUY 75 or SELL 75 or HOLD 50"""
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "temperature": 0.3},
                timeout=10)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].upper()
                sig = 1 if "BUY" in t else -1 if "SELL" in t else 0
                import re
                nums = re.findall(r'\d+', t)
                conf = int(nums[0]) if nums else 60
                return sig, min(90, conf)
        except:
            pass
        return 0, 50
    
    def run_cycle(self):
        now = datetime.now(timezone.utc)
        df = self.fetch_data()
        if df is None or len(df) < 30:
            print("No data")
            return
        
        df = self.calc_indicators(df)
        r = df.iloc[-1]
        prev = df.iloc[-2]
        
        price, rsi, macd, atr = r['close'], r['rsi'], r['macd_hist'], r['atr']
        trend = "UP" if r['ema8'] > r['ema21'] else "DOWN"
        mom = r['mom']
        
        # Simple rule signals
        rule = 0
        reason = ""
        
        # EMA crossover
        if r['ema8'] > r['ema21'] and prev['ema8'] <= prev['ema21']:
            rule = 1
            reason = "EMA cross UP"
        elif r['ema8'] < r['ema21'] and prev['ema8'] >= prev['ema21']:
            rule = -1
            reason = "EMA cross DOWN"
        # MACD crossover
        elif r['macd_hist'] > 0 and prev['macd_hist'] <= 0:
            rule = 1
            reason = "MACD cross UP"
        elif r['macd_hist'] < 0 and prev['macd_hist'] >= 0:
            rule = -1
            reason = "MACD cross DOWN"
        # RSI extremes with trend
        elif rsi < 35 and trend == "UP":
            rule = 1
            reason = "RSI oversold + uptrend"
        elif rsi > 65 and trend == "DOWN":
            rule = -1
            reason = "RSI overbought + downtrend"
        # Strong momentum
        elif mom > 0.5 and trend == "UP" and macd > 0:
            rule = 1
            reason = "Strong momentum UP"
        elif mom < -0.5 and trend == "DOWN" and macd < 0:
            rule = -1
            reason = "Strong momentum DOWN"
        
        # Get AI
        ai_sig, ai_conf = self.get_ai_signal(price, rsi, macd, mom, trend)
        
        # Final signal - MORE AGGRESSIVE
        final = 0
        conf = 0
        
        if rule != 0:
            # Rule triggered - check AI agrees or is neutral
            if ai_sig == rule:
                final = rule
                conf = ai_conf + 10
            elif ai_sig == 0:  # AI neutral, trust rules
                final = rule
                conf = 65
        elif ai_sig != 0 and ai_conf >= 70:
            # No rule but AI confident
            final = ai_sig
            conf = ai_conf
        
        # Display
        print(f"\n{'='*55}")
        print(f"{now.strftime('%H:%M:%S')} | {self.symbol} {self.timeframe} | {trend}")
        print(f"{'='*55}")
        print(f"${price:.2f} | RSI {rsi:.0f} | MACD {'+'if macd>0 else '-'} | Mom {mom:+.1f}%")
        print(f"Rule: {'BUY' if rule==1 else 'SELL' if rule==-1 else '-'} {reason}")
        print(f"AI: {'BUY' if ai_sig==1 else 'SELL' if ai_sig==-1 else 'HOLD'} ({ai_conf}%)")
        
        if final != 0:
            print(f">>> {'BUY' if final==1 else 'SELL'} SIGNAL ({conf}%) <<<")
        
        # Trailing stop
        if self.position and self.best_price:
            if self.position['side'] == "LONG" and price > self.best_price:
                self.best_price = price
                ns = self.best_price - 1.3 * atr
                if ns > self.position['stop']:
                    self.position['stop'] = ns
                    print(f"Trail -> ${ns:.2f}")
            elif self.position['side'] == "SHORT" and price < self.best_price:
                self.best_price = price
                ns = self.best_price + 1.3 * atr
                if ns < self.position['stop']:
                    self.position['stop'] = ns
                    print(f"Trail -> ${ns:.2f}")
        
        # Execute
        self._execute(price, atr, final, conf)
        self._status(price)
    
    def _execute(self, price, atr, signal, conf):
        # Check existing position
        if self.position:
            side = self.position['side']
            # Stop loss
            if (side == "LONG" and price <= self.position['stop']) or \
               (side == "SHORT" and price >= self.position['stop']):
                self._close("STOP", price)
            # Take profit
            elif (side == "LONG" and price >= self.position['target']) or \
                 (side == "SHORT" and price <= self.position['target']):
                self._close("TARGET", price)
            # Reverse on strong opposite signal
            elif (side == "LONG" and signal == -1 and conf >= 70) or \
                 (side == "SHORT" and signal == 1 and conf >= 70):
                self._close("REVERSE", price)
        
        # Open new position
        if self.position is None and signal != 0 and conf >= 60:
            size = (self.capital * 0.3) / price  # 30% position
            
            if signal == 1:
                stop = price - 1.5 * atr
                target = price + 2.5 * atr
                self.position = {"side": "LONG", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"\n*** OPEN LONG {size:.4f} @ ${price:.2f} ***")
                print(f"    Stop: ${stop:.2f} | Target: ${target:.2f}")
            else:
                stop = price + 1.5 * atr
                target = price - 2.5 * atr
                self.position = {"side": "SHORT", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"\n*** OPEN SHORT {size:.4f} @ ${price:.2f} ***")
                print(f"    Stop: ${stop:.2f} | Target: ${target:.2f}")
    
    def _close(self, reason, price):
        side = self.position['side']
        pnl = (price - self.position['entry']) * self.position['size'] if side == "LONG" else (self.position['entry'] - price) * self.position['size']
        pnl -= price * self.position['size'] * 0.0008  # Commission
        self.capital += pnl
        self.trades.append({"pnl": pnl, "reason": reason})
        
        win = "WIN" if pnl > 0 else "LOSS"
        print(f"\n*** {win} - CLOSE {side} ({reason}) ${pnl:+.2f} ***")
        self.position = None
        self.best_price = None
    
    def _status(self, price):
        pct = (self.capital / self.initial_capital - 1) * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = len(self.trades)
        
        print(f"\nCapital: ${self.capital:.2f} ({pct:+.2f}%)")
        if total > 0:
            print(f"Trades: {total} | Wins: {wins} ({wins/total*100:.0f}%)")
        
        if self.position:
            side = self.position['side']
            entry = self.position['entry']
            u = (price - entry) * self.position['size'] if side == "LONG" else (entry - price) * self.position['size']
            pct_move = ((price / entry) - 1) * 100 if side == "LONG" else ((entry / price) - 1) * 100
            print(f"POSITION: {side} @ ${entry:.2f} | Now ${price:.2f} ({pct_move:+.2f}%) | P&L ${u:+.2f}")
    
    def run(self, interval=30):
        print(f"FAST TRADER | {self.symbol} {self.timeframe}")
        print(f"Checking every {interval}s | Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\n{'='*55}")
            print("FINAL REPORT")
            print(f"{'='*55}")
            print(f"${self.initial_capital:.2f} -> ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
            if self.trades:
                wins = [t for t in self.trades if t['pnl'] > 0]
                losses = [t for t in self.trades if t['pnl'] <= 0]
                print(f"Trades: {len(self.trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
                if wins: print(f"Avg Win: ${sum(t['pnl'] for t in wins)/len(wins):.2f}")
                if losses: print(f"Avg Loss: ${sum(t['pnl'] for t in losses)/len(losses):.2f}")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    FastTrader(symbol=symbol, timeframe=tf).run(30)
