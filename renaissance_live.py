#!/usr/bin/env python3
"""
RENAISSANCE LIVE TRADER
Best setups: ETHUSDT 15m (Sharpe 6.64), BNBUSDT 4h, ETHUSDT 4h
"""
import os, sys, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

class RenaissanceLive:
    def __init__(self, symbol="ETHUSDT", timeframe="15m", capital=10000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.capital = capital
        self.initial_capital = capital
        self.position = None
        self.trades = []
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # Kelly tracking
        self.trade_history = []
        
    def fetch_data(self, limit=200):
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": self.symbol, "interval": self.timeframe, "limit": limit}, timeout=10)
            data = r.json()
            if not data: return None
            df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('ts', inplace=True)
            return df
        except:
            return None
    
    def calc_indicators(self, df):
        df = df.copy()
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        ema12, ema26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
        
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
        
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
        
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['range_pos'] = (df['close'] - df['support']) / (df['resistance'] - df['support'] + 1e-10)
        
        return df
    
    def detect_regime(self, df):
        row = df.iloc[-1]
        strength = 0
        if row['ema8'] > row['ema21'] > row['ema50']: strength += 2
        elif row['ema8'] < row['ema21'] < row['ema50']: strength -= 2
        if row['adx'] > 25:
            strength += 1 if row['plus_di'] > row['minus_di'] else -1
        
        if strength >= 2: return 'BULL', 0.7
        elif strength <= -2: return 'BEAR', 0.7
        return 'SIDEWAYS', 0.6
    
    def get_signals(self, df):
        row, prev = df.iloc[-1], df.iloc[-2]
        signals = {}
        
        # Mean reversion
        sig, str_ = 0, 0
        if row['zscore'] < -1.5 and row['bb_pct'] < 0.15 and row['rsi'] < 35:
            if row['rsi'] > prev['rsi']: sig, str_ = 1, 0.8
        elif row['zscore'] > 1.5 and row['bb_pct'] > 0.85 and row['rsi'] > 65:
            if row['rsi'] < prev['rsi']: sig, str_ = -1, 0.8
        signals['mean_rev'] = (sig, str_)
        
        # Momentum
        sig, str_ = 0, 0
        if row['macd_hist'] > 0 and prev['macd_hist'] <= 0 and row['ema8'] > row['ema21']:
            sig, str_ = 1, 0.7
        elif row['macd_hist'] < 0 and prev['macd_hist'] >= 0 and row['ema8'] < row['ema21']:
            sig, str_ = -1, 0.7
        signals['momentum'] = (sig, str_)
        
        # Volatility breakout
        sig, str_ = 0, 0
        if row['close'] > row['bb_upper']: sig, str_ = 1, 0.6
        elif row['close'] < row['bb_lower']: sig, str_ = -1, 0.6
        signals['vol_break'] = (sig, str_)
        
        # Support/Resistance
        sig, str_ = 0, 0
        if row['range_pos'] < 0.2 and row['ema21'] > row['ema50'] and row['rsi'] < 40:
            sig, str_ = 1, 0.65
        elif row['range_pos'] > 0.8 and row['ema21'] < row['ema50'] and row['rsi'] > 60:
            sig, str_ = -1, 0.65
        signals['sr'] = (sig, str_)
        
        return signals
    
    def kelly_size(self, confidence):
        if len(self.trade_history) < 5:
            return 0.1
        wins = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t <= 0]
        if not wins or not losses: return 0.1
        
        p = len(wins) / len(self.trade_history) * (0.5 + confidence * 0.5)
        b = np.mean(wins) / abs(np.mean(losses)) if losses else 1
        kelly = max(0, (p * b - (1-p)) / b) * 0.5  # Half Kelly
        return min(kelly, 0.25)
    
    def ai_decision(self, row, signals, regime):
        if not self.api_key:
            return self._rule_decision(signals, regime)
        
        prompt = f"""Renaissance-style quant decision for {self.symbol}:

PRICE: ${row['close']:.2f} | REGIME: {regime[0]} ({regime[1]:.0%})
RSI: {row['rsi']:.0f} | Z-Score: {row['zscore']:.2f} | ADX: {row['adx']:.0f}
MACD: {'Bullish' if row['macd_hist']>0 else 'Bearish'}

SIGNALS:
- Mean Reversion: {'LONG' if signals['mean_rev'][0]==1 else 'SHORT' if signals['mean_rev'][0]==-1 else '-'} ({signals['mean_rev'][1]:.0%})
- Momentum: {'LONG' if signals['momentum'][0]==1 else 'SHORT' if signals['momentum'][0]==-1 else '-'} ({signals['momentum'][1]:.0%})
- Vol Breakout: {'LONG' if signals['vol_break'][0]==1 else 'SHORT' if signals['vol_break'][0]==-1 else '-'} ({signals['vol_break'][1]:.0%})
- S/R: {'LONG' if signals['sr'][0]==1 else 'SHORT' if signals['sr'][0]==-1 else '-'} ({signals['sr'][1]:.0%})

Rules: SIDEWAYS=mean_reversion, BULL/BEAR=momentum with trend, need 2+ agreeing signals.
Reply: ACTION: BUY/SELL/WAIT | CONFIDENCE: 50-90 | REASON: brief"""

        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 100, "temperature": 0.3}, timeout=15)
            if r.status_code == 200:
                import re
                text = r.json()["choices"][0]["message"]["content"]
                action = "BUY" if "BUY" in text.upper() else "SELL" if "SELL" in text.upper() else "WAIT"
                conf = int(re.search(r'CONFIDENCE[:\s]*(\d+)', text.upper()).group(1))/100 if re.search(r'CONFIDENCE[:\s]*(\d+)', text.upper()) else 0.5
                reason = re.search(r'REASON[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
                reason = reason.group(1).strip()[:50] if reason else "AI"
                return {'action': action, 'confidence': min(conf, 0.9), 'reason': reason}
        except:
            pass
        return self._rule_decision(signals, regime)
    
    def _rule_decision(self, signals, regime):
        long_count = sum(1 for s in signals.values() if s[0] == 1)
        short_count = sum(1 for s in signals.values() if s[0] == -1)
        avg_str = np.mean([s[1] for s in signals.values() if s[0] != 0]) if any(s[0] != 0 for s in signals.values()) else 0
        
        action = "WAIT"
        if long_count >= 2 and short_count == 0 and regime[0] != 'BEAR':
            action = "BUY"
        elif short_count >= 2 and long_count == 0 and regime[0] != 'BULL':
            action = "SELL"
        
        return {'action': action, 'confidence': avg_str, 'reason': f"{long_count}L/{short_count}S signals"}
    
    def run_cycle(self):
        now = datetime.now(timezone.utc)
        df = self.fetch_data()
        if df is None or len(df) < 50:
            print("No data")
            return
        
        df = self.calc_indicators(df)
        row = df.iloc[-1]
        regime = self.detect_regime(df)
        signals = self.get_signals(df)
        decision = self.ai_decision(row, signals, regime)
        
        # Display
        print(f"\n{'='*65}")
        print(f"🎯 {now.strftime('%H:%M:%S')} | {self.symbol} {self.timeframe} | Regime: {regime[0]}")
        print(f"{'='*65}")
        print(f"💰 ${row['close']:.2f} | RSI {row['rsi']:.0f} | Z:{row['zscore']:+.2f} | ADX {row['adx']:.0f}")
        
        sig_str = []
        for name, (sig, strength) in signals.items():
            if sig != 0:
                sig_str.append(f"{name}:{'L' if sig==1 else 'S'}({strength:.0%})")
        print(f"📊 Signals: {', '.join(sig_str) if sig_str else 'None'}")
        print(f"🤖 AI: {decision['action']} ({decision['confidence']:.0%}) - {decision['reason']}")
        
        # Position management
        price, atr = row['close'], row['atr']
        
        if self.position:
            side = self.position['side']
            entry = self.position['entry']
            unrealized = (price - entry) * self.position['size'] if side == 'LONG' else (entry - price) * self.position['size']
            pct = ((price/entry)-1)*100 if side == 'LONG' else ((entry/price)-1)*100
            
            # Check exits
            hit_stop = (side == 'LONG' and price <= self.position['stop']) or (side == 'SHORT' and price >= self.position['stop'])
            hit_target = (side == 'LONG' and price >= self.position['target']) or (side == 'SHORT' and price <= self.position['target'])
            reverse = (side == 'LONG' and decision['action'] == 'SELL' and decision['confidence'] > 0.6) or \
                      (side == 'SHORT' and decision['action'] == 'BUY' and decision['confidence'] > 0.6)
            
            if hit_stop or hit_target or reverse:
                pnl = unrealized - price * self.position['size'] * 0.0008
                self.capital += pnl
                self.trades.append({'pnl': pnl})
                self.trade_history.append(pnl)
                if len(self.trade_history) > 50: self.trade_history = self.trade_history[-50:]
                
                reason = "STOP" if hit_stop else "TARGET" if hit_target else "REVERSE"
                emoji = "✅" if pnl > 0 else "❌"
                print(f"\n{emoji} CLOSE {side} ({reason}): ${pnl:+.2f}")
                self.position = None
        
        # Open new position
        if self.position is None and decision['action'] != 'WAIT' and decision['confidence'] > 0.55:
            kelly = self.kelly_size(decision['confidence'])
            size = (self.capital * kelly) / price
            
            if decision['action'] == 'BUY':
                self.position = {
                    'side': 'LONG', 'entry': price, 'size': size,
                    'stop': price - 2 * atr, 'target': price + 3 * atr
                }
                print(f"\n🟢 OPEN LONG {size:.4f} @ ${price:.2f} (Kelly: {kelly:.1%})")
                print(f"   SL: ${self.position['stop']:.2f} | TP: ${self.position['target']:.2f}")
            else:
                self.position = {
                    'side': 'SHORT', 'entry': price, 'size': size,
                    'stop': price + 2 * atr, 'target': price - 3 * atr
                }
                print(f"\n🔴 OPEN SHORT {size:.4f} @ ${price:.2f} (Kelly: {kelly:.1%})")
                print(f"   SL: ${self.position['stop']:.2f} | TP: ${self.position['target']:.2f}")
        
        # Status
        self._status(price)
    
    def _status(self, price):
        pct = (self.capital / self.initial_capital - 1) * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = len(self.trades)
        
        print(f"\n{'─'*65}")
        print(f"💼 ${self.capital:.2f} ({pct:+.2f}%)", end="")
        if total > 0:
            print(f" | Trades: {total} | Wins: {wins} ({wins/total*100:.0f}%)")
        else:
            print()
        
        if self.position:
            side, entry = self.position['side'], self.position['entry']
            u = (price - entry) * self.position['size'] if side == 'LONG' else (entry - price) * self.position['size']
            print(f"📍 {side} @ ${entry:.2f} -> ${price:.2f} | P&L: ${u:+.2f}")
    
    def run(self, interval=30):
        tf_intervals = {'5m': 20, '15m': 30, '1h': 60, '4h': 120}
        interval = tf_intervals.get(self.timeframe, 30)
        
        print(f"🎯 RENAISSANCE LIVE TRADER")
        print(f"📊 {self.symbol} {self.timeframe}")
        print(f"💰 Capital: ${self.capital}")
        print(f"⏱️  Interval: {interval}s")
        print(f"🤖 AI: {'Enabled' if self.api_key else 'Rule-based'}")
        print(f"\nCtrl+C to stop\n")
        
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\n{'='*65}")
            print("FINAL REPORT")
            print(f"{'='*65}")
            print(f"${self.initial_capital:.2f} -> ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
            if self.trades:
                wins = [t for t in self.trades if t['pnl'] > 0]
                losses = [t for t in self.trades if t['pnl'] <= 0]
                print(f"Trades: {len(self.trades)} | Wins: {len(wins)} ({len(wins)/len(self.trades)*100:.0f}%)")
                if wins: print(f"Avg Win: ${np.mean([t['pnl'] for t in wins]):.2f}")
                if losses: print(f"Avg Loss: ${np.mean([t['pnl'] for t in losses]):.2f}")


if __name__ == "__main__":
    # Best performers from backtest:
    # 1. ETHUSDT 15m - Sharpe 6.64, 90% win
    # 2. BNBUSDT 4h - Sharpe 3.17, 61% win
    # 3. ETHUSDT 4h - Sharpe 2.33, 57% win
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    
    RenaissanceLive(symbol=symbol, timeframe=tf).run()
