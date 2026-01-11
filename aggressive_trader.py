#!/usr/bin/env python3
"""
AGGRESSIVE AI TRADER - More trades, looser rules
15m timeframe, AI-driven decisions
"""
import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone

class GroqAI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        
    def analyze(self, symbol, rsi, macd, adx, price, trend, momentum):
        if not self.api_key:
            return {"signal": 0, "confidence": 50}
        
        prompt = f"""Quick crypto analysis for {symbol} at ${price:.2f}:
RSI: {rsi:.0f} | MACD: {'↑' if macd>0 else '↓'} | ADX: {adx:.0f} | Trend: {trend} | Mom: {momentum:.1f}%

Quick decision - reply ONLY:
BUY 85
or
SELL 75  
or
HOLD 50

(action + confidence number)"""

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "temperature": 0.1},
                timeout=10
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].upper().strip()
                signal = 1 if "BUY" in text else -1 if "SELL" in text else 0
                import re
                nums = re.findall(r'\d+', text)
                conf = int(nums[0]) if nums else 60
                return {"signal": signal, "confidence": min(conf, 100)}
        except:
            pass
        return {"signal": 0, "confidence": 50}


class AggressiveTrader:
    def __init__(self, symbol="ETHUSDT", capital=10000, timeframe="15m"):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.timeframe = timeframe
        self.position = None
        self.trades = []
        self.best_price = None
        
        self.ai = GroqAI()
        
        # LOOSE parameters for more trades
        self.min_confidence = 55  # Lower threshold
        self.position_size = 0.30  # 30% position
        self.atr_sl_mult = 1.5    # Tighter stop
        self.atr_tp_mult = 2.5    # Reasonable target (1.67:1 R:R)
        
    def fetch_data(self):
        import requests
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": self.symbol, "interval": self.timeframe, "limit": 200}
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'])
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Fetch error: {e}")
            return None
    
    def calc_indicators(self, df):
        df = df.copy()
        
        # Fast EMAs for quick signals
        df['ema5'] = df['close'].ewm(span=5).mean()
        df['ema13'] = df['close'].ewm(span=13).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # ADX
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        
        # Momentum
        df['momentum'] = (df['close'] / df['close'].shift(5) - 1) * 100
        
        # Trend
        df['trend'] = 'FLAT'
        df.loc[df['ema5'] > df['ema13'], 'trend'] = 'UP'
        df.loc[df['ema5'] < df['ema13'], 'trend'] = 'DOWN'
        
        return df
    
    def get_signal(self, df):
        row = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Rule-based signal (LOOSE rules)
        rule = 0
        
        # LONG: EMA crossover OR RSI bounce from oversold OR MACD cross
        long_ema = row['ema5'] > row['ema13'] and prev['ema5'] <= prev['ema13']
        long_rsi = row['rsi'] < 40 and row['rsi'] > prev['rsi']  # RSI bouncing
        long_macd = row['macd_hist'] > 0 and prev['macd_hist'] <= 0
        long_trend = row['ema5'] > row['ema21'] and row['macd_hist'] > 0
        
        # SHORT: opposite conditions
        short_ema = row['ema5'] < row['ema13'] and prev['ema5'] >= prev['ema13']
        short_rsi = row['rsi'] > 60 and row['rsi'] < prev['rsi']
        short_macd = row['macd_hist'] < 0 and prev['macd_hist'] >= 0
        short_trend = row['ema5'] < row['ema21'] and row['macd_hist'] < 0
        
        if long_ema or long_macd or (long_rsi and long_trend):
            rule = 1
        elif short_ema or short_macd or (short_rsi and short_trend):
            rule = -1
        
        # AI signal
        ai = self.ai.analyze(
            self.symbol, row['rsi'], row['macd_hist'], row['adx'],
            row['close'], row['trend'], row['momentum']
        )
        
        # Combine: trust AI more if it's confident
        final = 0
        conf = 0
        
        if ai['confidence'] >= 70:
            final = ai['signal']
            conf = ai['confidence']
        elif rule != 0 and ai['signal'] == rule:
            final = rule
            conf = ai['confidence'] + 15
        elif rule != 0:
            final = rule
            conf = 60
        elif ai['confidence'] >= 60:
            final = ai['signal']
            conf = ai['confidence']
        
        return {
            "signal": final, "confidence": conf,
            "rule": rule, "ai": ai['signal'], "ai_conf": ai['confidence'],
            "price": row['close'], "atr": row['atr'],
            "rsi": row['rsi'], "adx": row['adx'], "trend": row['trend'],
            "macd": row['macd_hist']
        }
    
    def run_cycle(self):
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}")
        print(f"⚡ {now.strftime('%H:%M:%S')} UTC | {self.symbol} {self.timeframe}")
        print(f"{'='*60}")
        
        df = self.fetch_data()
        if df is None or len(df) < 50:
            print("❌ No data")
            return
        
        df = self.calc_indicators(df)
        sig = self.get_signal(df)
        
        # Display
        macd_icon = "🟢" if sig['macd'] > 0 else "🔴"
        trend_icon = "📈" if sig['trend'] == 'UP' else "📉" if sig['trend'] == 'DOWN' else "➡️"
        
        print(f"💰 ${sig['price']:.2f} | RSI {sig['rsi']:.0f} | {macd_icon} MACD | ADX {sig['adx']:.0f}")
        print(f"{trend_icon} Trend: {sig['trend']}")
        print(f"📋 Rule: {'BUY' if sig['rule']==1 else 'SELL' if sig['rule']==-1 else 'WAIT'}")
        print(f"🤖 AI: {'BUY' if sig['ai']==1 else 'SELL' if sig['ai']==-1 else 'WAIT'} ({sig['ai_conf']}%)")
        print(f"✅ Signal: {'🟢 BUY' if sig['signal']==1 else '🔴 SELL' if sig['signal']==-1 else '⏸️ WAIT'} ({sig['confidence']}%)")
        
        # Update trailing stop
        if self.position and self.best_price:
            price = sig['price']
            atr = sig['atr']
            if self.position['side'] == "LONG" and price > self.best_price:
                self.best_price = price
                new_stop = self.best_price - 1.2 * atr
                if new_stop > self.position['stop']:
                    print(f"📈 Trail: ${self.position['stop']:.2f} → ${new_stop:.2f}")
                    self.position['stop'] = new_stop
            elif self.position['side'] == "SHORT" and price < self.best_price:
                self.best_price = price
                new_stop = self.best_price + 1.2 * atr
                if new_stop < self.position['stop']:
                    print(f"📉 Trail: ${self.position['stop']:.2f} → ${new_stop:.2f}")
                    self.position['stop'] = new_stop
        
        # Execute
        self._execute(sig)
        self._print_status(sig['price'])
    
    def _execute(self, sig):
        price, atr, signal, conf = sig['price'], sig['atr'], sig['signal'], sig['confidence']
        
        # Check stops/targets first
        if self.position:
            side = self.position['side']
            hit_stop = (side == "LONG" and price <= self.position['stop']) or \
                       (side == "SHORT" and price >= self.position['stop'])
            hit_target = (side == "LONG" and price >= self.position['target']) or \
                         (side == "SHORT" and price <= self.position['target'])
            should_reverse = (side == "LONG" and signal == -1 and conf >= 65) or \
                            (side == "SHORT" and signal == 1 and conf >= 65)
            
            if hit_stop:
                self._close("STOP", price)
            elif hit_target:
                self._close("TARGET", price)
            elif should_reverse:
                self._close("REVERSE", price)
        
        # Open new position
        if self.position is None and signal != 0 and conf >= self.min_confidence:
            size = (self.capital * self.position_size) / price
            
            if signal == 1:
                stop = price - self.atr_sl_mult * atr
                target = price + self.atr_tp_mult * atr
                self.position = {"side": "LONG", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"🟢 OPEN LONG: {size:.4f} @ ${price:.2f}")
                print(f"   SL: ${stop:.2f} ({(stop/price-1)*100:.1f}%) | TP: ${target:.2f} ({(target/price-1)*100:.1f}%)")
            else:
                stop = price + self.atr_sl_mult * atr
                target = price - self.atr_tp_mult * atr
                self.position = {"side": "SHORT", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"🔴 OPEN SHORT: {size:.4f} @ ${price:.2f}")
                print(f"   SL: ${stop:.2f} ({(stop/price-1)*100:.1f}%) | TP: ${target:.2f} ({(target/price-1)*100:.1f}%)")
    
    def _close(self, reason, price):
        if self.position['side'] == "LONG":
            pnl = (price - self.position['entry']) * self.position['size']
        else:
            pnl = (self.position['entry'] - price) * self.position['size']
        
        pnl -= price * self.position['size'] * 0.0008  # Commission
        self.capital += pnl
        self.trades.append({"pnl": pnl, "reason": reason, "side": self.position['side']})
        
        emoji = "✅" if pnl > 0 else "❌"
        pct = (pnl / self.initial_capital) * 100
        print(f"{emoji} CLOSE {self.position['side']} ({reason}): ${pnl:+.2f} ({pct:+.2f}%)")
        
        self.position = None
        self.best_price = None
    
    def _print_status(self, price):
        pnl_pct = (self.capital / self.initial_capital - 1) * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = len(self.trades)
        
        print(f"\n💼 ${self.capital:.2f} ({pnl_pct:+.2f}%)")
        if total > 0:
            print(f"📊 Trades: {total} | Wins: {wins} ({wins/total*100:.0f}%) | Total PnL: ${sum(t['pnl'] for t in self.trades):.2f}")
        
        if self.position:
            if self.position['side'] == "LONG":
                unreal = (price - self.position['entry']) * self.position['size']
            else:
                unreal = (self.position['entry'] - price) * self.position['size']
            pct = (unreal / self.initial_capital) * 100
            print(f"📍 {self.position['side']} @ ${self.position['entry']:.2f} | P&L: ${unreal:+.2f} ({pct:+.2f}%)")
            print(f"   SL: ${self.position['stop']:.2f} | TP: ${self.position['target']:.2f}")
    
    def run(self, interval=30):
        print(f"⚡ AGGRESSIVE TRADER | {self.symbol} {self.timeframe}")
        print(f"Min Confidence: {self.min_confidence}% | Position: {self.position_size*100:.0f}%")
        print("Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    print(f"⚠️ {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("🛑 FINAL REPORT")
            print(f"{'='*60}")
            print(f"Capital: ${self.initial_capital:.2f} → ${self.capital:.2f}")
            print(f"Return: {(self.capital/self.initial_capital-1)*100:+.2f}%")
            if self.trades:
                wins = [t for t in self.trades if t['pnl'] > 0]
                losses = [t for t in self.trades if t['pnl'] <= 0]
                print(f"Trades: {len(self.trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
                if wins:
                    print(f"Avg Win: ${sum(t['pnl'] for t in wins)/len(wins):.2f}")
                if losses:
                    print(f"Avg Loss: ${sum(t['pnl'] for t in losses)/len(losses):.2f}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    
    trader = AggressiveTrader(symbol=symbol, capital=10000, timeframe=tf)
    trader.run(interval=30)
