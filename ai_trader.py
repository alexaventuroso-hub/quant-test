#!/usr/bin/env python3
"""
AI TRADER - Let AI think and decide
Full context, detailed analysis, AI makes the call
"""
import os, sys, time, requests, pandas as pd
from datetime import datetime, timezone

class AITrader:
    def __init__(self, symbol="ETHUSDT", capital=10000, timeframe="15m"):
        self.symbol = symbol
        self.capital = capital
        self.initial_capital = capital
        self.timeframe = timeframe
        self.position = None
        self.trades = []
        self.best_price = None
        self.api_key = os.getenv("GROQ_API_KEY")
        self.last_decisions = []  # Track AI reasoning
        
    def fetch_data(self):
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines", 
                params={"symbol": self.symbol, "interval": self.timeframe, "limit": 100}, timeout=10)
            df = pd.DataFrame(r.json(), columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except Exception as e:
            print(f"Fetch error: {e}")
            return None
    
    def calc_indicators(self, df):
        df = df.copy()
        # EMAs
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
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
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # Price changes
        df['change_1'] = df['close'].pct_change(1) * 100
        df['change_3'] = df['close'].pct_change(3) * 100
        df['change_5'] = df['close'].pct_change(5) * 100
        
        # Highs/Lows
        df['high_10'] = df['high'].rolling(10).max()
        df['low_10'] = df['low'].rolling(10).min()
        
        return df
    
    def get_ai_decision(self, df):
        if not self.api_key:
            return {"action": "HOLD", "confidence": 50, "reasoning": "No API key"}
        
        # Get last 10 candles for context
        recent = df.tail(10)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Build candle history
        candle_history = ""
        for i, (idx, row) in enumerate(recent.iterrows()):
            direction = "GREEN" if row['close'] > row['open'] else "RED"
            body_pct = abs(row['close'] - row['open']) / row['open'] * 100
            candle_history += f"  {i+1}. {direction} {body_pct:.2f}% | Close: ${row['close']:.2f} | RSI: {row['rsi']:.0f}\n"
        
        # Current position info
        position_info = "No position"
        if self.position:
            side = self.position['side']
            entry = self.position['entry']
            pnl_pct = ((current['close'] / entry) - 1) * 100 if side == "LONG" else ((entry / current['close']) - 1) * 100
            position_info = f"{side} from ${entry:.2f} | Current P&L: {pnl_pct:+.2f}%"
        
        # Recent trades performance
        trade_info = "No trades yet"
        if self.trades:
            wins = sum(1 for t in self.trades if t['pnl'] > 0)
            trade_info = f"{len(self.trades)} trades, {wins} wins ({wins/len(self.trades)*100:.0f}%)"
        
        prompt = f"""You are an expert crypto trader. Analyze this market data and decide whether to BUY, SELL, or HOLD.

=== MARKET: {self.symbol} {self.timeframe} ===

CURRENT PRICE: ${current['close']:.2f}

TREND ANALYSIS:
- EMA8: ${current['ema8']:.2f} {'> EMA21' if current['ema8'] > current['ema21'] else '< EMA21'}
- EMA21: ${current['ema21']:.2f} {'> EMA50' if current['ema21'] > current['ema50'] else '< EMA50'}
- Price vs EMA21: {((current['close']/current['ema21'])-1)*100:+.2f}%
- Short-term trend: {'BULLISH' if current['ema8'] > current['ema21'] else 'BEARISH'}
- Medium-term trend: {'BULLISH' if current['ema21'] > current['ema50'] else 'BEARISH'}

MOMENTUM:
- RSI(14): {current['rsi']:.1f} {'OVERBOUGHT!' if current['rsi']>70 else 'OVERSOLD!' if current['rsi']<30 else ''}
- MACD Histogram: {current['macd_hist']:.4f} {'BULLISH' if current['macd_hist']>0 else 'BEARISH'}
- MACD vs previous: {'INCREASING' if current['macd_hist'] > prev['macd_hist'] else 'DECREASING'}

VOLATILITY & RANGE:
- ATR: ${current['atr']:.2f} ({current['atr']/current['close']*100:.2f}% of price)
- Bollinger %B: {current['bb_pct']:.2f} (0=lower band, 0.5=middle, 1=upper band)
- 10-bar High: ${current['high_10']:.2f} | 10-bar Low: ${current['low_10']:.2f}

PRICE ACTION (last 10 candles):
{candle_history}
RECENT CHANGES:
- Last 1 bar: {current['change_1']:+.2f}%
- Last 3 bars: {current['change_3']:+.2f}%
- Last 5 bars: {current['change_5']:+.2f}%

VOLUME: {current['vol_ratio']:.1f}x average {'HIGH VOLUME!' if current['vol_ratio']>1.5 else ''}

=== CURRENT STATUS ===
Position: {position_info}
Trade history: {trade_info}
Capital: ${self.capital:.2f}

=== YOUR TASK ===
Think step by step:
1. What is the overall trend?
2. Is momentum supporting a move?
3. Are we at a good entry point (support/resistance)?
4. What's the risk/reward?
5. Should we act now or wait?

Then give your decision in this EXACT format:
ACTION: BUY or SELL or HOLD
CONFIDENCE: 50-95 (how sure are you)
REASONING: One sentence explaining why

Be decisive! If there's a clear setup, take it. Don't always say HOLD."""

        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.4
                },
                timeout=20)
            
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                
                # Parse response
                action = "HOLD"
                if "ACTION: BUY" in text.upper() or "ACTION:BUY" in text.upper():
                    action = "BUY"
                elif "ACTION: SELL" in text.upper() or "ACTION:SELL" in text.upper():
                    action = "SELL"
                
                import re
                conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)', text.upper())
                confidence = int(conf_match.group(1)) if conf_match else 60
                confidence = min(95, max(50, confidence))
                
                reason_match = re.search(r'REASONING[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
                reasoning = reason_match.group(1).strip()[:80] if reason_match else "AI decision"
                
                return {"action": action, "confidence": confidence, "reasoning": reasoning, "full_response": text}
                
        except Exception as e:
            print(f"AI Error: {e}")
        
        return {"action": "HOLD", "confidence": 50, "reasoning": "API error"}
    
    def run_cycle(self):
        now = datetime.now(timezone.utc)
        df = self.fetch_data()
        if df is None or len(df) < 50:
            print("No data")
            return
        
        df = self.calc_indicators(df)
        current = df.iloc[-1]
        
        # Get AI decision
        decision = self.get_ai_decision(df)
        
        # Display
        trend = "UP" if current['ema8'] > current['ema21'] else "DOWN"
        print(f"\n{'='*60}")
        print(f"{now.strftime('%H:%M:%S')} | {self.symbol} {self.timeframe} | Trend: {trend}")
        print(f"{'='*60}")
        print(f"Price: ${current['close']:.2f} | RSI: {current['rsi']:.0f} | BB%: {current['bb_pct']:.2f}")
        print(f"MACD: {'+'if current['macd_hist']>0 else ''}{current['macd_hist']:.4f} | Vol: {current['vol_ratio']:.1f}x")
        print(f"\n🤖 AI DECISION: {decision['action']} ({decision['confidence']}%)")
        print(f"   Reason: {decision['reasoning']}")
        
        # Trailing stop update
        if self.position and self.best_price:
            price = current['close']
            atr = current['atr']
            if self.position['side'] == "LONG" and price > self.best_price:
                self.best_price = price
                new_stop = self.best_price - 1.5 * atr
                if new_stop > self.position['stop']:
                    self.position['stop'] = new_stop
                    print(f"   📈 Trail stop -> ${new_stop:.2f}")
            elif self.position['side'] == "SHORT" and price < self.best_price:
                self.best_price = price
                new_stop = self.best_price + 1.5 * atr
                if new_stop < self.position['stop']:
                    self.position['stop'] = new_stop
                    print(f"   📉 Trail stop -> ${new_stop:.2f}")
        
        # Execute based on AI decision
        self._execute(current['close'], current['atr'], decision)
        self._status(current['close'])
    
    def _execute(self, price, atr, decision):
        action = decision['action']
        conf = decision['confidence']
        
        # Check existing position first
        if self.position:
            side = self.position['side']
            
            # Stop loss hit
            if (side == "LONG" and price <= self.position['stop']) or \
               (side == "SHORT" and price >= self.position['stop']):
                self._close("STOP", price)
                return
            
            # Take profit hit
            if (side == "LONG" and price >= self.position['target']) or \
               (side == "SHORT" and price <= self.position['target']):
                self._close("TARGET", price)
                return
            
            # AI says reverse
            if (side == "LONG" and action == "SELL" and conf >= 65) or \
               (side == "SHORT" and action == "BUY" and conf >= 65):
                self._close("AI_REVERSE", price)
        
        # Open new position if no position and AI is confident
        if self.position is None and action != "HOLD" and conf >= 60:
            size = (self.capital * 0.25) / price
            
            if action == "BUY":
                stop = price - 2 * atr
                target = price + 3 * atr
                self.position = {"side": "LONG", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"\n   🟢 OPEN LONG {size:.4f} @ ${price:.2f}")
                print(f"      Stop: ${stop:.2f} | Target: ${target:.2f}")
            else:
                stop = price + 2 * atr
                target = price - 3 * atr
                self.position = {"side": "SHORT", "size": size, "entry": price, "stop": stop, "target": target}
                self.best_price = price
                print(f"\n   🔴 OPEN SHORT {size:.4f} @ ${price:.2f}")
                print(f"      Stop: ${stop:.2f} | Target: ${target:.2f}")
    
    def _close(self, reason, price):
        side = self.position['side']
        entry = self.position['entry']
        size = self.position['size']
        
        pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
        pnl -= price * size * 0.0008  # Commission
        
        self.capital += pnl
        self.trades.append({"pnl": pnl, "reason": reason, "side": side})
        
        emoji = "✅" if pnl > 0 else "❌"
        print(f"\n   {emoji} CLOSE {side} ({reason}): ${pnl:+.2f}")
        
        self.position = None
        self.best_price = None
    
    def _status(self, price):
        pct = (self.capital / self.initial_capital - 1) * 100
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total = len(self.trades)
        
        print(f"\n{'─'*60}")
        print(f"Capital: ${self.capital:.2f} ({pct:+.2f}%)", end="")
        if total > 0:
            print(f" | Trades: {total} | Wins: {wins} ({wins/total*100:.0f}%)")
        else:
            print()
        
        if self.position:
            side = self.position['side']
            entry = self.position['entry']
            u = (price - entry) * self.position['size'] if side == "LONG" else (entry - price) * self.position['size']
            pct_move = ((price / entry) - 1) * 100 if side == "LONG" else ((entry / price) - 1) * 100
            print(f"Position: {side} @ ${entry:.2f} -> ${price:.2f} ({pct_move:+.2f}%) | P&L: ${u:+.2f}")
            print(f"Stop: ${self.position['stop']:.2f} | Target: ${self.position['target']:.2f}")
    
    def run(self, interval=45):
        print(f"🤖 AI TRADER | {self.symbol} {self.timeframe}")
        print(f"AI makes decisions with full market context")
        print(f"Checking every {interval}s | Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("FINAL REPORT")
            print(f"{'='*60}")
            print(f"${self.initial_capital:.2f} -> ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
            if self.trades:
                wins = [t for t in self.trades if t['pnl'] > 0]
                losses = [t for t in self.trades if t['pnl'] <= 0]
                print(f"Total Trades: {len(self.trades)}")
                print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/len(self.trades)*100:.0f}%")
                if wins:
                    print(f"Avg Win: ${sum(t['pnl'] for t in wins)/len(wins):.2f}")
                if losses:
                    print(f"Avg Loss: ${sum(t['pnl'] for t in losses)/len(losses):.2f}")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    AITrader(symbol=symbol, timeframe=tf).run(45)
