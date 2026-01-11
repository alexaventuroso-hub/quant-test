#!/usr/bin/env python3
"""
RENAISSANCE-INSPIRED TRADING SYSTEM
Based on techniques from "The Man Who Solved the Market"

Core Principles:
1. Hidden Markov Models for regime detection
2. Kelly Criterion for position sizing
3. Mean reversion + momentum signals
4. 5-minute data intervals
5. AI as central decision maker
6. Transaction cost optimization
7. "50.75% right, 100% of the time" - law of large numbers
"""
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from scipy import stats

# ============================================================
# HIDDEN MARKOV MODEL - REGIME DETECTION
# ============================================================
class HMMRegimeDetector:
    """
    Simplified Hidden Markov Model for market regime detection
    States: BULL, BEAR, SIDEWAYS
    """
    def __init__(self):
        self.states = ['BULL', 'BEAR', 'SIDEWAYS']
        
    def detect_regime(self, df):
        """Detect current market regime using multiple signals"""
        if len(df) < 50:
            return 'SIDEWAYS', 0.5
        
        row = df.iloc[-1]
        lookback = df.tail(20)
        
        # Factors for regime detection
        trend_strength = 0
        
        # 1. EMA alignment
        if row['ema8'] > row['ema21'] > row['ema50']:
            trend_strength += 2
        elif row['ema8'] < row['ema21'] < row['ema50']:
            trend_strength -= 2
        
        # 2. ADX (trend strength)
        if row['adx'] > 25:
            trend_strength += 1 if row['plus_di'] > row['minus_di'] else -1
        
        # 3. Price momentum
        mom = (row['close'] / lookback['close'].iloc[0] - 1) * 100
        if mom > 3:
            trend_strength += 1
        elif mom < -3:
            trend_strength -= 1
        
        # 4. Volatility regime
        vol_expanding = row['atr'] > df['atr'].rolling(50).mean().iloc[-1]
        
        # Determine regime
        if trend_strength >= 2:
            regime = 'BULL'
            confidence = min(0.9, 0.5 + trend_strength * 0.1)
        elif trend_strength <= -2:
            regime = 'BEAR'
            confidence = min(0.9, 0.5 + abs(trend_strength) * 0.1)
        else:
            regime = 'SIDEWAYS'
            confidence = 0.6 if row['adx'] < 20 else 0.5
        
        return regime, confidence


# ============================================================
# KELLY CRITERION - POSITION SIZING
# ============================================================
class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing
    f* = (p * b - q) / b
    where p = win prob, q = lose prob, b = win/loss ratio
    """
    def __init__(self, max_kelly_fraction=0.25):
        self.max_fraction = max_kelly_fraction
        self.trade_history = []
    
    def add_trade(self, pnl):
        self.trade_history.append(pnl)
        # Keep last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def calculate_kelly(self, signal_confidence=0.5):
        """Calculate optimal position size"""
        if len(self.trade_history) < 10:
            # Default conservative sizing
            return 0.1
        
        wins = [t for t in self.trade_history if t > 0]
        losses = [t for t in self.trade_history if t <= 0]
        
        if not wins or not losses:
            return 0.1
        
        # Win probability adjusted by signal confidence
        base_win_prob = len(wins) / len(self.trade_history)
        p = base_win_prob * (0.5 + signal_confidence * 0.5)  # Scale by confidence
        q = 1 - p
        
        # Win/loss ratio
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        b = avg_win / avg_loss if avg_loss > 0 else 1
        
        # Kelly formula
        if b > 0:
            kelly = (p * b - q) / b
        else:
            kelly = 0
        
        # Apply half-Kelly for safety and cap at max
        kelly = max(0, min(kelly * 0.5, self.max_fraction))
        
        return kelly


# ============================================================
# TRANSACTION COST MODEL
# ============================================================
class TransactionCostModel:
    """
    Model transaction costs to avoid unprofitable trades
    Renaissance's secret weapon was superior cost estimation
    """
    def __init__(self, base_commission=0.0004, slippage_factor=0.0002):
        self.base_commission = base_commission
        self.slippage_factor = slippage_factor
    
    def estimate_cost(self, price, size, volatility_pct):
        """Estimate total transaction cost including slippage"""
        # Commission (both sides)
        commission = price * size * self.base_commission * 2
        
        # Slippage increases with volatility
        slippage = price * size * self.slippage_factor * (1 + volatility_pct)
        
        return commission + slippage
    
    def is_trade_profitable(self, expected_move_pct, price, size, volatility_pct):
        """Check if expected move covers transaction costs"""
        cost = self.estimate_cost(price, size, volatility_pct)
        expected_profit = price * size * (expected_move_pct / 100)
        
        # Need at least 1.5x cost coverage
        return expected_profit > cost * 1.5


# ============================================================
# SIGNAL GENERATORS (Multiple Strategies)
# ============================================================
class SignalGenerator:
    """Generate trading signals from multiple strategies"""
    
    @staticmethod
    def mean_reversion_signal(df):
        """Mean reversion using z-score and Bollinger"""
        row = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal, strength = 0, 0
        
        # Long: Oversold
        if row['zscore'] < -1.5 and row['bb_pct'] < 0.15 and row['rsi'] < 35:
            if row['rsi'] > prev['rsi']:  # Turning up
                signal = 1
                strength = min(abs(row['zscore']) / 3, 1) * 0.8
        
        # Short: Overbought
        elif row['zscore'] > 1.5 and row['bb_pct'] > 0.85 and row['rsi'] > 65:
            if row['rsi'] < prev['rsi']:  # Turning down
                signal = -1
                strength = min(abs(row['zscore']) / 3, 1) * 0.8
        
        return signal, strength
    
    @staticmethod
    def momentum_signal(df):
        """Momentum/trend following signal"""
        row = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal, strength = 0, 0
        
        # MACD crossover
        if row['macd_hist'] > 0 and prev['macd_hist'] <= 0:
            if row['ema8'] > row['ema21']:  # Trend alignment
                signal = 1
                strength = 0.7
        elif row['macd_hist'] < 0 and prev['macd_hist'] >= 0:
            if row['ema8'] < row['ema21']:
                signal = -1
                strength = 0.7
        
        return signal, strength
    
    @staticmethod
    def volatility_breakout_signal(df):
        """Volatility breakout signal"""
        row = df.iloc[-1]
        
        signal, strength = 0, 0
        
        # Bollinger Band breakout with volume
        if row['close'] > row['bb_upper'] and row['vol_ratio'] > 1.5:
            signal = 1
            strength = 0.6
        elif row['close'] < row['bb_lower'] and row['vol_ratio'] > 1.5:
            signal = -1
            strength = 0.6
        
        return signal, strength
    
    @staticmethod
    def support_resistance_signal(df):
        """Support/resistance bounce signal"""
        row = df.iloc[-1]
        
        signal, strength = 0, 0
        
        # Near support in uptrend
        if row['range_position'] < 0.2 and row['ema21'] > row['ema50']:
            if row['rsi'] < 40:
                signal = 1
                strength = 0.65
        
        # Near resistance in downtrend
        elif row['range_position'] > 0.8 and row['ema21'] < row['ema50']:
            if row['rsi'] > 60:
                signal = -1
                strength = 0.65
        
        return signal, strength


# ============================================================
# AI DECISION MAKER
# ============================================================
class AIDecisionMaker:
    """
    AI as the central brain - combines all signals
    Uses Groq API for final decision
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
    
    def make_decision(self, market_data, signals, regime, kelly_size):
        """AI makes final trading decision"""
        if not self.api_key:
            return self._rule_based_decision(signals, regime)
        
        prompt = f"""You are a Renaissance Technologies-style quant trader. Make a trading decision.

MARKET STATE:
- Symbol: {market_data['symbol']}
- Price: ${market_data['price']:.2f}
- Regime: {regime['state']} (confidence: {regime['confidence']:.0%})

TECHNICAL SIGNALS:
- Mean Reversion: {'LONG' if signals['mean_rev'][0]==1 else 'SHORT' if signals['mean_rev'][0]==-1 else 'NEUTRAL'} (strength: {signals['mean_rev'][1]:.0%})
- Momentum: {'LONG' if signals['momentum'][0]==1 else 'SHORT' if signals['momentum'][0]==-1 else 'NEUTRAL'} (strength: {signals['momentum'][1]:.0%})
- Volatility Breakout: {'LONG' if signals['vol_break'][0]==1 else 'SHORT' if signals['vol_break'][0]==-1 else 'NEUTRAL'} (strength: {signals['vol_break'][1]:.0%})
- Support/Resistance: {'LONG' if signals['sr'][0]==1 else 'SHORT' if signals['sr'][0]==-1 else 'NEUTRAL'} (strength: {signals['sr'][1]:.0%})

INDICATORS:
- RSI: {market_data['rsi']:.0f}
- MACD: {'Bullish' if market_data['macd']>0 else 'Bearish'}
- ADX: {market_data['adx']:.0f}
- Z-Score: {market_data['zscore']:.2f}

POSITION SIZING:
- Kelly suggests: {kelly_size:.1%} of capital

DECISION RULES:
1. In SIDEWAYS regime: prefer mean reversion
2. In BULL/BEAR regime: prefer momentum with trend
3. Multiple agreeing signals = higher confidence
4. Need expected move > 1.5x transaction cost

Reply EXACTLY:
ACTION: BUY or SELL or WAIT
CONFIDENCE: 50-95
SIZE: {kelly_size:.1%} or smaller
REASON: brief explanation"""

        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.3
                },
                timeout=15)
            
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                return self._parse_ai_response(text, kelly_size)
        except Exception as e:
            pass
        
        return self._rule_based_decision(signals, regime)
    
    def _parse_ai_response(self, text, kelly_size):
        import re
        
        action = "WAIT"
        if "ACTION: BUY" in text.upper(): action = "BUY"
        elif "ACTION: SELL" in text.upper(): action = "SELL"
        
        conf_match = re.search(r'CONFIDENCE[:\s]*(\d+)', text.upper())
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5
        
        size_match = re.search(r'SIZE[:\s]*([\d.]+)%', text)
        size = float(size_match.group(1)) / 100 if size_match else kelly_size
        
        reason_match = re.search(r'REASON[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
        reason = reason_match.group(1).strip()[:60] if reason_match else "AI decision"
        
        return {
            'action': action,
            'confidence': confidence,
            'size': min(size, kelly_size),
            'reason': reason
        }
    
    def _rule_based_decision(self, signals, regime):
        """Fallback rule-based decision"""
        # Count agreeing signals
        long_score = sum(1 for s in signals.values() if s[0] == 1)
        short_score = sum(1 for s in signals.values() if s[0] == -1)
        avg_strength = np.mean([s[1] for s in signals.values() if s[0] != 0]) if any(s[0] != 0 for s in signals.values()) else 0
        
        action = "WAIT"
        if long_score >= 2 and short_score == 0:
            if regime['state'] != 'BEAR':
                action = "BUY"
        elif short_score >= 2 and long_score == 0:
            if regime['state'] != 'BULL':
                action = "SELL"
        
        return {
            'action': action,
            'confidence': avg_strength if action != "WAIT" else 0,
            'size': 0.15,
            'reason': f"{long_score} long, {short_score} short signals"
        }


# ============================================================
# DATA & INDICATORS
# ============================================================
def fetch_data(symbol, timeframe, limit=500):
    """Fetch data from Binance Futures"""
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": timeframe, "limit": limit},
            timeout=10)
        data = r.json()
        if not data or isinstance(data, dict):
            return None
        
        df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        return df
    except:
        return None


def calc_indicators(df):
    """Calculate all indicators"""
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
    ema12, ema26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Z-score
    df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
    
    # ADX
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-10)
    
    # Support/Resistance
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    df['range_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'] + 1e-10)
    
    return df


# ============================================================
# BACKTESTER
# ============================================================
class RenaissanceBacktester:
    """Backtest the Renaissance-style system"""
    
    def __init__(self, capital=10000):
        self.initial_capital = capital
        self.hmm = HMMRegimeDetector()
        self.kelly = KellyCriterion(max_kelly_fraction=0.25)
        self.cost_model = TransactionCostModel()
        self.ai = AIDecisionMaker()
    
    def run(self, df, use_ai=False):
        capital = self.initial_capital
        position = None
        trades = []
        equity = [capital]
        
        for i in range(50, len(df)):
            current_df = df.iloc[:i+1]
            row = df.iloc[i]
            price = row['close']
            atr = row['atr']
            
            # 1. Detect regime
            regime_state, regime_conf = self.hmm.detect_regime(current_df)
            regime = {'state': regime_state, 'confidence': regime_conf}
            
            # 2. Generate signals
            signals = {
                'mean_rev': SignalGenerator.mean_reversion_signal(current_df),
                'momentum': SignalGenerator.momentum_signal(current_df),
                'vol_break': SignalGenerator.volatility_breakout_signal(current_df),
                'sr': SignalGenerator.support_resistance_signal(current_df)
            }
            
            # 3. Calculate Kelly size
            avg_signal_strength = np.mean([s[1] for s in signals.values() if s[1] > 0]) if any(s[1] > 0 for s in signals.values()) else 0.5
            kelly_size = self.kelly.calculate_kelly(avg_signal_strength)
            
            # 4. Check transaction costs
            vol_pct = row['atr_pct']
            expected_move = 2 * vol_pct  # Expect 2x ATR move
            
            # 5. Make decision
            if use_ai:
                market_data = {
                    'symbol': 'TEST',
                    'price': price,
                    'rsi': row['rsi'],
                    'macd': row['macd_hist'],
                    'adx': row['adx'],
                    'zscore': row['zscore']
                }
                decision = self.ai.make_decision(market_data, signals, regime, kelly_size)
            else:
                decision = self.ai._rule_based_decision(signals, regime)
            
            # 6. Check position
            if position:
                # Trailing stop / exit logic
                if position['side'] == 'LONG':
                    if price <= position['stop'] or price >= position['target']:
                        pnl = (price - position['entry']) * position['size']
                        pnl -= self.cost_model.estimate_cost(price, position['size'], vol_pct)
                        capital += pnl
                        trades.append({'pnl': pnl})
                        self.kelly.add_trade(pnl)
                        position = None
                    elif decision['action'] == 'SELL' and decision['confidence'] > 0.6:
                        pnl = (price - position['entry']) * position['size']
                        pnl -= self.cost_model.estimate_cost(price, position['size'], vol_pct)
                        capital += pnl
                        trades.append({'pnl': pnl})
                        self.kelly.add_trade(pnl)
                        position = None
                else:  # SHORT
                    if price >= position['stop'] or price <= position['target']:
                        pnl = (position['entry'] - price) * position['size']
                        pnl -= self.cost_model.estimate_cost(price, position['size'], vol_pct)
                        capital += pnl
                        trades.append({'pnl': pnl})
                        self.kelly.add_trade(pnl)
                        position = None
                    elif decision['action'] == 'BUY' and decision['confidence'] > 0.6:
                        pnl = (position['entry'] - price) * position['size']
                        pnl -= self.cost_model.estimate_cost(price, position['size'], vol_pct)
                        capital += pnl
                        trades.append({'pnl': pnl})
                        self.kelly.add_trade(pnl)
                        position = None
            
            # 7. Open new position
            if position is None and decision['action'] != 'WAIT':
                if decision['confidence'] > 0.55:
                    # Check if trade is profitable after costs
                    size = (capital * decision['size']) / price
                    if self.cost_model.is_trade_profitable(expected_move, price, size, vol_pct):
                        if decision['action'] == 'BUY':
                            position = {
                                'side': 'LONG',
                                'entry': price,
                                'size': size,
                                'stop': price - 2 * atr,
                                'target': price + 3 * atr
                            }
                        else:
                            position = {
                                'side': 'SHORT',
                                'entry': price,
                                'size': size,
                                'stop': price + 2 * atr,
                                'target': price - 3 * atr
                            }
            
            equity.append(capital)
        
        # Close open position
        if position:
            price = df['close'].iloc[-1]
            if position['side'] == 'LONG':
                pnl = (price - position['entry']) * position['size']
            else:
                pnl = (position['entry'] - price) * position['size']
            capital += pnl
            trades.append({'pnl': pnl})
        
        return self._calc_metrics(trades, equity, capital)
    
    def _calc_metrics(self, trades, equity, final_capital):
        if not trades:
            return {'ret': 0, 'trades': 0, 'win': 0, 'dd': 0, 'sharpe': 0, 'pf': 0}
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        ret = (final_capital / self.initial_capital - 1) * 100
        win_rate = len(wins) / len(trades) * 100
        
        # Max drawdown
        peak, max_dd = equity[0], 0
        for eq in equity:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd: max_dd = dd
        
        # Sharpe ratio
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() > 0 else 0
        
        # Profit factor
        gross_win = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        
        return {
            'ret': ret,
            'trades': len(trades),
            'win': win_rate,
            'dd': max_dd,
            'sharpe': sharpe,
            'pf': pf,
            'wins': len(wins),
            'losses': len(losses)
        }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("RENAISSANCE-INSPIRED TRADING SYSTEM")
    print("Based on 'The Man Who Solved the Market'")
    print("=" * 70)
    print("\nCore Components:")
    print("  - Hidden Markov Model for regime detection")
    print("  - Kelly Criterion position sizing")
    print("  - Multiple signal generators")
    print("  - Transaction cost optimization")
    print("  - AI decision maker")
    print()
    
    PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    TIMEFRAMES = ['5m', '15m', '1h', '4h']
    
    results = []
    bt = RenaissanceBacktester()
    
    for pair in PAIRS:
        print(f"\n{pair}")
        for tf in TIMEFRAMES:
            print(f"  {tf}...", end=" ", flush=True)
            
            # Fetch more data for lower TFs
            days = 30 if tf == '5m' else 90 if tf == '15m' else 180
            limit = min(days * 24 * 60 // {'5m': 5, '15m': 15, '1h': 60, '4h': 240}[tf], 1000)
            
            df = fetch_data(pair, tf, limit)
            if df is None or len(df) < 100:
                print("X")
                continue
            
            df = calc_indicators(df)
            metrics = bt.run(df, use_ai=False)  # Set True for AI
            
            results.append({'pair': pair, 'tf': tf, **metrics})
            
            e = "+" if metrics['ret'] > 0 else "-"
            print(f"{len(df)} bars | {e} Ret: {metrics['ret']:>6.2f}% | Win: {metrics['win']:>5.1f}% | DD: {metrics['dd']:>5.1f}% | Sharpe: {metrics['sharpe']:.2f}")
    
    # Results
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS")
    print("=" * 70)
    best = sorted(results, key=lambda x: x['ret'], reverse=True)[:10]
    for i, r in enumerate(best, 1):
        print(f"{i:2}. {r['pair']:8} {r['tf']:4} | Ret: {r['ret']:>7.2f}% | Win: {r['win']:>5.1f}% | DD: {r['dd']:>5.1f}% | Sharpe: {r['sharpe']:.2f}")
    
    print("\n" + "=" * 70)
    print("BEST RISK-ADJUSTED (Sharpe > 0.5)")
    print("=" * 70)
    best_sharpe = sorted([r for r in results if r['sharpe'] > 0.5], key=lambda x: x['sharpe'], reverse=True)[:10]
    if best_sharpe:
        for i, r in enumerate(best_sharpe, 1):
            print(f"{i:2}. {r['pair']:8} {r['tf']:4} | Sharpe: {r['sharpe']:.2f} | Ret: {r['ret']:>7.2f}%")
    else:
        print("   No strategies with Sharpe > 0.5")
        print("   Best available:")
        for r in sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]:
            print(f"   {r['pair']:8} {r['tf']:4} | Sharpe: {r['sharpe']:.2f} | Ret: {r['ret']:>7.2f}%")


if __name__ == "__main__":
    main()
