#!/usr/bin/env python3
"""
COMPREHENSIVE BACKTESTER
Test strategies across pairs/timeframes, find winners
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

class Strategy:
    """Base strategy with multiple variants"""
    
    @staticmethod
    def calc_indicators(df):
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
        
        # Bollinger
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
        plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        
        # Momentum
        df['mom3'] = df['close'].pct_change(3) * 100
        df['mom5'] = df['close'].pct_change(5) * 100
        
        return df
    
    @staticmethod
    def ema_cross(df):
        """EMA 8/21 crossover strategy"""
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['ema8'].iloc[i] > df['ema21'].iloc[i] and df['ema8'].iloc[i-1] <= df['ema21'].iloc[i-1]:
                signals.iloc[i] = 1
            elif df['ema8'].iloc[i] < df['ema21'].iloc[i] and df['ema8'].iloc[i-1] >= df['ema21'].iloc[i-1]:
                signals.iloc[i] = -1
        return signals
    
    @staticmethod
    def macd_cross(df):
        """MACD histogram crossover"""
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['macd_hist'].iloc[i] > 0 and df['macd_hist'].iloc[i-1] <= 0:
                signals.iloc[i] = 1
            elif df['macd_hist'].iloc[i] < 0 and df['macd_hist'].iloc[i-1] >= 0:
                signals.iloc[i] = -1
        return signals
    
    @staticmethod
    def rsi_extreme(df):
        """RSI oversold/overbought with trend"""
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            # Oversold bounce in uptrend
            if df['rsi'].iloc[i] < 30 and df['rsi'].iloc[i] > df['rsi'].iloc[i-1] and df['ema21'].iloc[i] > df['ema50'].iloc[i]:
                signals.iloc[i] = 1
            # Overbought drop in downtrend
            elif df['rsi'].iloc[i] > 70 and df['rsi'].iloc[i] < df['rsi'].iloc[i-1] and df['ema21'].iloc[i] < df['ema50'].iloc[i]:
                signals.iloc[i] = -1
        return signals
    
    @staticmethod
    def mean_reversion(df):
        """Mean reversion using z-score and Bollinger"""
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            # Oversold + z-score extreme
            if df['zscore'].iloc[i] < -1.5 and df['bb_pct'].iloc[i] < 0.1 and df['rsi'].iloc[i] < 35:
                signals.iloc[i] = 1
            # Overbought + z-score extreme
            elif df['zscore'].iloc[i] > 1.5 and df['bb_pct'].iloc[i] > 0.9 and df['rsi'].iloc[i] > 65:
                signals.iloc[i] = -1
        return signals
    
    @staticmethod
    def trend_follow(df):
        """Trend following - aligned EMAs + momentum"""
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            # Strong uptrend
            if df['ema8'].iloc[i] > df['ema21'].iloc[i] > df['ema50'].iloc[i]:
                if df['macd_hist'].iloc[i] > 0 and df['rsi'].iloc[i] > 50 and df['rsi'].iloc[i] < 70:
                    if df['adx'].iloc[i] > 20:
                        signals.iloc[i] = 1
            # Strong downtrend
            elif df['ema8'].iloc[i] < df['ema21'].iloc[i] < df['ema50'].iloc[i]:
                if df['macd_hist'].iloc[i] < 0 and df['rsi'].iloc[i] < 50 and df['rsi'].iloc[i] > 30:
                    if df['adx'].iloc[i] > 20:
                        signals.iloc[i] = -1
        return signals
    
    @staticmethod
    def combo(df):
        """Combined strategy - multiple confirmations"""
        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            long_score = 0
            short_score = 0
            
            # EMA trend
            if df['ema8'].iloc[i] > df['ema21'].iloc[i]: long_score += 1
            else: short_score += 1
            
            # MACD
            if df['macd_hist'].iloc[i] > 0: long_score += 1
            else: short_score += 1
            
            # RSI
            if df['rsi'].iloc[i] > 50 and df['rsi'].iloc[i] < 70: long_score += 1
            elif df['rsi'].iloc[i] < 50 and df['rsi'].iloc[i] > 30: short_score += 1
            
            # Price vs EMA
            if df['close'].iloc[i] > df['ema21'].iloc[i]: long_score += 1
            else: short_score += 1
            
            # Momentum
            if df['mom5'].iloc[i] > 0: long_score += 1
            else: short_score += 1
            
            if long_score >= 4: signals.iloc[i] = 1
            elif short_score >= 4: signals.iloc[i] = -1
        
        return signals


class Backtester:
    def __init__(self, capital=10000, commission=0.0004):
        self.initial_capital = capital
        self.commission = commission
    
    def run(self, df, signals, atr_sl=2.0, atr_tp=3.0, position_size=0.25):
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [capital]
        
        for i in range(50, len(df)):
            price = df['close'].iloc[i]
            atr = df['atr'].iloc[i]
            signal = signals.iloc[i]
            
            # Check existing position
            if position:
                # Stop loss
                if position['side'] == 'LONG' and price <= position['stop']:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * self.commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'type': 'STOP', 'side': 'LONG'})
                    position = None
                elif position['side'] == 'SHORT' and price >= position['stop']:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * self.commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'type': 'STOP', 'side': 'SHORT'})
                    position = None
                # Take profit
                elif position['side'] == 'LONG' and price >= position['target']:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * self.commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'type': 'TARGET', 'side': 'LONG'})
                    position = None
                elif position['side'] == 'SHORT' and price <= position['target']:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * self.commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'type': 'TARGET', 'side': 'SHORT'})
                    position = None
                # Reverse signal
                elif position['side'] == 'LONG' and signal == -1:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * self.commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'type': 'REVERSE', 'side': 'LONG'})
                    position = None
                elif position['side'] == 'SHORT' and signal == 1:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * self.commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'type': 'REVERSE', 'side': 'SHORT'})
                    position = None
            
            # Open new position
            if position is None and signal != 0:
                size = (capital * position_size) / price
                if signal == 1:
                    position = {
                        'side': 'LONG',
                        'entry': price,
                        'size': size,
                        'stop': price - atr_sl * atr,
                        'target': price + atr_tp * atr
                    }
                else:
                    position = {
                        'side': 'SHORT',
                        'entry': price,
                        'size': size,
                        'stop': price + atr_sl * atr,
                        'target': price - atr_tp * atr
                    }
            
            equity_curve.append(capital)
        
        # Close any open position at end
        if position:
            price = df['close'].iloc[-1]
            if position['side'] == 'LONG':
                pnl = (price - position['entry']) * position['size']
            else:
                pnl = (position['entry'] - price) * position['size']
            capital += pnl
            trades.append({'pnl': pnl, 'type': 'END', 'side': position['side']})
        
        return self._calc_metrics(trades, equity_curve, capital)
    
    def _calc_metrics(self, trades, equity_curve, final_capital):
        if not trades:
            return {'return': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 0, 'sharpe': 0, 'profit_factor': 0}
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        total_return = (final_capital / self.initial_capital - 1) * 100
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe (simplified)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        return {
            'return': total_return,
            'trades': len(trades),
            'win_rate': win_rate,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'wins': len(wins),
            'losses': len(losses)
        }


def fetch_data(symbol, timeframe, days=180):
    """Fetch historical data from Binance"""
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    
    # Calculate how many candles we need
    tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    minutes = tf_minutes.get(timeframe, 60)
    total_candles = (days * 24 * 60) // minutes
    
    while len(all_data) < total_candles:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": symbol, "interval": timeframe, "limit": 1500, "endTime": end_time},
                timeout=10)
            data = r.json()
            if not data:
                break
            all_data = data + all_data
            end_time = data[0][0] - 1
            time.sleep(0.1)
        except:
            break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data, columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df


def main():
    print("="*80)
    print("COMPREHENSIVE STRATEGY BACKTEST")
    print("="*80)
    
    # Config
    PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    TIMEFRAMES = ['15m', '1h', '4h']
    STRATEGIES = {
        'ema_cross': Strategy.ema_cross,
        'macd_cross': Strategy.macd_cross,
        'rsi_extreme': Strategy.rsi_extreme,
        'mean_reversion': Strategy.mean_reversion,
        'trend_follow': Strategy.trend_follow,
        'combo': Strategy.combo,
    }
    
    results = []
    bt = Backtester(capital=10000, commission=0.0004)
    
    for pair in PAIRS:
        print(f"\n📊 {pair}")
        for tf in TIMEFRAMES:
            print(f"  Fetching {tf}...", end=" ", flush=True)
            df = fetch_data(pair, tf, days=180)
            if df is None or len(df) < 100:
                print("❌ No data")
                continue
            print(f"✓ {len(df)} candles")
            
            df = Strategy.calc_indicators(df)
            
            for strat_name, strat_func in STRATEGIES.items():
                signals = strat_func(df)
                metrics = bt.run(df, signals)
                
                results.append({
                    'pair': pair,
                    'tf': tf,
                    'strategy': strat_name,
                    **metrics
                })
                
                emoji = "✅" if metrics['return'] > 0 else "❌"
                print(f"    {emoji} {strat_name:15} | Ret: {metrics['return']:>7.2f}% | Win: {metrics['win_rate']:>5.1f}% | DD: {metrics['max_dd']:>5.1f}% | Trades: {metrics['trades']:>4} | PF: {metrics['profit_factor']:.2f}")
    
    # Sort and show best results
    print("\n" + "="*80)
    print("🏆 TOP 15 BY RETURN")
    print("="*80)
    sorted_by_return = sorted(results, key=lambda x: x['return'], reverse=True)[:15]
    for i, r in enumerate(sorted_by_return, 1):
        print(f"{i:2}. {r['pair']:8} {r['tf']:4} {r['strategy']:15} | Ret: {r['return']:>7.2f}% | Win: {r['win_rate']:>5.1f}% | DD: {r['max_dd']:>5.1f}% | PF: {r['profit_factor']:.2f}")
    
    print("\n" + "="*80)
    print("🎯 TOP 15 BY SHARPE (Risk-Adjusted)")
    print("="*80)
    sorted_by_sharpe = sorted([r for r in results if r['trades'] > 10], key=lambda x: x['sharpe'], reverse=True)[:15]
    for i, r in enumerate(sorted_by_sharpe, 1):
        print(f"{i:2}. {r['pair']:8} {r['tf']:4} {r['strategy']:15} | Sharpe: {r['sharpe']:>5.2f} | Ret: {r['return']:>7.2f}% | Win: {r['win_rate']:>5.1f}%")
    
    print("\n" + "="*80)
    print("💰 TOP 15 BY PROFIT FACTOR")
    print("="*80)
    sorted_by_pf = sorted([r for r in results if r['trades'] > 10], key=lambda x: x['profit_factor'], reverse=True)[:15]
    for i, r in enumerate(sorted_by_pf, 1):
        print(f"{i:2}. {r['pair']:8} {r['tf']:4} {r['strategy']:15} | PF: {r['profit_factor']:>5.2f} | Ret: {r['return']:>7.2f}% | Win: {r['win_rate']:>5.1f}%")
    
    print("\n" + "="*80)
    print("⚖️ BEST RISK/REWARD (Return > 5%, DD < 10%, Win Rate > 40%)")
    print("="*80)
    good_results = [r for r in results if r['return'] > 5 and r['max_dd'] < 10 and r['win_rate'] > 40 and r['trades'] > 10]
    good_results = sorted(good_results, key=lambda x: x['return'], reverse=True)[:15]
    for i, r in enumerate(good_results, 1):
        print(f"{i:2}. {r['pair']:8} {r['tf']:4} {r['strategy']:15} | Ret: {r['return']:>7.2f}% | DD: {r['max_dd']:>5.1f}% | Win: {r['win_rate']:>5.1f}%")
    
    if not good_results:
        print("   No strategies met all criteria. Relaxing constraints...")
        ok_results = sorted([r for r in results if r['return'] > 0 and r['trades'] > 10], key=lambda x: x['return'], reverse=True)[:10]
        for i, r in enumerate(ok_results, 1):
            print(f"{i:2}. {r['pair']:8} {r['tf']:4} {r['strategy']:15} | Ret: {r['return']:>7.2f}% | DD: {r['max_dd']:>5.1f}% | Win: {r['win_rate']:>5.1f}%")

if __name__ == "__main__":
    main()
