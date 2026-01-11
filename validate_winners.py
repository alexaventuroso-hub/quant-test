#!/usr/bin/env python3
"""
VALIDATE ORIGINAL WINNERS
Test the strategies that actually showed profit on 180 days
Original winners: SOLUSDT 4h mean_reversion (+19.63%), ETHUSDT 4h macd_cross (+12.67%)
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_data(symbol, timeframe, days=180):
    print(f"Fetching {symbol} {timeframe}...", end=" ")
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    tf_min = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
    total = (days * 24 * 60) // tf_min.get(timeframe, 240)
    
    while len(all_data) < total:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": symbol, "interval": timeframe, "limit": 1500, "endTime": end_time}, timeout=10)
            data = r.json()
            if not data: break
            all_data = data + all_data
            end_time = data[0][0] - 1
            time.sleep(0.1)
        except: break
    
    print(f"{len(all_data)} candles")
    df = pd.DataFrame(all_data, columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
    for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def calc_indicators(df):
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
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
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
    plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).rolling(14).mean()
    
    return df

# STRATEGIES - exactly as in original backtest
def mean_reversion(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['zscore'].iloc[i] < -1.5 and df['bb_pct'].iloc[i] < 0.1 and df['rsi'].iloc[i] < 35:
            signals.iloc[i] = 1
        elif df['zscore'].iloc[i] > 1.5 and df['bb_pct'].iloc[i] > 0.9 and df['rsi'].iloc[i] > 65:
            signals.iloc[i] = -1
    return signals

def macd_cross(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['macd_hist'].iloc[i] > 0 and df['macd_hist'].iloc[i-1] <= 0:
            signals.iloc[i] = 1
        elif df['macd_hist'].iloc[i] < 0 and df['macd_hist'].iloc[i-1] >= 0:
            signals.iloc[i] = -1
    return signals

def trend_follow(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['ema8'].iloc[i] > df['ema21'].iloc[i] > df['ema50'].iloc[i]:
            if df['macd_hist'].iloc[i] > 0 and 50 < df['rsi'].iloc[i] < 70 and df['adx'].iloc[i] > 20:
                signals.iloc[i] = 1
        elif df['ema8'].iloc[i] < df['ema21'].iloc[i] < df['ema50'].iloc[i]:
            if df['macd_hist'].iloc[i] < 0 and 30 < df['rsi'].iloc[i] < 50 and df['adx'].iloc[i] > 20:
                signals.iloc[i] = -1
    return signals

def rsi_extreme(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['rsi'].iloc[i] < 30 and df['rsi'].iloc[i] > df['rsi'].iloc[i-1] and df['ema21'].iloc[i] > df['ema50'].iloc[i]:
            signals.iloc[i] = 1
        elif df['rsi'].iloc[i] > 70 and df['rsi'].iloc[i] < df['rsi'].iloc[i-1] and df['ema21'].iloc[i] < df['ema50'].iloc[i]:
            signals.iloc[i] = -1
    return signals

def ema_cross(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['ema8'].iloc[i] > df['ema21'].iloc[i] and df['ema8'].iloc[i-1] <= df['ema21'].iloc[i-1]:
            signals.iloc[i] = 1
        elif df['ema8'].iloc[i] < df['ema21'].iloc[i] and df['ema8'].iloc[i-1] >= df['ema21'].iloc[i-1]:
            signals.iloc[i] = -1
    return signals

def backtest(df, signals, capital=10000, commission=0.0004, atr_sl=2.0, atr_tp=3.0, pos_size=0.25):
    position = None
    trades = []
    equity = [capital]
    monthly = {}
    
    for i in range(50, len(df)):
        price, atr, sig = df['close'].iloc[i], df['atr'].iloc[i], signals.iloc[i]
        
        if position:
            side = position['side']
            if side == 'LONG':
                if price <= position['stop'] or price >= position['target'] or sig == -1:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': side, 'date': df.index[i]})
                    position = None
            else:
                if price >= position['stop'] or price <= position['target'] or sig == 1:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * commission * 2
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': side, 'date': df.index[i]})
                    position = None
        
        if position is None and sig != 0:
            size = (capital * pos_size) / price
            if sig == 1:
                position = {'side': 'LONG', 'entry': price, 'size': size, 'stop': price - atr_sl * atr, 'target': price + atr_tp * atr}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size, 'stop': price + atr_sl * atr, 'target': price - atr_tp * atr}
        
        equity.append(capital)
        month = df.index[i].strftime('%Y-%m')
        if month not in monthly: monthly[month] = {'start': capital}
        monthly[month]['end'] = capital
    
    if position:
        price = df['close'].iloc[-1]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'side': position['side'], 'date': df.index[-1]})
    
    return trades, equity, capital, monthly

def analyze(name, trades, equity, final, monthly, initial=10000):
    if not trades:
        print(f"  {name}: No trades")
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    ret = (final / initial - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    
    peak, max_dd = equity[0], 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 6)) if returns.std() > 0 else 0  # 4h = 6 per day
    
    gross_win = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']
    long_pnl = sum(t['pnl'] for t in long_trades)
    short_pnl = sum(t['pnl'] for t in short_trades)
    
    emoji = "+" if ret > 0 else "-"
    print(f"  {name:15} | {emoji}{abs(ret):>6.2f}% | Win:{win_rate:>5.1f}% | DD:{max_dd:>5.1f}% | Sharpe:{sharpe:>5.2f} | PF:{pf:>4.2f} | Trades:{len(trades):>3}")
    print(f"                   | LONG: ${long_pnl:>+8.2f} ({len(long_trades)} trades) | SHORT: ${short_pnl:>+8.2f} ({len(short_trades)} trades)")
    
    return {'name': name, 'ret': ret, 'win': win_rate, 'dd': max_dd, 'sharpe': sharpe, 'pf': pf, 'trades': len(trades)}

def main():
    print("="*80)
    print("VALIDATING ORIGINAL WINNERS")
    print("Testing strategies that showed profit in initial 180-day backtest")
    print("="*80)
    
    # Original winners were:
    # 1. SOLUSDT 4h mean_reversion: +19.63%
    # 2. SOLUSDT 4h macd_cross: +13.70%
    # 3. ETHUSDT 4h macd_cross: +12.67%
    # 4. ETHUSDT 4h mean_reversion: +9.32%
    # 5. BNBUSDT 4h macd_cross: +6.09%
    
    tests = [
        ('SOLUSDT', '4h'),
        ('ETHUSDT', '4h'),
        ('BNBUSDT', '4h'),
        ('BTCUSDT', '4h'),
    ]
    
    strategies = {
        'mean_reversion': mean_reversion,
        'macd_cross': macd_cross,
        'trend_follow': trend_follow,
        'rsi_extreme': rsi_extreme,
        'ema_cross': ema_cross,
    }
    
    all_results = []
    
    for symbol, tf in tests:
        print(f"\n{'='*80}")
        print(f"{symbol} {tf}")
        print(f"{'='*80}")
        
        df = fetch_data(symbol, tf, 180)
        if df is None or len(df) < 200:
            print("  Not enough data")
            continue
        
        df = calc_indicators(df)
        print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print()
        
        for strat_name, strat_func in strategies.items():
            signals = strat_func(df)
            trades, equity, final, monthly = backtest(df, signals)
            result = analyze(strat_name, trades, equity, final, monthly)
            if result:
                result['symbol'] = symbol
                result['tf'] = tf
                all_results.append(result)
        
        # Monthly breakdown for best strategy
        print(f"\n  Monthly returns (mean_reversion):")
        signals = mean_reversion(df)
        trades, equity, final, monthly = backtest(df, signals)
        for m, data in sorted(monthly.items()):
            ret = (data['end'] / data['start'] - 1) * 100
            print(f"    {m}: {'+' if ret > 0 else ''}{ret:.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("TOP 10 OVERALL (by return)")
    print("="*80)
    best = sorted(all_results, key=lambda x: x['ret'], reverse=True)[:10]
    for i, r in enumerate(best, 1):
        emoji = "+" if r['ret'] > 0 else "-"
        print(f"{i:2}. {r['symbol']:8} {r['tf']:4} {r['name']:15} | {emoji}{abs(r['ret']):>6.2f}% | Win:{r['win']:>5.1f}% | DD:{r['dd']:>5.1f}%")
    
    print("\n" + "="*80)
    print("PROFITABLE STRATEGIES (Return > 0)")
    print("="*80)
    profitable = [r for r in all_results if r['ret'] > 0]
    if profitable:
        for r in sorted(profitable, key=lambda x: x['ret'], reverse=True):
            print(f"  {r['symbol']:8} {r['tf']:4} {r['name']:15} | +{r['ret']:.2f}% | Win:{r['win']:.1f}% | DD:{r['dd']:.1f}% | Sharpe:{r['sharpe']:.2f}")
    else:
        print("  No profitable strategies found with current parameters")
        print("\n  Trying with different parameters...")
    
    # Try optimization on best pair
    print("\n" + "="*80)
    print("PARAMETER OPTIMIZATION - SOLUSDT 4h mean_reversion")
    print("="*80)
    
    df = fetch_data('SOLUSDT', '4h', 180)
    df = calc_indicators(df)
    
    best_result = None
    print("\nTesting parameter combinations...")
    
    for atr_sl in [1.5, 2.0, 2.5, 3.0]:
        for atr_tp in [2.0, 3.0, 4.0, 5.0]:
            for pos_size in [0.15, 0.20, 0.25, 0.30]:
                signals = mean_reversion(df)
                trades, equity, final, _ = backtest(df, signals, atr_sl=atr_sl, atr_tp=atr_tp, pos_size=pos_size)
                if trades:
                    ret = (final / 10000 - 1) * 100
                    wins = sum(1 for t in trades if t['pnl'] > 0)
                    win_rate = wins / len(trades) * 100
                    
                    peak, max_dd = equity[0], 0
                    for eq in equity:
                        if eq > peak: peak = eq
                        dd = (peak - eq) / peak * 100
                        if dd > max_dd: max_dd = dd
                    
                    if best_result is None or ret > best_result['ret']:
                        best_result = {'sl': atr_sl, 'tp': atr_tp, 'size': pos_size, 'ret': ret, 'win': win_rate, 'dd': max_dd, 'trades': len(trades)}
    
    if best_result:
        print(f"\nBest parameters found:")
        print(f"  SL: {best_result['sl']}x ATR")
        print(f"  TP: {best_result['tp']}x ATR")
        print(f"  Position Size: {best_result['size']:.0%}")
        print(f"  Return: {'+' if best_result['ret'] > 0 else ''}{best_result['ret']:.2f}%")
        print(f"  Win Rate: {best_result['win']:.1f}%")
        print(f"  Max DD: {best_result['dd']:.1f}%")
        print(f"  Trades: {best_result['trades']}")


if __name__ == "__main__":
    main()
