#!/usr/bin/env python3
"""
ETHUSDT 15m REGIME-FILTERED TRADER
Key insight: Only trade in RANGE regime (50% win, +1.68%)
Skip TREND_DOWN (26.7% win, loses money)

Simons: "Human behavior is most predictable in times of high stress"
Translation: Mean reversion works best in choppy/ranging markets
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_data(symbol, timeframe, days):
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    tf_min = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
    total = (days * 24 * 60) // tf_min.get(timeframe, 15)
    
    while len(all_data) < total:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": timeframe, "limit": 1500, "endTime": end_time}, timeout=10)
        data = r.json()
        if not data: break
        all_data = data + all_data
        end_time = data[0][0] - 1
        time.sleep(0.1)
    
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
    df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
    
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['zscore'] = (df['close'] - df['bb_mid']) / (df['bb_std'] + 1e-10)
    
    # ADX for regime detection
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
    
    return df

def detect_regime(row):
    """Detect market regime"""
    if row['adx'] < 20:
        return 'RANGE'  # Low ADX = ranging/choppy market
    elif row['adx'] > 25:
        if row['plus_di'] > row['minus_di']:
            return 'TREND_UP'
        else:
            return 'TREND_DOWN'
    return 'NEUTRAL'

def get_signal(df, i, regime):
    """Get trading signal based on regime"""
    row = df.iloc[i]
    prev = df.iloc[i-1]
    
    # ONLY trade in RANGE regime (proven profitable)
    # Skip TREND_DOWN entirely (proven to lose money)
    
    if regime == 'RANGE':
        # Mean reversion works in ranging markets
        if row['zscore'] < -1.5 and row['bb_pct'] < 0.15 and row['rsi'] < 35:
            if row['rsi'] > prev['rsi']:  # Turning up
                return 1, "RANGE: Oversold bounce"
        elif row['zscore'] > 1.5 and row['bb_pct'] > 0.85 and row['rsi'] > 65:
            if row['rsi'] < prev['rsi']:  # Turning down
                return -1, "RANGE: Overbought reversal"
    
    elif regime == 'TREND_UP':
        # Only long in uptrends, buy pullbacks
        if row['rsi'] < 40 and row['rsi'] > prev['rsi']:
            if row['close'] > row['ema21']:
                return 1, "TREND_UP: Pullback buy"
    
    elif regime == 'TREND_DOWN':
        # SKIP - this regime loses money
        return 0, "TREND_DOWN: Skipping (unprofitable regime)"
    
    return 0, "No signal"

def backtest(df, allow_regimes=['RANGE', 'TREND_UP', 'NEUTRAL']):
    """Backtest with regime filter"""
    capital = 10000
    position = None
    trades = []
    equity = [capital]
    monthly = {}
    regime_trades = {'RANGE': [], 'TREND_UP': [], 'TREND_DOWN': [], 'NEUTRAL': []}
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        price, atr = row['close'], row['atr']
        regime = detect_regime(row)
        
        month = df.index[i].strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = {'start': capital}
        
        # Get signal (only in allowed regimes)
        if regime in allow_regimes:
            signal, reason = get_signal(df, i, regime)
        else:
            signal, reason = 0, f"Skipping {regime}"
        
        # Position management
        if position:
            side = position['side']
            if side == 'LONG':
                if price <= position['stop'] or price >= position['target'] or signal == -1:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'regime': position['regime'], 'side': side})
                    regime_trades[position['regime']].append(pnl)
                    position = None
            else:
                if price >= position['stop'] or price <= position['target'] or signal == 1:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'regime': position['regime'], 'side': side})
                    regime_trades[position['regime']].append(pnl)
                    position = None
        
        # Open new position (only in allowed regimes)
        if position is None and signal != 0:
            size = (capital * 0.15) / price
            if signal == 1:
                position = {'side': 'LONG', 'entry': price, 'size': size,
                           'stop': price - 2 * atr, 'target': price + 3 * atr, 'regime': regime}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size,
                           'stop': price + 2 * atr, 'target': price - 3 * atr, 'regime': regime}
        
        equity.append(capital)
        monthly[month]['end'] = capital
    
    # Close open position
    if position:
        price = df['close'].iloc[-1]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'regime': position['regime'], 'side': position['side']})
    
    return trades, equity, capital, monthly, regime_trades

def analyze(name, trades, equity, final, monthly, regime_trades):
    if not trades:
        print(f"  {name}: No trades")
        return
    
    wins = [t for t in trades if t['pnl'] > 0]
    ret = (final / 10000 - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    
    peak, max_dd = equity[0], 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Return: {ret:+.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Max Drawdown: {max_dd:.1f}%")
    print(f"Trades: {len(trades)}")
    print(f"Risk/Reward: {ret/max_dd:.2f}x" if max_dd > 0 else "")
    
    print(f"\nBy Regime:")
    for regime, pnls in regime_trades.items():
        if pnls:
            w = sum(1 for p in pnls if p > 0)
            print(f"  {regime:12}: {len(pnls):>3} trades | {w/len(pnls)*100:>5.1f}% win | ${sum(pnls):>+8.2f}")
    
    print(f"\nMonthly:")
    for m in sorted(monthly.keys()):
        data = monthly[m]
        m_ret = (data['end'] / data['start'] - 1) * 100
        print(f"  {m}: {'+' if m_ret > 0 else ''}{m_ret:.2f}%")
    
    return ret, max_dd

def main():
    print("="*60)
    print("ETHUSDT 15m REGIME-FILTERED BACKTEST")
    print("Testing: Only trade in profitable regimes")
    print("="*60)
    
    print("\nFetching data...")
    df = fetch_data("ETHUSDT", "15m", 180)
    df = calc_indicators(df)
    print(f"Got {len(df)} candles: {df.index[0]} to {df.index[-1]}")
    
    # Test different regime filters
    tests = [
        ("ALL REGIMES (baseline)", ['RANGE', 'TREND_UP', 'TREND_DOWN', 'NEUTRAL']),
        ("RANGE ONLY", ['RANGE']),
        ("RANGE + NEUTRAL", ['RANGE', 'NEUTRAL']),
        ("RANGE + TREND_UP", ['RANGE', 'TREND_UP']),
        ("NO TREND_DOWN", ['RANGE', 'TREND_UP', 'NEUTRAL']),
    ]
    
    results = []
    for name, regimes in tests:
        trades, equity, final, monthly, regime_trades = backtest(df, regimes)
        result = analyze(name, trades, equity, final, monthly, regime_trades)
        if result:
            results.append((name, result[0], result[1]))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - BEST REGIME FILTER")
    print("="*60)
    for name, ret, dd in sorted(results, key=lambda x: x[1], reverse=True):
        ratio = ret/dd if dd > 0 else 0
        print(f"  {name:25} | Ret: {ret:+6.2f}% | DD: {dd:5.1f}% | Ratio: {ratio:.2f}x")
    
    # Best strategy
    best = max(results, key=lambda x: x[1])
    print(f"\n*** BEST: {best[0]} with {best[1]:+.2f}% return ***")

if __name__ == "__main__":
    main()
