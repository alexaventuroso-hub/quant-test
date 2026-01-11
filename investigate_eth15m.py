#!/usr/bin/env python3
"""
INVESTIGATE ETHUSDT 15m
- What made the short period work?
- Can we identify the conditions?
- Apply Simons' regime detection
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_data(symbol, timeframe, days):
    print(f"Fetching {days} days of {symbol} {timeframe}...")
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
    
    print(f"  Got {len(all_data)} candles")
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
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['zscore'] = (df['close'] - df['bb_mid']) / (df['bb_std'] + 1e-10)
    
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
    
    # Regime detection
    df['regime'] = 'NEUTRAL'
    df.loc[(df['adx'] > 25) & (df['plus_di'] > df['minus_di']), 'regime'] = 'TREND_UP'
    df.loc[(df['adx'] > 25) & (df['plus_di'] < df['minus_di']), 'regime'] = 'TREND_DOWN'
    df.loc[df['adx'] < 20, 'regime'] = 'RANGE'
    
    # Volatility regime
    df['vol_sma'] = df['atr_pct'].rolling(50).mean()
    df['vol_regime'] = 'NORMAL'
    df.loc[df['atr_pct'] > df['vol_sma'] * 1.5, 'vol_regime'] = 'HIGH_VOL'
    df.loc[df['atr_pct'] < df['vol_sma'] * 0.7, 'vol_regime'] = 'LOW_VOL'
    
    return df

def renaissance_signals(df, i):
    """Renaissance-style signals with regime awareness"""
    row = df.iloc[i]
    prev = df.iloc[i-1]
    regime = row['regime']
    
    signals = {}
    
    # Mean reversion (best in RANGE)
    sig, strength = 0, 0
    if row['zscore'] < -1.5 and row['bb_pct'] < 0.15 and row['rsi'] < 35:
        if row['rsi'] > prev['rsi']:
            sig, strength = 1, 0.8
    elif row['zscore'] > 1.5 and row['bb_pct'] > 0.85 and row['rsi'] > 65:
        if row['rsi'] < prev['rsi']:
            sig, strength = -1, 0.8
    signals['mean_rev'] = (sig, strength)
    
    # Momentum (best in TREND)
    sig, strength = 0, 0
    if row['macd_hist'] > 0 and prev['macd_hist'] <= 0 and row['ema8'] > row['ema21']:
        sig, strength = 1, 0.7
    elif row['macd_hist'] < 0 and prev['macd_hist'] >= 0 and row['ema8'] < row['ema21']:
        sig, strength = -1, 0.7
    signals['momentum'] = (sig, strength)
    
    # Breakout
    sig, strength = 0, 0
    if row['close'] > row['bb_upper']: sig, strength = 1, 0.6
    elif row['close'] < row['bb_lower']: sig, strength = -1, 0.6
    signals['breakout'] = (sig, strength)
    
    # SR bounce
    sig, strength = 0, 0
    range_pos = (row['close'] - df['low'].rolling(20).min().iloc[i]) / (df['high'].rolling(20).max().iloc[i] - df['low'].rolling(20).min().iloc[i] + 1e-10)
    if range_pos < 0.2 and row['rsi'] < 40 and row['ema21'] > row['ema50']:
        sig, strength = 1, 0.65
    elif range_pos > 0.8 and row['rsi'] > 60 and row['ema21'] < row['ema50']:
        sig, strength = -1, 0.65
    signals['sr'] = (sig, strength)
    
    return signals, regime

def backtest_period(df, start_idx, end_idx, strategy='renaissance', require_regime=None):
    """Backtest a specific period"""
    capital = 10000
    position = None
    trades = []
    
    for i in range(max(50, start_idx), min(len(df), end_idx)):
        row = df.iloc[i]
        price, atr = row['close'], row['atr']
        regime = row['regime']
        
        signals, _ = renaissance_signals(df, i)
        
        # Count agreeing signals
        long_count = sum(1 for s in signals.values() if s[0] == 1)
        short_count = sum(1 for s in signals.values() if s[0] == -1)
        
        # Decision with optional regime filter
        action = None
        if require_regime and regime not in require_regime:
            action = None  # Skip if not in required regime
        elif long_count >= 2 and short_count == 0:
            if regime != 'TREND_DOWN':
                action = 'BUY'
        elif short_count >= 2 and long_count == 0:
            if regime != 'TREND_UP':
                action = 'SELL'
        
        # Position management
        if position:
            side = position['side']
            if side == 'LONG':
                if price <= position['stop'] or price >= position['target'] or action == 'SELL':
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'date': df.index[i], 'regime': regime, 'side': side})
                    position = None
            else:
                if price >= position['stop'] or price <= position['target'] or action == 'BUY':
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'date': df.index[i], 'regime': regime, 'side': side})
                    position = None
        
        if position is None and action:
            size = (capital * 0.15) / price
            if action == 'BUY':
                position = {'side': 'LONG', 'entry': price, 'size': size, 
                           'stop': price - 2 * atr, 'target': price + 3 * atr}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size,
                           'stop': price + 2 * atr, 'target': price - 3 * atr}
    
    # Close open position
    if position:
        price = df['close'].iloc[min(end_idx-1, len(df)-1)]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'date': df.index[min(end_idx-1, len(df)-1)], 'regime': 'END', 'side': position['side']})
    
    return trades, capital

def analyze_trades_by_regime(trades):
    """Analyze which regimes produce winning trades"""
    if not trades:
        return {}
    
    regime_stats = {}
    for t in trades:
        regime = t.get('regime', 'UNKNOWN')
        if regime not in regime_stats:
            regime_stats[regime] = {'wins': 0, 'losses': 0, 'pnl': 0}
        
        if t['pnl'] > 0:
            regime_stats[regime]['wins'] += 1
        else:
            regime_stats[regime]['losses'] += 1
        regime_stats[regime]['pnl'] += t['pnl']
    
    return regime_stats

def main():
    print("="*70)
    print("INVESTIGATING ETHUSDT 15m")
    print("Why did short period work but long period fail?")
    print("="*70)
    
    # Fetch data
    df = fetch_data("ETHUSDT", "15m", 180)
    df = calc_indicators(df)
    
    print(f"\nData: {df.index[0]} to {df.index[-1]}")
    print(f"Total candles: {len(df)}")
    
    # 1. Analyze regime distribution
    print("\n" + "="*70)
    print("REGIME DISTRIBUTION (Full 180 days)")
    print("="*70)
    regime_counts = df['regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} candles ({count/len(df)*100:.1f}%)")
    
    vol_counts = df['vol_regime'].value_counts()
    print("\nVolatility regimes:")
    for regime, count in vol_counts.items():
        print(f"  {regime}: {count} candles ({count/len(df)*100:.1f}%)")
    
    # 2. Test last 10 days vs full period
    print("\n" + "="*70)
    print("COMPARING PERIODS")
    print("="*70)
    
    # Last 10 days (~960 candles)
    last_10_days = len(df) - 960
    trades_recent, capital_recent = backtest_period(df, last_10_days, len(df))
    wins_recent = sum(1 for t in trades_recent if t['pnl'] > 0)
    
    print(f"\nLAST 10 DAYS ({df.index[last_10_days].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}):")
    print(f"  Trades: {len(trades_recent)}")
    print(f"  Wins: {wins_recent} ({wins_recent/len(trades_recent)*100:.1f}%)" if trades_recent else "  No trades")
    print(f"  Return: {(capital_recent/10000-1)*100:+.2f}%")
    
    # Full period
    trades_full, capital_full = backtest_period(df, 0, len(df))
    wins_full = sum(1 for t in trades_full if t['pnl'] > 0)
    
    print(f"\nFULL 180 DAYS:")
    print(f"  Trades: {len(trades_full)}")
    print(f"  Wins: {wins_full} ({wins_full/len(trades_full)*100:.1f}%)" if trades_full else "  No trades")
    print(f"  Return: {(capital_full/10000-1)*100:+.2f}%")
    
    # 3. Analyze by regime
    print("\n" + "="*70)
    print("PERFORMANCE BY REGIME")
    print("="*70)
    
    regime_stats = analyze_trades_by_regime(trades_full)
    for regime, stats in sorted(regime_stats.items()):
        total = stats['wins'] + stats['losses']
        win_rate = stats['wins'] / total * 100 if total > 0 else 0
        print(f"  {regime:12}: {total:>3} trades | {win_rate:>5.1f}% win | ${stats['pnl']:>+8.2f}")
    
    # 4. Test regime-filtered strategies
    print("\n" + "="*70)
    print("REGIME-FILTERED BACKTESTS")
    print("="*70)
    
    # Only trade in RANGE
    trades_range, capital_range = backtest_period(df, 0, len(df), require_regime=['RANGE'])
    wins_range = sum(1 for t in trades_range if t['pnl'] > 0) if trades_range else 0
    print(f"\nRANGE ONLY:")
    print(f"  Trades: {len(trades_range)}")
    if trades_range:
        print(f"  Wins: {wins_range} ({wins_range/len(trades_range)*100:.1f}%)")
        print(f"  Return: {(capital_range/10000-1)*100:+.2f}%")
    
    # Only trade in TREND_UP
    trades_up, capital_up = backtest_period(df, 0, len(df), require_regime=['TREND_UP'])
    wins_up = sum(1 for t in trades_up if t['pnl'] > 0) if trades_up else 0
    print(f"\nTREND_UP ONLY:")
    print(f"  Trades: {len(trades_up)}")
    if trades_up:
        print(f"  Wins: {wins_up} ({wins_up/len(trades_up)*100:.1f}%)")
        print(f"  Return: {(capital_up/10000-1)*100:+.2f}%")
    
    # Only trade in TREND_DOWN
    trades_down, capital_down = backtest_period(df, 0, len(df), require_regime=['TREND_DOWN'])
    wins_down = sum(1 for t in trades_down if t['pnl'] > 0) if trades_down else 0
    print(f"\nTREND_DOWN ONLY:")
    print(f"  Trades: {len(trades_down)}")
    if trades_down:
        print(f"  Wins: {wins_down} ({wins_down/len(trades_down)*100:.1f}%)")
        print(f"  Return: {(capital_down/10000-1)*100:+.2f}%")
    
    # 5. Sliding window analysis - find GOOD periods
    print("\n" + "="*70)
    print("SLIDING WINDOW ANALYSIS (7-day windows)")
    print("="*70)
    
    window_size = 7 * 24 * 4  # 7 days in 15m candles
    best_windows = []
    
    for start in range(50, len(df) - window_size, window_size // 2):
        end = start + window_size
        trades_w, capital_w = backtest_period(df, start, end)
        if trades_w:
            ret = (capital_w / 10000 - 1) * 100
            wins = sum(1 for t in trades_w if t['pnl'] > 0)
            win_rate = wins / len(trades_w) * 100
            
            # Get regime distribution in this window
            window_df = df.iloc[start:end]
            regime_dist = window_df['regime'].value_counts(normalize=True) * 100
            
            best_windows.append({
                'start': df.index[start],
                'end': df.index[end-1],
                'ret': ret,
                'win_rate': win_rate,
                'trades': len(trades_w),
                'regime_dist': regime_dist.to_dict()
            })
    
    # Sort by return
    best_windows = sorted(best_windows, key=lambda x: x['ret'], reverse=True)
    
    print("\nTOP 5 BEST 7-DAY PERIODS:")
    for w in best_windows[:5]:
        print(f"\n  {w['start'].strftime('%Y-%m-%d')} to {w['end'].strftime('%Y-%m-%d')}")
        print(f"    Return: {w['ret']:+.2f}% | Win Rate: {w['win_rate']:.1f}% | Trades: {w['trades']}")
        print(f"    Regimes: ", end="")
        for r, pct in w['regime_dist'].items():
            print(f"{r}:{pct:.0f}% ", end="")
        print()
    
    print("\nWORST 5 PERIODS:")
    for w in best_windows[-5:]:
        print(f"\n  {w['start'].strftime('%Y-%m-%d')} to {w['end'].strftime('%Y-%m-%d')}")
        print(f"    Return: {w['ret']:+.2f}% | Win Rate: {w['win_rate']:.1f}% | Trades: {w['trades']}")
        print(f"    Regimes: ", end="")
        for r, pct in w['regime_dist'].items():
            print(f"{r}:{pct:.0f}% ", end="")
        print()
    
    # 6. KEY INSIGHT - what conditions work?
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Average regime distribution in winning vs losing periods
    winning_periods = [w for w in best_windows if w['ret'] > 1]
    losing_periods = [w for w in best_windows if w['ret'] < -1]
    
    if winning_periods:
        print("\nWINNING PERIODS characteristics:")
        avg_range = np.mean([w['regime_dist'].get('RANGE', 0) for w in winning_periods])
        avg_trend_up = np.mean([w['regime_dist'].get('TREND_UP', 0) for w in winning_periods])
        avg_trend_down = np.mean([w['regime_dist'].get('TREND_DOWN', 0) for w in winning_periods])
        print(f"  Avg RANGE: {avg_range:.1f}%")
        print(f"  Avg TREND_UP: {avg_trend_up:.1f}%")
        print(f"  Avg TREND_DOWN: {avg_trend_down:.1f}%")
    
    if losing_periods:
        print("\nLOSING PERIODS characteristics:")
        avg_range = np.mean([w['regime_dist'].get('RANGE', 0) for w in losing_periods])
        avg_trend_up = np.mean([w['regime_dist'].get('TREND_UP', 0) for w in losing_periods])
        avg_trend_down = np.mean([w['regime_dist'].get('TREND_DOWN', 0) for w in losing_periods])
        print(f"  Avg RANGE: {avg_range:.1f}%")
        print(f"  Avg TREND_UP: {avg_trend_up:.1f}%")
        print(f"  Avg TREND_DOWN: {avg_trend_down:.1f}%")


if __name__ == "__main__":
    main()
