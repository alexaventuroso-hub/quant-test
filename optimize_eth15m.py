#!/usr/bin/env python3
"""
OPTIMIZE ETHUSDT 15m
Goal: Higher returns with low drawdown
Method: Improve avg win/loss ratio, test many parameters
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
from itertools import product

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
        time.sleep(0.05)
    
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
    
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-10)
    
    # Momentum
    df['mom'] = df['close'].pct_change(5) * 100
    
    return df

def get_signals(df, i, zscore_thresh, rsi_low, rsi_high, require_turn, require_vol):
    """Configurable signal generator"""
    row = df.iloc[i]
    prev = df.iloc[i-1]
    
    # Only trade in RANGE (ADX < 20)
    if row['adx'] >= 20:
        return 0
    
    # Volume filter
    if require_vol and row['vol_ratio'] < 1.0:
        return 0
    
    # LONG signal
    if row['zscore'] < -zscore_thresh and row['bb_pct'] < 0.15 and row['rsi'] < rsi_low:
        if not require_turn or row['rsi'] > prev['rsi']:
            return 1
    
    # SHORT signal
    if row['zscore'] > zscore_thresh and row['bb_pct'] > 0.85 and row['rsi'] > rsi_high:
        if not require_turn or row['rsi'] < prev['rsi']:
            return -1
    
    return 0

def backtest(df, params):
    """Backtest with configurable parameters"""
    zscore_thresh = params['zscore']
    rsi_low = params['rsi_low']
    rsi_high = params['rsi_high']
    atr_sl = params['atr_sl']
    atr_tp = params['atr_tp']
    pos_size = params['pos_size']
    require_turn = params['require_turn']
    require_vol = params['require_vol']
    use_trailing = params.get('trailing', False)
    
    capital = 10000
    position = None
    trades = []
    equity = [capital]
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        price, atr = row['close'], row['atr']
        
        signal = get_signals(df, i, zscore_thresh, rsi_low, rsi_high, require_turn, require_vol)
        
        # Position management
        if position:
            side = position['side']
            
            # Trailing stop update
            if use_trailing:
                if side == 'LONG' and price > position['best']:
                    position['best'] = price
                    position['stop'] = max(position['stop'], price - atr_sl * atr)
                elif side == 'SHORT' and price < position['best']:
                    position['best'] = price
                    position['stop'] = min(position['stop'], price + atr_sl * atr)
            
            # Exit conditions
            if side == 'LONG':
                if price <= position['stop'] or price >= position['target'] or signal == -1:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': side, 'entry': position['entry'], 'exit': price})
                    position = None
            else:
                if price >= position['stop'] or price <= position['target'] or signal == 1:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': side, 'entry': position['entry'], 'exit': price})
                    position = None
        
        # Open new position
        if position is None and signal != 0:
            size = (capital * pos_size) / price
            if signal == 1:
                position = {'side': 'LONG', 'entry': price, 'size': size,
                           'stop': price - atr_sl * atr, 'target': price + atr_tp * atr,
                           'best': price}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size,
                           'stop': price + atr_sl * atr, 'target': price - atr_tp * atr,
                           'best': price}
        
        equity.append(capital)
    
    # Close open position
    if position:
        price = df['close'].iloc[-1]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'side': position['side'], 'entry': position['entry'], 'exit': price})
    
    if not trades:
        return None
    
    # Calculate metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    ret = (capital / 10000 - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    pf = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses else 0
    
    peak, max_dd = equity[0], 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    
    return {
        'ret': ret,
        'win_rate': win_rate,
        'trades': len(trades),
        'max_dd': max_dd,
        'pf': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'expectancy': expectancy,
        'params': params
    }

def main():
    print("="*70)
    print("OPTIMIZING ETHUSDT 15m")
    print("Goal: High returns, low drawdown, good win/loss ratio")
    print("="*70)
    
    print("\nFetching data...")
    df = fetch_data("ETHUSDT", "15m", 180)
    df = calc_indicators(df)
    print(f"Got {len(df)} candles")
    
    # Parameter grid
    param_grid = {
        'zscore': [1.0, 1.5, 2.0],
        'rsi_low': [25, 30, 35],
        'rsi_high': [65, 70, 75],
        'atr_sl': [1.0, 1.5, 2.0],
        'atr_tp': [2.0, 3.0, 4.0, 5.0, 6.0],
        'pos_size': [0.15, 0.20, 0.25, 0.30],
        'require_turn': [True, False],
        'require_vol': [True, False],
        'trailing': [True, False],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    print(f"\nTesting {len(combinations)} parameter combinations...")
    
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        result = backtest(df, params)
        if result and result['trades'] >= 10:  # Minimum trades
            results.append(result)
        
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(combinations)} tested...")
    
    print(f"\nValid results: {len(results)}")
    
    # Sort by different criteria
    print("\n" + "="*70)
    print("TOP 10 BY RETURN")
    print("="*70)
    by_ret = sorted(results, key=lambda x: x['ret'], reverse=True)[:10]
    for i, r in enumerate(by_ret, 1):
        print(f"{i:2}. Ret:{r['ret']:>+6.2f}% | Win:{r['win_rate']:>5.1f}% | DD:{r['max_dd']:>5.1f}% | PF:{r['pf']:.2f} | W/L:{r['win_loss_ratio']:.2f} | Trades:{r['trades']}")
        print(f"    Params: Z={r['params']['zscore']} RSI={r['params']['rsi_low']}/{r['params']['rsi_high']} SL={r['params']['atr_sl']} TP={r['params']['atr_tp']} Size={r['params']['pos_size']:.0%}")
    
    print("\n" + "="*70)
    print("TOP 10 BY PROFIT FACTOR")
    print("="*70)
    by_pf = sorted([r for r in results if r['pf'] > 0], key=lambda x: x['pf'], reverse=True)[:10]
    for i, r in enumerate(by_pf, 1):
        print(f"{i:2}. PF:{r['pf']:>5.2f} | Ret:{r['ret']:>+6.2f}% | Win:{r['win_rate']:>5.1f}% | DD:{r['max_dd']:>5.1f}% | W/L:{r['win_loss_ratio']:.2f}")
        print(f"    Params: Z={r['params']['zscore']} RSI={r['params']['rsi_low']}/{r['params']['rsi_high']} SL={r['params']['atr_sl']} TP={r['params']['atr_tp']}")
    
    print("\n" + "="*70)
    print("TOP 10 BY WIN/LOSS RATIO (min 40% win rate)")
    print("="*70)
    by_wl = sorted([r for r in results if r['win_rate'] >= 40], key=lambda x: x['win_loss_ratio'], reverse=True)[:10]
    for i, r in enumerate(by_wl, 1):
        print(f"{i:2}. W/L:{r['win_loss_ratio']:>5.2f} | Ret:{r['ret']:>+6.2f}% | Win:{r['win_rate']:>5.1f}% | DD:{r['max_dd']:>5.1f}% | AvgWin:${r['avg_win']:.2f} AvgLoss:${r['avg_loss']:.2f}")
    
    print("\n" + "="*70)
    print("TOP 10 BY RISK-ADJUSTED (Return/MaxDD, min 5% return)")
    print("="*70)
    by_risk = sorted([r for r in results if r['ret'] > 5 and r['max_dd'] > 0], key=lambda x: x['ret']/x['max_dd'], reverse=True)[:10]
    if by_risk:
        for i, r in enumerate(by_risk, 1):
            ratio = r['ret']/r['max_dd']
            print(f"{i:2}. Ratio:{ratio:>5.2f}x | Ret:{r['ret']:>+6.2f}% | DD:{r['max_dd']:>5.1f}% | Win:{r['win_rate']:>5.1f}%")
            print(f"    Params: Z={r['params']['zscore']} RSI={r['params']['rsi_low']}/{r['params']['rsi_high']} SL={r['params']['atr_sl']} TP={r['params']['atr_tp']} Size={r['params']['pos_size']:.0%} Trail={r['params']['trailing']}")
    else:
        print("  No strategies with >5% return")
        by_risk = sorted([r for r in results if r['ret'] > 0 and r['max_dd'] > 0], key=lambda x: x['ret']/x['max_dd'], reverse=True)[:10]
        for i, r in enumerate(by_risk, 1):
            ratio = r['ret']/r['max_dd']
            print(f"{i:2}. Ratio:{ratio:>5.2f}x | Ret:{r['ret']:>+6.2f}% | DD:{r['max_dd']:>5.1f}%")
    
    print("\n" + "="*70)
    print("TOP 10 BY EXPECTANCY")
    print("="*70)
    by_exp = sorted(results, key=lambda x: x['expectancy'], reverse=True)[:10]
    for i, r in enumerate(by_exp, 1):
        print(f"{i:2}. Exp:${r['expectancy']:>6.2f} | Ret:{r['ret']:>+6.2f}% | Win:{r['win_rate']:>5.1f}% | Trades:{r['trades']}")
    
    # Find THE BEST overall
    print("\n" + "="*70)
    print("BEST OVERALL (Ret>3%, DD<10%, Win>40%, PF>1.2)")
    print("="*70)
    best = [r for r in results if r['ret'] > 3 and r['max_dd'] < 10 and r['win_rate'] > 40 and r['pf'] > 1.2]
    best = sorted(best, key=lambda x: x['ret'], reverse=True)[:10]
    if best:
        for i, r in enumerate(best, 1):
            print(f"\n{i}. RETURN: {r['ret']:+.2f}%")
            print(f"   Win Rate: {r['win_rate']:.1f}% | Max DD: {r['max_dd']:.1f}%")
            print(f"   Profit Factor: {r['pf']:.2f} | Win/Loss Ratio: {r['win_loss_ratio']:.2f}")
            print(f"   Avg Win: ${r['avg_win']:.2f} | Avg Loss: ${r['avg_loss']:.2f}")
            print(f"   Trades: {r['trades']} | Expectancy: ${r['expectancy']:.2f}/trade")
            print(f"   Parameters:")
            print(f"     Z-Score: {r['params']['zscore']} | RSI: {r['params']['rsi_low']}/{r['params']['rsi_high']}")
            print(f"     SL: {r['params']['atr_sl']}x ATR | TP: {r['params']['atr_tp']}x ATR")
            print(f"     Position Size: {r['params']['pos_size']:.0%}")
            print(f"     Require Turn: {r['params']['require_turn']} | Require Vol: {r['params']['require_vol']}")
            print(f"     Trailing Stop: {r['params']['trailing']}")
    else:
        print("No strategies met all criteria. Relaxing...")
        best = sorted([r for r in results if r['ret'] > 0], key=lambda x: x['ret'], reverse=True)[:5]
        for r in best:
            print(f"  Ret:{r['ret']:+.2f}% Win:{r['win_rate']:.1f}% DD:{r['max_dd']:.1f}% PF:{r['pf']:.2f}")


if __name__ == "__main__":
    main()
