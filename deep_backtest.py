#!/usr/bin/env python3
"""
DEEP BACKTEST - ETHUSDT 15m
Extended testing with full metrics
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

def fetch_data(symbol, timeframe, days=180):
    print(f"Fetching {symbol} {timeframe} for {days} days...")
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    tf_min = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
    total = (days * 24 * 60) // tf_min.get(timeframe, 15)
    
    while len(all_data) < total:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": symbol, "interval": timeframe, "limit": 1500, "endTime": end_time}, timeout=10)
            data = r.json()
            if not data: break
            all_data = data + all_data
            end_time = data[0][0] - 1
            print(f"  {len(all_data)} candles...", end="\r")
            time.sleep(0.1)
        except: break
    
    print(f"  Got {len(all_data)} candles total")
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

def detect_regime(df, i):
    row = df.iloc[i]
    strength = 0
    if row['ema8'] > row['ema21'] > row['ema50']: strength += 2
    elif row['ema8'] < row['ema21'] < row['ema50']: strength -= 2
    if row['adx'] > 25:
        strength += 1 if row['plus_di'] > row['minus_di'] else -1
    if strength >= 2: return 'BULL'
    elif strength <= -2: return 'BEAR'
    return 'SIDEWAYS'

def get_signals(df, i):
    row, prev = df.iloc[i], df.iloc[i-1]
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

def backtest(df, atr_sl=2.0, atr_tp=3.0, pos_size=0.15, min_signals=2):
    capital = 10000
    position = None
    trades = []
    equity = [capital]
    monthly_returns = {}
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        price, atr = row['close'], row['atr']
        regime = detect_regime(df, i)
        signals = get_signals(df, i)
        
        # Count signals
        long_count = sum(1 for s in signals.values() if s[0] == 1)
        short_count = sum(1 for s in signals.values() if s[0] == -1)
        
        # Decision
        action = None
        if long_count >= min_signals and short_count == 0 and regime != 'BEAR':
            action = 'BUY'
        elif short_count >= min_signals and long_count == 0 and regime != 'BULL':
            action = 'SELL'
        
        # Position management
        if position:
            side = position['side']
            if side == 'LONG':
                if price <= position['stop'] or price >= position['target'] or action == 'SELL':
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({
                        'pnl': pnl, 
                        'side': side, 
                        'entry': position['entry'], 
                        'exit': price,
                        'date': df.index[i],
                        'type': 'STOP' if price <= position['stop'] else 'TARGET' if price >= position['target'] else 'REVERSE'
                    })
                    position = None
            else:
                if price >= position['stop'] or price <= position['target'] or action == 'BUY':
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({
                        'pnl': pnl,
                        'side': side,
                        'entry': position['entry'],
                        'exit': price,
                        'date': df.index[i],
                        'type': 'STOP' if price >= position['stop'] else 'TARGET' if price <= position['target'] else 'REVERSE'
                    })
                    position = None
        
        # Open new position
        if position is None and action:
            size = (capital * pos_size) / price
            if action == 'BUY':
                position = {'side': 'LONG', 'entry': price, 'size': size, 
                           'stop': price - atr_sl * atr, 'target': price + atr_tp * atr}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size,
                           'stop': price + atr_sl * atr, 'target': price - atr_tp * atr}
        
        equity.append(capital)
        
        # Track monthly
        month_key = df.index[i].strftime('%Y-%m')
        if month_key not in monthly_returns:
            monthly_returns[month_key] = {'start': capital, 'end': capital}
        monthly_returns[month_key]['end'] = capital
    
    # Close open position
    if position:
        price = df['close'].iloc[-1]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'side': position['side'], 'entry': position['entry'], 'exit': price, 'date': df.index[-1], 'type': 'END'})
    
    return trades, equity, capital, monthly_returns

def analyze_results(trades, equity, final_capital, monthly_returns, initial=10000):
    print("\n" + "="*70)
    print("DETAILED BACKTEST RESULTS - ETHUSDT 15m")
    print("="*70)
    
    if not trades:
        print("No trades executed")
        return
    
    # Basic metrics
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    total_return = (final_capital / initial - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    
    # Drawdown
    peak, max_dd, max_dd_duration = equity[0], 0, 0
    dd_start = 0
    for i, eq in enumerate(equity):
        if eq > peak:
            peak = eq
            dd_start = i
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_duration = i - dd_start
    
    # Sharpe
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)) if returns.std() > 0 else 0  # 15m = 4 per hour
    
    # Profit factor
    gross_win = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    # Avg trade metrics
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    print(f"\n{'='*40}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*40}")
    print(f"Total Return:     {total_return:>10.2f}%")
    print(f"Final Capital:    ${final_capital:>10.2f}")
    print(f"Max Drawdown:     {max_dd:>10.2f}%")
    print(f"Sharpe Ratio:     {sharpe:>10.2f}")
    print(f"Profit Factor:    {pf:>10.2f}")
    
    print(f"\n{'='*40}")
    print("TRADE STATISTICS")
    print(f"{'='*40}")
    print(f"Total Trades:     {len(trades):>10}")
    print(f"Winning Trades:   {len(wins):>10} ({win_rate:.1f}%)")
    print(f"Losing Trades:    {len(losses):>10} ({100-win_rate:.1f}%)")
    print(f"Avg Win:          ${avg_win:>10.2f}")
    print(f"Avg Loss:         ${avg_loss:>10.2f}")
    print(f"Expectancy:       ${expectancy:>10.2f}")
    print(f"Win/Loss Ratio:   {abs(avg_win/avg_loss) if avg_loss else 0:>10.2f}")
    
    # By exit type
    print(f"\n{'='*40}")
    print("BY EXIT TYPE")
    print(f"{'='*40}")
    for exit_type in ['TARGET', 'STOP', 'REVERSE', 'END']:
        type_trades = [t for t in trades if t.get('type') == exit_type]
        if type_trades:
            type_wins = sum(1 for t in type_trades if t['pnl'] > 0)
            type_pnl = sum(t['pnl'] for t in type_trades)
            print(f"{exit_type:12} | {len(type_trades):>3} trades | {type_wins/len(type_trades)*100:>5.1f}% win | ${type_pnl:>8.2f}")
    
    # By side
    print(f"\n{'='*40}")
    print("BY SIDE")
    print(f"{'='*40}")
    for side in ['LONG', 'SHORT']:
        side_trades = [t for t in trades if t.get('side') == side]
        if side_trades:
            side_wins = sum(1 for t in side_trades if t['pnl'] > 0)
            side_pnl = sum(t['pnl'] for t in side_trades)
            print(f"{side:12} | {len(side_trades):>3} trades | {side_wins/len(side_trades)*100:>5.1f}% win | ${side_pnl:>8.2f}")
    
    # Monthly breakdown
    print(f"\n{'='*40}")
    print("MONTHLY RETURNS")
    print(f"{'='*40}")
    for month, data in sorted(monthly_returns.items()):
        ret = (data['end'] / data['start'] - 1) * 100
        emoji = "+" if ret > 0 else "-"
        print(f"{month}: {emoji}{abs(ret):>6.2f}%")
    
    # Best/Worst trades
    print(f"\n{'='*40}")
    print("NOTABLE TRADES")
    print(f"{'='*40}")
    sorted_trades = sorted(trades, key=lambda x: x['pnl'], reverse=True)
    print("Best 3:")
    for t in sorted_trades[:3]:
        print(f"  ${t['pnl']:>8.2f} | {t['side']} | {t.get('type', 'N/A')}")
    print("Worst 3:")
    for t in sorted_trades[-3:]:
        print(f"  ${t['pnl']:>8.2f} | {t['side']} | {t.get('type', 'N/A')}")
    
    # Risk metrics
    print(f"\n{'='*40}")
    print("RISK METRICS")
    print(f"{'='*40}")
    returns_list = [t['pnl']/initial*100 for t in trades]
    print(f"Std Dev (per trade):  {np.std(returns_list):>8.2f}%")
    print(f"Max Single Win:       ${max(t['pnl'] for t in trades):>8.2f}")
    print(f"Max Single Loss:      ${min(t['pnl'] for t in trades):>8.2f}")
    
    consecutive_wins, consecutive_losses = 0, 0
    max_cons_wins, max_cons_losses = 0, 0
    for t in trades:
        if t['pnl'] > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_cons_wins = max(max_cons_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_cons_losses = max(max_cons_losses, consecutive_losses)
    
    print(f"Max Consecutive Wins:  {max_cons_wins:>8}")
    print(f"Max Consecutive Losses:{max_cons_losses:>8}")

def optimize_parameters(df):
    print("\n" + "="*70)
    print("PARAMETER OPTIMIZATION")
    print("="*70)
    
    results = []
    
    for atr_sl in [1.5, 2.0, 2.5]:
        for atr_tp in [2.0, 2.5, 3.0, 4.0]:
            for pos_size in [0.10, 0.15, 0.20, 0.25]:
                for min_signals in [1, 2]:
                    trades, equity, final, _ = backtest(df, atr_sl, atr_tp, pos_size, min_signals)
                    if trades:
                        wins = sum(1 for t in trades if t['pnl'] > 0)
                        ret = (final / 10000 - 1) * 100
                        win_rate = wins / len(trades) * 100
                        
                        peak, max_dd = equity[0], 0
                        for eq in equity:
                            if eq > peak: peak = eq
                            dd = (peak - eq) / peak * 100
                            if dd > max_dd: max_dd = dd
                        
                        results.append({
                            'atr_sl': atr_sl, 'atr_tp': atr_tp, 'pos_size': pos_size, 
                            'min_sig': min_signals, 'ret': ret, 'win': win_rate, 
                            'dd': max_dd, 'trades': len(trades)
                        })
    
    # Sort by return
    results = sorted(results, key=lambda x: x['ret'], reverse=True)
    
    print("\nTOP 10 PARAMETER COMBINATIONS:")
    print("-"*90)
    print(f"{'SL':>5} {'TP':>5} {'Size':>6} {'MinSig':>6} {'Return':>8} {'Win%':>6} {'DD%':>6} {'Trades':>7}")
    print("-"*90)
    for r in results[:10]:
        print(f"{r['atr_sl']:>5.1f} {r['atr_tp']:>5.1f} {r['pos_size']:>6.0%} {r['min_sig']:>6} {r['ret']:>7.2f}% {r['win']:>5.1f}% {r['dd']:>5.1f}% {r['trades']:>7}")
    
    print("\nBEST RISK-ADJUSTED (Return/DD):")
    risk_adj = sorted([r for r in results if r['dd'] > 0], key=lambda x: x['ret']/x['dd'], reverse=True)
    for r in risk_adj[:5]:
        print(f"SL:{r['atr_sl']:.1f} TP:{r['atr_tp']:.1f} Size:{r['pos_size']:.0%} | Ret:{r['ret']:.2f}% DD:{r['dd']:.1f}% Ratio:{r['ret']/r['dd']:.2f}")
    
    return results[0] if results else None


def main():
    print("="*70)
    print("DEEP BACKTEST - ETHUSDT 15m Renaissance Strategy")
    print("="*70)
    
    # Fetch extended data
    df = fetch_data("ETHUSDT", "15m", days=180)
    if df is None or len(df) < 500:
        print("Not enough data")
        return
    
    df = calc_indicators(df)
    print(f"\nData range: {df.index[0]} to {df.index[-1]}")
    print(f"Total candles: {len(df)}")
    
    # Run backtest with default params
    print("\n" + "="*70)
    print("BACKTEST WITH DEFAULT PARAMETERS")
    print("="*70)
    trades, equity, final, monthly = backtest(df)
    analyze_results(trades, equity, final, monthly)
    
    # Optimize
    best = optimize_parameters(df)
    
    if best:
        print("\n" + "="*70)
        print("BACKTEST WITH OPTIMIZED PARAMETERS")
        print("="*70)
        print(f"Using: SL={best['atr_sl']}, TP={best['atr_tp']}, Size={best['pos_size']:.0%}, MinSignals={best['min_sig']}")
        trades2, equity2, final2, monthly2 = backtest(df, best['atr_sl'], best['atr_tp'], best['pos_size'], best['min_sig'])
        analyze_results(trades2, equity2, final2, monthly2)


if __name__ == "__main__":
    main()
