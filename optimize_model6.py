#!/usr/bin/env python3
"""
OPTIMIZE MODEL #6 - THE BEST ONE
Base: Z=1.0, RSI=25/75, SL=1.5x, TP=6.0x, 30% size
Result: +10.95%, 1.0% DD, 42.2% win, PF 2.33

Adding advanced indicators to try to beat it.
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
    
    # EMAs
    df['ema8'] = df['close'].ewm(span=8).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Stochastic
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Stochastic RSI
    rsi_min = df['rsi'].rolling(14).min()
    rsi_max = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = 100 * (df['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-10)
    
    # Williams %R
    df['williams_r'] = -100 * (high14 - df['close']) / (high14 - low14 + 1e-10)
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)
    
    # MFI
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    
    # MACD
    ema12, ema26 = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
    
    # ATR
    tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['zscore'] = (df['close'] - df['bb_mid']) / (df['bb_std'] + 1e-10)
    
    # Keltner Channels
    df['kelt_mid'] = df['close'].ewm(span=20).mean()
    df['kelt_upper'] = df['kelt_mid'] + 2 * df['atr']
    df['kelt_lower'] = df['kelt_mid'] - 2 * df['atr']
    
    # ADX
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-10)
    
    return df

def get_signal(df, i, params):
    """Model 6 base + enhancements"""
    row = df.iloc[i]
    prev = df.iloc[i-1]
    
    # RANGE regime only (ADX < threshold)
    if row['adx'] >= params.get('adx_max', 20):
        return 0
    
    zscore = params.get('zscore', 1.0)
    rsi_low = params.get('rsi_low', 25)
    rsi_high = params.get('rsi_high', 75)
    bb_low = params.get('bb_low', 0.1)
    bb_high = params.get('bb_high', 0.9)
    
    # Get filters
    stoch_filter = params.get('stoch', False)
    stoch_rsi_filter = params.get('stoch_rsi', False)
    williams_filter = params.get('williams', False)
    cci_filter = params.get('cci', False)
    mfi_filter = params.get('mfi', False)
    macd_turn = params.get('macd_turn', False)
    kelt_filter = params.get('kelt', False)
    vol_filter = params.get('vol', False)
    rsi_turn = params.get('rsi_turn', False)
    
    # LONG CONDITIONS
    long_base = row['zscore'] < -zscore and row['rsi'] < rsi_low and row['bb_pct'] < bb_low
    
    if long_base:
        # Apply filters
        if stoch_filter and row['stoch_k'] >= 20:
            return 0
        if stoch_rsi_filter and row['stoch_rsi'] >= 20:
            return 0
        if williams_filter and row['williams_r'] >= -80:
            return 0
        if cci_filter and row['cci'] >= -100:
            return 0
        if mfi_filter and row['mfi'] >= 30:
            return 0
        if macd_turn and not (row['macd_hist'] > prev['macd_hist']):
            return 0
        if kelt_filter and row['close'] >= row['kelt_lower']:
            return 0
        if vol_filter and row['vol_ratio'] < 1.0:
            return 0
        if rsi_turn and not (row['rsi'] > prev['rsi']):
            return 0
        return 1
    
    # SHORT CONDITIONS
    short_base = row['zscore'] > zscore and row['rsi'] > rsi_high and row['bb_pct'] > bb_high
    
    if short_base:
        if stoch_filter and row['stoch_k'] <= 80:
            return 0
        if stoch_rsi_filter and row['stoch_rsi'] <= 80:
            return 0
        if williams_filter and row['williams_r'] <= -20:
            return 0
        if cci_filter and row['cci'] <= 100:
            return 0
        if mfi_filter and row['mfi'] <= 70:
            return 0
        if macd_turn and not (row['macd_hist'] < prev['macd_hist']):
            return 0
        if kelt_filter and row['close'] <= row['kelt_upper']:
            return 0
        if vol_filter and row['vol_ratio'] < 1.0:
            return 0
        if rsi_turn and not (row['rsi'] < prev['rsi']):
            return 0
        return -1
    
    return 0

def backtest(df, params):
    capital = 10000
    position = None
    trades = []
    equity = [capital]
    monthly = {}
    
    atr_sl = params.get('atr_sl', 1.5)
    atr_tp = params.get('atr_tp', 6.0)
    pos_size = params.get('pos_size', 0.30)
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        price, atr = row['close'], row['atr']
        month = df.index[i].strftime('%Y-%m')
        
        if month not in monthly:
            monthly[month] = {'start': capital}
        
        signal = get_signal(df, i, params)
        
        if position:
            side = position['side']
            if side == 'LONG':
                if price <= position['stop'] or price >= position['target'] or signal == -1:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': side})
                    position = None
            else:
                if price >= position['stop'] or price <= position['target'] or signal == 1:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * 0.0008
                    capital += pnl
                    trades.append({'pnl': pnl, 'side': side})
                    position = None
        
        if position is None and signal != 0:
            size = (capital * pos_size) / price
            if signal == 1:
                position = {'side': 'LONG', 'entry': price, 'size': size,
                           'stop': price - atr_sl * atr, 'target': price + atr_tp * atr}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size,
                           'stop': price + atr_sl * atr, 'target': price - atr_tp * atr}
        
        equity.append(capital)
        monthly[month]['end'] = capital
    
    if position:
        price = df['close'].iloc[-1]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'side': position['side']})
    
    if not trades:
        return None
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    ret = (capital / 10000 - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
    pf = sum(t['pnl'] for t in wins) / abs(sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0
    
    peak, max_dd = equity[0], 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    
    long_pnl = sum(t['pnl'] for t in trades if t['side'] == 'LONG')
    short_pnl = sum(t['pnl'] for t in trades if t['side'] == 'SHORT')
    
    return {
        'ret': ret, 'win_rate': win_rate, 'trades': len(trades), 'max_dd': max_dd,
        'pf': pf, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
        'expectancy': expectancy, 'params': params, 'monthly': monthly,
        'long_pnl': long_pnl, 'short_pnl': short_pnl
    }

def main():
    print("="*70)
    print("OPTIMIZING MODEL #6 (THE BEST ONE)")
    print("Base: +10.95%, 1.0% DD, 42.2% win, PF 2.33")
    print("Params: Z=1.0, RSI=25/75, SL=1.5x, TP=6.0x, Size=30%")
    print("="*70)
    
    print("\nFetching data...")
    df = fetch_data("ETHUSDT", "15m", 180)
    df = calc_indicators(df)
    print(f"Got {len(df)} candles")
    
    # First: Reproduce baseline
    print("\n" + "="*70)
    print("BASELINE (Model #6 exact)")
    print("="*70)
    
    baseline_params = {
        'zscore': 1.0, 'rsi_low': 25, 'rsi_high': 75,
        'bb_low': 0.1, 'bb_high': 0.9,
        'atr_sl': 1.5, 'atr_tp': 6.0, 'pos_size': 0.30,
        'adx_max': 20
    }
    baseline = backtest(df, baseline_params)
    if baseline:
        print(f"Return: {baseline['ret']:+.2f}%")
        print(f"Max DD: {baseline['max_dd']:.2f}%")
        print(f"Win Rate: {baseline['win_rate']:.1f}%")
        print(f"Profit Factor: {baseline['pf']:.2f}")
        print(f"Trades: {baseline['trades']}")
        print(f"Expectancy: ${baseline['expectancy']:.2f}/trade")
        print(f"LONG P&L: ${baseline['long_pnl']:.2f} | SHORT P&L: ${baseline['short_pnl']:.2f}")
    
    # Parameter variations around Model #6
    results = []
    
    # Test 1: Core parameters variations
    print("\n" + "="*70)
    print("TEST 1: Core Parameter Variations (~50 combos)")
    print("="*70)
    
    core_params = list(product(
        [0.8, 1.0, 1.2],       # zscore
        [20, 25, 30],          # rsi_low
        [70, 75, 80],          # rsi_high
        [1.0, 1.5, 2.0],       # atr_sl
        [5.0, 6.0, 7.0],       # atr_tp
    ))
    
    # Sample ~50
    import random
    random.seed(42)
    core_sample = random.sample(core_params, min(50, len(core_params)))
    
    for combo in core_sample:
        params = {
            'zscore': combo[0], 'rsi_low': combo[1], 'rsi_high': combo[2],
            'bb_low': 0.1, 'bb_high': 0.9,
            'atr_sl': combo[3], 'atr_tp': combo[4], 'pos_size': 0.30,
            'adx_max': 20
        }
        result = backtest(df, params)
        if result and result['trades'] >= 5:
            result['test'] = 'CORE'
            results.append(result)
    
    print(f"Tested {len(core_sample)} combinations, {len(results)} valid")
    
    # Test 2: Single filter additions
    print("\n" + "="*70)
    print("TEST 2: Single Filter Additions (~50 combos)")
    print("="*70)
    
    filters = ['stoch', 'stoch_rsi', 'williams', 'cci', 'mfi', 'macd_turn', 'kelt', 'vol', 'rsi_turn']
    
    for filt in filters:
        for zscore in [0.8, 1.0, 1.2]:
            for atr_tp in [5.0, 6.0, 7.0]:
                params = baseline_params.copy()
                params['zscore'] = zscore
                params['atr_tp'] = atr_tp
                params[filt] = True
                result = backtest(df, params)
                if result and result['trades'] >= 5:
                    result['test'] = f'FILTER:{filt}'
                    results.append(result)
    
    print(f"Total results so far: {len(results)}")
    
    # Test 3: Filter combinations
    print("\n" + "="*70)
    print("TEST 3: Filter Combinations (~50 combos)")
    print("="*70)
    
    filter_combos = [
        ['stoch', 'rsi_turn'],
        ['stoch', 'macd_turn'],
        ['williams', 'rsi_turn'],
        ['cci', 'rsi_turn'],
        ['mfi', 'stoch'],
        ['stoch_rsi', 'macd_turn'],
        ['kelt', 'stoch'],
        ['vol', 'stoch'],
        ['stoch', 'cci'],
        ['williams', 'mfi'],
    ]
    
    for combo in filter_combos:
        for zscore in [0.8, 1.0, 1.2]:
            for atr_tp in [5.0, 6.0, 7.0]:
                params = baseline_params.copy()
                params['zscore'] = zscore
                params['atr_tp'] = atr_tp
                for f in combo:
                    params[f] = True
                result = backtest(df, params)
                if result and result['trades'] >= 5:
                    result['test'] = f'COMBO:{"+".join(combo)}'
                    results.append(result)
    
    print(f"Total results: {len(results)}")
    
    # RESULTS
    print("\n" + "="*70)
    print("TOP 15 BY RETURN")
    print("="*70)
    
    by_ret = sorted(results, key=lambda x: x['ret'], reverse=True)[:15]
    for i, r in enumerate(by_ret, 1):
        ratio = r['ret']/r['max_dd'] if r['max_dd'] > 0 else 0
        print(f"\n{i:2}. [{r['test'][:20]:20}]")
        print(f"    Return: {r['ret']:+.2f}% | DD: {r['max_dd']:.2f}% | Ratio: {ratio:.1f}x")
        print(f"    Win: {r['win_rate']:.1f}% | PF: {r['pf']:.2f} | W/L: {r['win_loss_ratio']:.2f} | Trades: {r['trades']}")
        print(f"    Exp: ${r['expectancy']:.2f}/trade | AvgWin: ${r['avg_win']:.2f} | AvgLoss: ${r['avg_loss']:.2f}")
    
    print("\n" + "="*70)
    print("TOP 10 BY RISK-ADJUSTED (Return/DD)")
    print("="*70)
    
    by_ratio = sorted([r for r in results if r['max_dd'] > 0], key=lambda x: x['ret']/x['max_dd'], reverse=True)[:10]
    for i, r in enumerate(by_ratio, 1):
        ratio = r['ret']/r['max_dd']
        print(f"{i:2}. [{r['test'][:15]:15}] Ratio:{ratio:>5.1f}x | Ret:{r['ret']:>+6.2f}% | DD:{r['max_dd']:>4.2f}% | Win:{r['win_rate']:.0f}%")
    
    print("\n" + "="*70)
    print("TOP 10 BY EXPECTANCY ($/trade)")
    print("="*70)
    
    by_exp = sorted(results, key=lambda x: x['expectancy'], reverse=True)[:10]
    for i, r in enumerate(by_exp, 1):
        print(f"{i:2}. [{r['test'][:15]:15}] ${r['expectancy']:>6.2f}/trade | Ret:{r['ret']:>+6.2f}% | Trades:{r['trades']}")
    
    print("\n" + "="*70)
    print("TOP 10 BY PROFIT FACTOR")
    print("="*70)
    
    by_pf = sorted([r for r in results if r['pf'] > 0], key=lambda x: x['pf'], reverse=True)[:10]
    for i, r in enumerate(by_pf, 1):
        print(f"{i:2}. [{r['test'][:15]:15}] PF:{r['pf']:>5.2f} | Ret:{r['ret']:>+6.2f}% | Win:{r['win_rate']:.0f}% | W/L:{r['win_loss_ratio']:.2f}")
    
    # THE WINNER
    print("\n" + "="*70)
    print("*** BEST OVERALL (Score = Ret * PF * sqrt(trades) / DD) ***")
    print("="*70)
    
    def score(r):
        if r['max_dd'] == 0 or r['trades'] < 10:
            return 0
        return r['ret'] * r['pf'] * np.sqrt(r['trades']) / (r['max_dd'] + 0.1)
    
    winner = max(results, key=score)
    
    print(f"\nTest: {winner['test']}")
    print(f"\n📈 PERFORMANCE:")
    print(f"   Return: {winner['ret']:+.2f}% ({winner['ret']/6:.2f}%/month)")
    print(f"   Max Drawdown: {winner['max_dd']:.2f}%")
    print(f"   Risk/Reward Ratio: {winner['ret']/winner['max_dd']:.1f}x")
    
    print(f"\n📊 STATISTICS:")
    print(f"   Win Rate: {winner['win_rate']:.1f}%")
    print(f"   Profit Factor: {winner['pf']:.2f}")
    print(f"   Win/Loss Ratio: {winner['win_loss_ratio']:.2f}x")
    print(f"   Avg Win: ${winner['avg_win']:.2f}")
    print(f"   Avg Loss: ${winner['avg_loss']:.2f}")
    print(f"   Expectancy: ${winner['expectancy']:.2f}/trade")
    print(f"   Trades: {winner['trades']} (~{winner['trades']/6:.0f}/month)")
    
    print(f"\n💰 PROJECTED RETURNS:")
    for cap in [1000, 5000, 10000, 50000]:
        profit = cap * winner['ret'] / 100
        monthly = profit / 6
        max_loss = cap * winner['max_dd'] / 100
        print(f"   ${cap:,}: +${profit:,.0f} total | ~${monthly:,.0f}/month | Max loss: ${max_loss:,.0f}")
    
    print(f"\n⚙️ PARAMETERS:")
    for k, v in winner['params'].items():
        print(f"   {k}: {v}")
    
    print(f"\n📅 MONTHLY BREAKDOWN:")
    for m, data in sorted(winner['monthly'].items()):
        ret = (data['end'] / data['start'] - 1) * 100
        print(f"   {m}: {'+' if ret >= 0 else ''}{ret:.2f}%")
    
    # Compare to baseline
    print("\n" + "="*70)
    print("COMPARISON: WINNER vs BASELINE")
    print("="*70)
    print(f"{'Metric':<20} {'Baseline':>12} {'Winner':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Return':<20} {baseline['ret']:>+11.2f}% {winner['ret']:>+11.2f}% {winner['ret']-baseline['ret']:>+11.2f}%")
    print(f"{'Max DD':<20} {baseline['max_dd']:>11.2f}% {winner['max_dd']:>11.2f}% {winner['max_dd']-baseline['max_dd']:>+11.2f}%")
    print(f"{'Win Rate':<20} {baseline['win_rate']:>11.1f}% {winner['win_rate']:>11.1f}% {winner['win_rate']-baseline['win_rate']:>+11.1f}%")
    print(f"{'Profit Factor':<20} {baseline['pf']:>12.2f} {winner['pf']:>12.2f} {winner['pf']-baseline['pf']:>+12.2f}")
    print(f"{'Expectancy':<20} ${baseline['expectancy']:>10.2f} ${winner['expectancy']:>10.2f} ${winner['expectancy']-baseline['expectancy']:>+10.2f}")
    print(f"{'Trades':<20} {baseline['trades']:>12} {winner['trades']:>12} {winner['trades']-baseline['trades']:>+12}")


if __name__ == "__main__":
    main()
