#!/usr/bin/env python3
"""
GOLD TRADER - Yahoo Finance for analysis, Binance for trading
Uses real gold (XAUUSD) data from Yahoo, trades PAXGUSDT on Binance
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def fetch_yahoo_gold(days=365):
    """Fetch real gold prices from Yahoo Finance"""
    print("📥 Fetching XAUUSD from Yahoo Finance...")
    gold = yf.download("GC=F", period=f"{days}d", interval="1h", progress=False)
    if gold.empty:
        gold = yf.download("GLD", period=f"{days}d", interval="1h", progress=False)
    
    gold.columns = [c.lower() for c in gold.columns]
    print(f"   Got {len(gold)} candles from Yahoo")
    return gold

def fetch_binance_gold(symbol="PAXGUSDT", timeframe="1h", days=180):
    """Fetch gold proxy from Binance"""
    print(f"📥 Fetching {symbol} from Binance...")
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    tf_minutes = {'15m': 15, '1h': 60, '4h': 240}
    total = min((days * 24 * 60) // tf_minutes.get(timeframe, 60), 5000)
    
    while len(all_data) < total:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": symbol, "interval": timeframe, "limit": 1500, "endTime": end_time}, timeout=10)
            data = r.json()
            if not data or isinstance(data, dict): break
            all_data = data + all_data
            end_time = data[0][0] - 1
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data, columns=['ts','open','high','low','close','volume','ct','qv','t','tb','tq','i'])
    for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    print(f"   Got {len(df)} candles from Binance")
    return df

def calc_indicators(df):
    """Calculate indicators"""
    df = df.copy()
    df['ema8'] = df['close'].ewm(span=8).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
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
    
    df['mom5'] = df['close'].pct_change(5) * 100
    
    return df

# STRATEGIES
def mean_reversion(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['zscore'].iloc[i] < -1.5 and df['bb_pct'].iloc[i] < 0.15 and df['rsi'].iloc[i] < 35:
            signals.iloc[i] = 1
        elif df['zscore'].iloc[i] > 1.5 and df['bb_pct'].iloc[i] > 0.85 and df['rsi'].iloc[i] > 65:
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

def combo(df):
    signals = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        long_score = sum([
            df['ema8'].iloc[i] > df['ema21'].iloc[i],
            df['macd_hist'].iloc[i] > 0,
            40 < df['rsi'].iloc[i] < 70,
            df['close'].iloc[i] > df['ema21'].iloc[i],
            df['mom5'].iloc[i] > 0,
            df['adx'].iloc[i] > 20
        ])
        short_score = sum([
            df['ema8'].iloc[i] < df['ema21'].iloc[i],
            df['macd_hist'].iloc[i] < 0,
            30 < df['rsi'].iloc[i] < 60,
            df['close'].iloc[i] < df['ema21'].iloc[i],
            df['mom5'].iloc[i] < 0,
            df['adx'].iloc[i] > 20
        ])
        if long_score >= 4: signals.iloc[i] = 1
        elif short_score >= 4: signals.iloc[i] = -1
    return signals

def backtest(df, signals, capital=10000, commission=0.0004, atr_sl=2.0, atr_tp=3.0, pos_size=0.25):
    position = None
    trades = []
    equity = [capital]
    
    for i in range(50, len(df)):
        price, atr, sig = df['close'].iloc[i], df['atr'].iloc[i], signals.iloc[i]
        
        if position:
            if position['side'] == 'LONG':
                if price <= position['stop']:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * commission * 2
                    trades.append({'pnl': pnl, 'type': 'SL'})
                    capital += pnl
                    position = None
                elif price >= position['target']:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * commission * 2
                    trades.append({'pnl': pnl, 'type': 'TP'})
                    capital += pnl
                    position = None
                elif sig == -1:
                    pnl = (price - position['entry']) * position['size'] - price * position['size'] * commission * 2
                    trades.append({'pnl': pnl, 'type': 'REV'})
                    capital += pnl
                    position = None
            else:
                if price >= position['stop']:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * commission * 2
                    trades.append({'pnl': pnl, 'type': 'SL'})
                    capital += pnl
                    position = None
                elif price <= position['target']:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * commission * 2
                    trades.append({'pnl': pnl, 'type': 'TP'})
                    capital += pnl
                    position = None
                elif sig == 1:
                    pnl = (position['entry'] - price) * position['size'] - price * position['size'] * commission * 2
                    trades.append({'pnl': pnl, 'type': 'REV'})
                    capital += pnl
                    position = None
        
        if position is None and sig != 0:
            size = (capital * pos_size) / price
            if sig == 1:
                position = {'side': 'LONG', 'entry': price, 'size': size, 'stop': price - atr_sl * atr, 'target': price + atr_tp * atr}
            else:
                position = {'side': 'SHORT', 'entry': price, 'size': size, 'stop': price + atr_sl * atr, 'target': price - atr_tp * atr}
        
        equity.append(capital)
    
    if position:
        price = df['close'].iloc[-1]
        pnl = (price - position['entry']) * position['size'] if position['side'] == 'LONG' else (position['entry'] - price) * position['size']
        capital += pnl
        trades.append({'pnl': pnl, 'type': 'END'})
    
    if not trades:
        return {'ret': 0, 'trades': 0, 'win': 0, 'dd': 0, 'pf': 0}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    ret = (capital / 10000 - 1) * 100
    win_rate = len(wins) / len(trades) * 100
    
    peak, max_dd = equity[0], 0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd: max_dd = dd
    
    gross_win = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0
    
    return {'ret': ret, 'trades': len(trades), 'win': win_rate, 'dd': max_dd, 'pf': pf, 'wins': len(wins), 'losses': len(losses)}


def main():
    print("="*70)
    print("GOLD TRADING BACKTEST - Yahoo + Binance")
    print("="*70)
    
    # Try different gold symbols on Binance
    SYMBOLS = ['PAXGUSDT', 'XAUUSDT']
    TIMEFRAMES = ['15m', '1h', '4h']
    STRATEGIES = {
        'mean_rev': mean_reversion,
        'macd_cross': macd_cross,
        'trend_follow': trend_follow,
        'rsi_extreme': rsi_extreme,
        'combo': combo,
    }
    
    # Also get Yahoo gold for comparison
    try:
        yahoo_gold = fetch_yahoo_gold(180)
        if not yahoo_gold.empty:
            yahoo_gold = calc_indicators(yahoo_gold)
            print(f"\n📊 YAHOO GOLD (GC=F) Analysis:")
            print(f"   Current: ${yahoo_gold['close'].iloc[-1]:.2f}")
            print(f"   EMA21: ${yahoo_gold['ema21'].iloc[-1]:.2f}")
            print(f"   RSI: {yahoo_gold['rsi'].iloc[-1]:.1f}")
            print(f"   Trend: {'UP' if yahoo_gold['ema8'].iloc[-1] > yahoo_gold['ema21'].iloc[-1] else 'DOWN'}")
    except Exception as e:
        print(f"   Yahoo error: {e}")
    
    results = []
    
    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"📊 {symbol}")
        print(f"{'='*70}")
        
        for tf in TIMEFRAMES:
            df = fetch_binance_gold(symbol, tf, 180)
            if df is None or len(df) < 100:
                print(f"  {tf}: ❌ No data")
                continue
            
            df = calc_indicators(df)
            print(f"\n  {tf} ({len(df)} candles):")
            
            for name, strat in STRATEGIES.items():
                signals = strat(df)
                metrics = backtest(df, signals)
                results.append({'symbol': symbol, 'tf': tf, 'strat': name, **metrics})
                
                emoji = "✅" if metrics['ret'] > 0 else "❌"
                print(f"    {emoji} {name:12} | Ret: {metrics['ret']:>7.2f}% | Win: {metrics['win']:>5.1f}% | DD: {metrics['dd']:>5.1f}% | Trades: {metrics['trades']:>3} | PF: {metrics['pf']:.2f}")
    
    # Best results
    print("\n" + "="*70)
    print("🏆 TOP 10 GOLD STRATEGIES")
    print("="*70)
    best = sorted(results, key=lambda x: x['ret'], reverse=True)[:10]
    for i, r in enumerate(best, 1):
        print(f"{i:2}. {r['symbol']:10} {r['tf']:4} {r['strat']:12} | Ret: {r['ret']:>7.2f}% | Win: {r['win']:>5.1f}% | PF: {r['pf']:.2f}")
    
    # Live-ready
    print("\n" + "="*70)
    print("💎 LIVE-READY (Ret>3%, DD<15%, WR>35%)")
    print("="*70)
    live = [r for r in results if r['ret'] > 3 and r['dd'] < 15 and r['win'] > 35 and r['trades'] > 10]
    live = sorted(live, key=lambda x: x['ret'], reverse=True)[:10]
    if live:
        for i, r in enumerate(live, 1):
            print(f"{i:2}. {r['symbol']:10} {r['tf']:4} {r['strat']:12} | Ret: {r['ret']:>7.2f}% | Win: {r['win']:>5.1f}% | DD: {r['dd']:>5.1f}%")
    else:
        print("   No strategies met criteria - showing best profitable:")
        ok = sorted([r for r in results if r['ret'] > 0], key=lambda x: x['ret'], reverse=True)[:5]
        for i, r in enumerate(ok, 1):
            print(f"{i:2}. {r['symbol']:10} {r['tf']:4} {r['strat']:12} | Ret: {r['ret']:>7.2f}%")


if __name__ == "__main__":
    main()
