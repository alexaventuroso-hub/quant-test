#!/usr/bin/env python3
"""
MULTI-PAIR MULTI-TIMEFRAME BACKTESTER
=====================================
Test Renaissance strategy across crypto pairs
"""
import sys
sys.path.insert(0, '/Users/alexfuge23/Downloads/crypto_quant-12')

from strategies import get_strategy, STRATEGIES
from strategies.renaissance import RENAISSANCE_STRATEGIES
from data_fetcher import BinanceDataFetcher
from config import APIConfig
from backtester import Backtester
import time

# Update strategies
STRATEGIES.update(RENAISSANCE_STRATEGIES)

# Test pairs (high liquidity)
PAIRS = [
    'BTCUSDT',
    'ETHUSDT', 
    'SOLUSDT',
    'XRPUSDT',
    'BNBUSDT',
    'DOGEUSDT',
]

# Timeframes to test
TIMEFRAMES = ['5m', '15m', '1h', '4h']

# Strategies to test
STRATS = ['renaissance', 'renaissance_agg', 'pro']

def run_backtest(symbol, timeframe, strategy_name, days=90):
    try:
        config = APIConfig()
        fetcher = BinanceDataFetcher(config)
        df = fetcher.get_historical_data(symbol, timeframe, days)
        
        if df is None or len(df) < 100:
            return None
        
        strat = get_strategy(strategy_name)
        bt = Backtester(initial_capital=10000, commission=0.0004)
        result = bt.run(df, strat, symbol)
        m = result.metrics
        
        return {
            'symbol': symbol,
            'tf': timeframe,
            'strategy': strategy_name,
            'return': m.get('total_return_pct', 0),
            'max_dd': m.get('max_drawdown_pct', 0),
            'trades': m.get('total_trades', 0),
            'sharpe': m.get('sharpe_ratio', 0),
            'win_rate': m.get('win_rate', 0),
        }
    except Exception as e:
        print(f"Error {symbol} {timeframe}: {e}")
        return None

def main():
    print("="*80)
    print("RENAISSANCE STRATEGY - MULTI-PAIR BACKTEST")
    print("="*80)
    
    results = []
    
    for symbol in PAIRS:
        print(f"\n📊 Testing {symbol}...")
        for tf in TIMEFRAMES:
            for strat in STRATS:
                r = run_backtest(symbol, tf, strat, days=180)
                if r:
                    results.append(r)
                    emoji = "✅" if r['return'] > 0 else "❌"
                    print(f"  {emoji} {tf:4} {strat:20} | Ret: {r['return']:>7.2f}% | DD: {r['max_dd']:>6.2f}% | Trades: {r['trades']:>4} | Sharpe: {r['sharpe']:.2f}")
                time.sleep(0.3)  # Rate limit
    
    # Sort by return
    print("\n" + "="*80)
    print("TOP 10 BEST COMBINATIONS")
    print("="*80)
    
    sorted_results = sorted(results, key=lambda x: x['return'], reverse=True)[:10]
    for i, r in enumerate(sorted_results, 1):
        print(f"{i:2}. {r['symbol']:10} {r['tf']:4} {r['strategy']:20} | Ret: {r['return']:>7.2f}% | DD: {r['max_dd']:>6.2f}% | Sharpe: {r['sharpe']:.2f}")
    
    # Best risk-adjusted (by Sharpe)
    print("\n" + "="*80)
    print("TOP 10 BY SHARPE RATIO (Risk-Adjusted)")
    print("="*80)
    
    sharpe_sorted = sorted([r for r in results if r['trades'] > 5], key=lambda x: x['sharpe'], reverse=True)[:10]
    for i, r in enumerate(sharpe_sorted, 1):
        print(f"{i:2}. {r['symbol']:10} {r['tf']:4} {r['strategy']:20} | Sharpe: {r['sharpe']:.2f} | Ret: {r['return']:>7.2f}% | DD: {r['max_dd']:>6.2f}%")

if __name__ == "__main__":
    main()
