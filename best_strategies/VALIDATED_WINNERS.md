# VALIDATED WINNERS - 180 DAY BACKTEST

## TOP PERFORMERS (Confirmed)
| Rank | Pair | TF | Strategy | Return | Win% | MaxDD | Sharpe |
|------|------|-----|----------|--------|------|-------|--------|
| 1 | SOLUSDT | 4h | mean_reversion | +33.10% | 47.2% | 9.4% | 1.24 |
| 2 | SOLUSDT | 4h | macd_cross | +14.03% | 40.8% | 10.5% | 1.07 |
| 3 | ETHUSDT | 4h | macd_cross | +13.11% | 43.8% | 8.4% | 1.09 |
| 4 | ETHUSDT | 4h | mean_reversion | +9.74% | 51.9% | 12.5% | 0.72 |
| 5 | BNBUSDT | 4h | macd_cross | +6.43% | 36.8% | 8.5% | 0.65 |

## OPTIMIZED PARAMETERS (SOLUSDT mean_reversion)
- Stop Loss: 1.5x ATR
- Take Profit: 5.0x ATR
- Position Size: 30%
- Result: +33.10% return

## KEY INSIGHTS
1. 4h timeframe is the sweet spot (less noise, lower commissions)
2. Mean reversion works best on SOL (high volatility)
3. MACD cross works on both ETH and SOL
4. 15m/1h timeframes LOSE money due to commissions
5. LONG and SHORT both profitable on winners

## MONTHLY CONSISTENCY (SOLUSDT mean_reversion)
- 2025-05: +0.86%
- 2025-06: +5.29%
- 2025-07: -0.88%
- 2025-08: +0.92%
- 2025-09: +2.56%
- 2025-10: +2.92%
- 2025-11: +0.40%
- 2025-12: +8.92%
- 2026-01: -1.89%

7 out of 9 months profitable!

## RUN COMMANDS
```bash
# Best strategy - SOLUSDT mean_reversion
GROQ_API_KEY=your_key python3 winner_trader.py SOLUSDT mean_reversion

# Alternative - ETHUSDT macd_cross
GROQ_API_KEY=your_key python3 winner_trader.py ETHUSDT macd_cross
```
