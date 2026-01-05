# 🚀 Crypto Quant Trading System

A modular, professional-grade algorithmic trading system for cryptocurrency markets, inspired by quantitative trading approaches like those used by Renaissance Technologies.

## ✨ Features

- **Multiple Trading Strategies**: MA Crossover, RSI Mean Reversion, Bollinger Bands, Momentum, MACD, and Ensemble
- **Backtesting Engine**: Test strategies on historical data with realistic assumptions
- **Paper Trading**: Practice without risking real money
- **Live Trading**: Execute real trades on Binance
- **Risk Management**: Position sizing, stop losses, take profits, drawdown protection
- **Technical Indicators**: 20+ built-in indicators
- **Modular Architecture**: Easy to add new strategies and data sources

## 📁 Project Structure

```
crypto_quant/
├── config.py          # Configuration settings
├── data_fetcher.py    # Binance API data fetching & indicators
├── strategies.py      # Trading strategies
├── backtester.py      # Backtesting engine
├── trader.py          # Live/Paper trading execution
├── main.py            # Main bot orchestrator
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🛠️ Installation

```bash
# Clone or download the project
cd crypto_quant

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

### API Keys Setup

1. Create a Binance account at https://www.binance.com
2. Generate API keys (Settings → API Management)
3. For paper trading, use Binance Testnet: https://testnet.binance.vision
4. Add your keys to `config.py` or use environment variables:

```python
# In config.py
api_config = APIConfig(
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True  # Use testnet for paper trading
)
```

Or use environment variables:
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

## 🎯 Usage

### 1. Backtesting

Test strategies on historical data before trading:

```bash
# Run backtest with default settings
python main.py --mode backtest

# Backtest specific symbols and strategy
python main.py --mode backtest --symbols BTCUSDT ETHUSDT --strategy rsi --days 90

# Backtest with ensemble strategy
python main.py --mode backtest --strategy ensemble --capital 10000
```

### 2. Paper Trading

Practice without real money:

```bash
# Start paper trading
python main.py --mode paper

# With custom settings
python main.py --mode paper --symbols BTCUSDT SOLUSDT --interval 3600 --capital 10000
```

### 3. Live Trading

⚠️ **WARNING: This uses real money. Start small and monitor closely!**

```bash
python main.py --mode live --symbols BTCUSDT --strategy ma_crossover
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | `backtest`, `paper`, or `live` | `paper` |
| `--symbols` | Trading pairs | `BTCUSDT ETHUSDT` |
| `--strategy` | Strategy to use | `ensemble` |
| `--interval` | Trading interval (seconds) | `3600` |
| `--capital` | Initial capital | `10000` |
| `--days` | Backtest period | `90` |

## 📈 Strategies

### 1. MA Crossover (`ma_crossover`)
Classic trend-following using fast/slow moving average crossovers.

### 2. RSI Mean Reversion (`rsi`)
Contrarian strategy based on oversold/overbought RSI levels.

### 3. Bollinger Bands (`bollinger`)
Mean reversion based on price deviation from bands.

### 4. Momentum (`momentum`)
Trend-following based on price momentum.

### 5. MACD (`macd`)
Signal line crossover strategy.

### 6. Ensemble (`ensemble`)
Combines all strategies using weighted voting.

## 🔧 Creating Custom Strategies

```python
from strategies import BaseStrategy, Signal, TradeSignal

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MyCustomStrategy")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        # Your logic here
        # signals[condition] = 1   # Buy
        # signals[condition] = -1  # Sell
        
        return signals
    
    def get_signal(self, df: pd.DataFrame) -> TradeSignal:
        signals = self.generate_signals(df)
        current = signals.iloc[-1]
        
        signal_type = Signal.HOLD
        if current > 0:
            signal_type = Signal.BUY
        elif current < 0:
            signal_type = Signal.SELL
        
        return TradeSignal(
            signal=signal_type,
            symbol=df.name,
            price=df["close"].iloc[-1],
            timestamp=df.index[-1]
        )
```

## 📊 Performance Metrics

The backtester calculates:

- **Total Return**: Overall strategy return
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Alpha**: Excess return vs buy-and-hold

## ⚠️ Risk Warnings

1. **Cryptocurrency is highly volatile** - You can lose your entire investment
2. **Past performance doesn't guarantee future results**
3. **Start with paper trading** before risking real money
4. **Never invest more than you can afford to lose**
5. **Monitor your bot regularly** - Bugs can cause losses
6. **Test thoroughly** before live trading

## 🔐 Security Best Practices

1. **Never share API keys** or commit them to git
2. **Use environment variables** for sensitive data
3. **Restrict API key permissions** (disable withdrawals)
4. **IP whitelist** your trading server
5. **Use read-only keys** for monitoring

## 📚 Resources

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Binance Testnet](https://testnet.binance.vision/)
- [Technical Analysis Library](https://ta-lib.org/)
- [Quantitative Trading](https://www.quantstart.com/)

## 📝 License

MIT License - Use at your own risk.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Disclaimer**: This software is for educational purposes only. Trading cryptocurrencies carries significant risk. Always do your own research and consult with a financial advisor before trading.
