"""
Configuration for Crypto Quant Trading System
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    """Binance API Configuration"""
    api_key: str = "PM2qnx6o8kcl4u1IQMVWB9MHevOwmHWj0Fx9naCmXBpZN7nDtnclv8kRaDpYjb2w"
    api_secret: str = "iNMPUJurYtudZg9yKY0ygKxtHkLQQHXVojd0h0jL2m3s1AjE6vfcnHflmcN1Clw5"
    testnet: bool = True  # Use testnet for paper trading
    
    # Testnet endpoints
    TESTNET_BASE_URL = "https://testnet.binance.vision"
    TESTNET_WS_URL = "wss://testnet.binance.vision/ws"
    
    # Production endpoints
    PROD_BASE_URL = "https://api.binance.com"
    PROD_WS_URL = "wss://stream.binance.com:9443/ws"
    
    @property
    def base_url(self) -> str:
        return self.TESTNET_BASE_URL if self.testnet else self.PROD_BASE_URL
    
    @property
    def ws_url(self) -> str:
        return self.TESTNET_WS_URL if self.testnet else self.PROD_WS_URL


@dataclass
class TradingConfig:
    """Trading Parameters"""
    symbols: list = None  # Trading pairs
    timeframe: str = "1h"  # Candle timeframe
    lookback_days: int = 90  # Historical data for backtesting
    
    # Risk Management
    max_position_size: float = 0.1  # Max 10% of portfolio per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_open_positions: int = 3  # Max concurrent positions
    
    # Trading fees (Binance)
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@dataclass 
class BacktestConfig:
    """Backtesting Parameters"""
    initial_capital: float = 10000.0  # Starting capital in USDT
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage assumption


# Timeframe mappings for Binance API
TIMEFRAME_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M"
}

# Timeframe to milliseconds
TIMEFRAME_MS = {
    "1m": 60000, "5m": 300000, "15m": 900000, "30m": 1800000,
    "1h": 3600000, "4h": 14400000, "1d": 86400000, "1w": 604800000
}
