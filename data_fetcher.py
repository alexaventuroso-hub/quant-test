"""
Data Fetcher Module - Binance API Integration
Handles historical OHLCV data and real-time market feeds
"""
import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from config import APIConfig, TradingConfig, TIMEFRAME_MAP, TIMEFRAME_MS


class BinanceDataFetcher:
    """
    Fetches market data from Binance API
    - Historical OHLCV candles
    - Real-time price data
    - Order book data
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": config.api_key
        })
    
    def _sign_request(self, params: dict) -> dict:
        """Sign request for authenticated endpoints"""
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params
    
    def get_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch klines with pagination for large date ranges"""
        all_data = []
        current_start = start_time
        
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": 1500,
            }
            if current_start:
                params["startTime"] = current_start
            if end_time:
                params["endTime"] = end_time
                
            response = requests.get(f"https://fapi.binance.com/fapi/v1/klines", params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            print(f"  Fetched {len(all_data)} candles...")
            
            if len(data) < 1500:
                break
            
            current_start = data[-1][0] + 1
            
            if end_time and current_start >= end_time:
                break
            if len(all_data) > 200000:
                break
        
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep='first')]
        return df[["open", "high", "low", "close", "volume"]]

    def get_historical_data(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 90
    ) -> pd.DataFrame:
        """
        Fetch extended historical data (handles pagination)
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        current_start = start_time
        
        while current_start < end_time:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            
            if df.empty:
                break
                
            all_data.append(df)
            
            # Move start time forward
            # last_time = int(df["close_time"].iloc[-1].timestamp() * 1000)
            break  # pagination now in get_klines
            
            # Rate limiting
            time.sleep(0.1)
        
        if all_data:
            return pd.concat(all_data).drop_duplicates()
        return pd.DataFrame()
    
    def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        endpoint = f"{self.config.base_url}/fapi/v1/ticker/price"
        response = self.session.get(endpoint, params={"symbol": symbol})
        response.raise_for_status()
        return float(response.json()["price"])
    
    def get_all_tickers(self) -> Dict[str, float]:
        """Get all current prices"""
        endpoint = f"{self.config.base_url}/fapi/v1/ticker/price"
        response = self.session.get(endpoint)
        response.raise_for_status()
        return {item["symbol"]: float(item["price"]) for item in response.json()}
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth"""
        endpoint = f"{self.config.base_url}/fapi/v1/depth"
        response = self.session.get(endpoint, params={"symbol": symbol, "limit": limit})
        response.raise_for_status()
        data = response.json()
        
        return {
            "bids": [(float(p), float(q)) for p, q in data["bids"]],
            "asks": [(float(p), float(q)) for p, q in data["asks"]],
            "spread": float(data["asks"][0][0]) - float(data["bids"][0][0])
        }
    
    def get_24h_stats(self, symbol: str) -> Dict:
        """Get 24h trading statistics"""
        endpoint = f"{self.config.base_url}/fapi/v1/ticker/24hr"
        response = self.session.get(endpoint, params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        
        return {
            "price_change_pct": float(data["priceChangePercent"]),
            "volume": float(data["volume"]),
            "quote_volume": float(data["quoteVolume"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "trades": int(data["count"])
        }


class DataPreprocessor:
    """
    Preprocesses market data and adds technical indicators
    """
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators"""
        df = df.copy()
        
        # Simple Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        
        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # RSI
        df["rsi"] = DataPreprocessor._calculate_rsi(df["close"], period=14)
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        # ATR (Average True Range)
        df["atr"] = DataPreprocessor._calculate_atr(df, period=14)
        
        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Volatility
        df["volatility"] = df["returns"].rolling(window=20).std() * np.sqrt(24)  # Annualized for hourly
        
        # Momentum
        df["momentum"] = df["close"] / df["close"].shift(10) - 1
        
        return df
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        df = DataPreprocessor.add_indicators(df)
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
            df[f"volume_ratio_lag_{lag}"] = df["volume_ratio"].shift(lag)
        
        # Price position features
        df["price_position_bb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        df["price_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]
        
        # Target variable (next period return)
        df["target"] = df["returns"].shift(-1)
        df["target_direction"] = (df["target"] > 0).astype(int)
        
        return df.dropna()


if __name__ == "__main__":
    # Example usage
    config = APIConfig(testnet=True)
    fetcher = BinanceDataFetcher(config)
    
    # Fetch historical data
    print("Fetching BTC/USDT historical data...")
    df = fetcher.get_historical_data("BTCUSDT", interval="1h", days=30)
    
    # Add indicators
    df = DataPreprocessor.add_indicators(df)
    
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
