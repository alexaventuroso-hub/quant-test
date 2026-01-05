"""
Live Trading Module - Binance API Execution
Handles order placement, position management, and real-time trading
"""
import time
import hmac
import hashlib
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import logging

from config import APIConfig, TradingConfig
from strategies import BaseStrategy, Signal, TradeSignal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Represents an order"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    timestamp: datetime
    filled_qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_time: datetime = None


class BinanceTrader:
    """
    Live trading interface for Binance
    
    Features:
    - Market and limit orders
    - Position tracking
    - Risk management
    - Order management
    """
    
    def __init__(self, api_config: APIConfig, trading_config: TradingConfig):
        self.api_config = api_config
        self.trading_config = trading_config
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": api_config.api_key
        })
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
    
    def _sign_request(self, params: dict) -> dict:
        """Sign request with API secret"""
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params
    
    def _make_request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """Make API request"""
        url = f"{self.api_config.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        if signed:
            params = self._sign_request(params)
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params)
            elif method == "POST":
                response = self.session.post(url, params=params)
            elif method == "DELETE":
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_account_info(self) -> dict:
        """Get account information and balances"""
        return self._make_request("GET", "/api/v3/account", signed=True)
    
    def get_balance(self, asset: str = "USDT") -> float:
        """Get balance for specific asset"""
        account = self.get_account_info()
        for balance in account["balances"]:
            if balance["asset"] == asset:
                return float(balance["free"])
        return 0.0
    
    def get_symbol_info(self, symbol: str) -> dict:
        """Get trading rules for a symbol"""
        info = self._make_request("GET", "/api/v3/exchangeInfo", params={"symbol": symbol})
        return info["symbols"][0] if info.get("symbols") else {}
    
    def get_ticker_price(self, symbol: str) -> float:
        """Get current price"""
        result = self._make_request("GET", "/api/v3/ticker/price", params={"symbol": symbol})
        return float(result["price"])
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        quantity: Optional[float] = None,
        quote_quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Order:
        """
        Place an order
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: BUY or SELL
            order_type: MARKET, LIMIT, etc.
            quantity: Base asset quantity
            quote_quantity: Quote asset quantity (for market orders)
            price: Limit price
            stop_price: Stop trigger price
            time_in_force: GTC, IOC, FOK
        """
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
        }
        
        if quantity:
            params["quantity"] = f"{quantity:.8f}"
        if quote_quantity:
            params["quoteOrderQty"] = f"{quote_quantity:.2f}"
        if price and order_type != OrderType.MARKET:
            params["price"] = f"{price:.8f}"
            params["timeInForce"] = time_in_force
        if stop_price:
            params["stopPrice"] = f"{stop_price:.8f}"
        
        logger.info(f"Placing {side.value} {order_type.value} order for {symbol}")
        
        result = self._make_request("POST", "/api/v3/order", params=params, signed=True)
        
        order = Order(
            order_id=str(result["orderId"]),
            symbol=result["symbol"],
            side=result["side"],
            order_type=result["type"],
            quantity=float(result.get("origQty", 0)),
            price=float(result.get("price", 0)) if result.get("price") else None,
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["transactTime"] / 1000),
            filled_qty=float(result.get("executedQty", 0)),
            avg_price=float(result.get("fills", [{}])[0].get("price", 0)) if result.get("fills") else 0
        )
        
        self.open_orders[order.order_id] = order
        logger.info(f"Order placed: {order.order_id} - Status: {order.status}")
        
        return order
    
    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order"""
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        result = self._make_request("DELETE", "/api/v3/order", params=params, signed=True)
        
        if order_id in self.open_orders:
            del self.open_orders[order_id]
        
        logger.info(f"Order cancelled: {order_id}")
        return result
    
    def cancel_all_orders(self, symbol: str) -> list:
        """Cancel all open orders for a symbol"""
        result = self._make_request("DELETE", "/api/v3/openOrders", 
                                   params={"symbol": symbol}, signed=True)
        logger.info(f"All orders cancelled for {symbol}")
        return result
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = self._make_request("GET", "/api/v3/openOrders", params=params, signed=True)
        
        orders = []
        for o in result:
            order = Order(
                order_id=str(o["orderId"]),
                symbol=o["symbol"],
                side=o["side"],
                order_type=o["type"],
                quantity=float(o["origQty"]),
                price=float(o["price"]) if o.get("price") else None,
                status=o["status"],
                timestamp=datetime.fromtimestamp(o["time"] / 1000),
                filled_qty=float(o["executedQty"])
            )
            orders.append(order)
        
        return orders
    
    def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Get status of a specific order"""
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        result = self._make_request("GET", "/api/v3/order", params=params, signed=True)
        
        return Order(
            order_id=str(result["orderId"]),
            symbol=result["symbol"],
            side=result["side"],
            order_type=result["type"],
            quantity=float(result["origQty"]),
            price=float(result["price"]) if result.get("price") else None,
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["time"] / 1000),
            filled_qty=float(result["executedQty"]),
            avg_price=float(result.get("price", 0))
        )
    
    def calculate_position_size(self, symbol: str, side: OrderSide) -> float:
        """Calculate position size based on risk management rules"""
        balance = self.get_balance("USDT")
        current_price = self.get_ticker_price(symbol)
        
        # Max position value
        max_position_value = balance * self.trading_config.max_position_size
        
        # Check number of open positions
        if len(self.positions) >= self.trading_config.max_open_positions:
            logger.warning("Max open positions reached")
            return 0.0
        
        # Calculate quantity
        quantity = max_position_value / current_price
        
        # Round to appropriate precision (get from symbol info)
        return round(quantity, 6)
    
    def execute_signal(self, signal: TradeSignal) -> Optional[Order]:
        """Execute a trading signal"""
        if signal.signal == Signal.HOLD:
            return None
        
        symbol = signal.symbol
        current_position = self.positions.get(symbol)
        
        if signal.signal == Signal.BUY:
            if current_position and current_position.side == "long":
                logger.info(f"Already in long position for {symbol}")
                return None
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, OrderSide.BUY)
            if quantity <= 0:
                return None
            
            # Place market buy order
            order = self.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            # Track position
            if order.status in ["FILLED", "NEW"]:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side="long",
                    quantity=quantity,
                    entry_price=signal.price,
                    current_price=signal.price,
                    unrealized_pnl=0.0,
                    entry_time=datetime.now()
                )
            
            return order
            
        elif signal.signal == Signal.SELL:
            if not current_position:
                logger.info(f"No position to sell for {symbol}")
                return None
            
            # Place market sell order
            order = self.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=current_position.quantity
            )
            
            # Remove position
            if order.status in ["FILLED", "NEW"]:
                del self.positions[symbol]
            
            return order
        
        return None
    
    def update_positions(self):
        """Update position P&L with current prices"""
        for symbol, position in self.positions.items():
            current_price = self.get_ticker_price(symbol)
            position.current_price = current_price
            
            if position.side == "long":
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions"""
        self.update_positions()
        
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, pos in self.positions.items():
            pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
            data.append({
                "Symbol": symbol,
                "Side": pos.side,
                "Quantity": pos.quantity,
                "Entry Price": pos.entry_price,
                "Current Price": pos.current_price,
                "Unrealized P&L": pos.unrealized_pnl,
                "P&L %": pnl_pct,
                "Entry Time": pos.entry_time
            })
        
        return pd.DataFrame(data)


class RiskManager:
    """
    Risk management for live trading
    
    Features:
    - Position sizing
    - Stop loss management
    - Exposure limits
    - Drawdown protection
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
    
    def check_position_allowed(self, current_positions: int) -> bool:
        """Check if new position is allowed"""
        return current_positions < self.config.max_open_positions
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == "long":
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == "long":
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)
    
    def update_equity(self, equity: float):
        """Update equity tracking"""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def should_stop_trading(self, max_drawdown: float = 0.1) -> bool:
        """Check if trading should be paused due to drawdown"""
        return self.get_current_drawdown() > max_drawdown


class PaperTrader(BinanceTrader):
    """
    Paper trading (simulated) mode
    
    Uses real market data but simulates order execution
    """
    
    def __init__(self, api_config: APIConfig, trading_config: TradingConfig, 
                 initial_balance: float = 10000.0):
        super().__init__(api_config, trading_config)
        self.paper_balance = {"USDT": initial_balance}
        self.paper_orders = []
        self.order_counter = 0
    
    def get_balance(self, asset: str = "USDT") -> float:
        """Get simulated balance"""
        return self.paper_balance.get(asset, 0.0)
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        quantity: Optional[float] = None,
        quote_quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Order:
        """Simulate order placement"""
        current_price = self.get_ticker_price(symbol)  # Real price from API
        
        # Calculate quantity if quote_quantity provided
        if quote_quantity and not quantity:
            quantity = quote_quantity / current_price
        
        # Simulate execution at current price (with small slippage)
        slippage = 0.001
        if side == OrderSide.BUY:
            exec_price = current_price * (1 + slippage)
            cost = quantity * exec_price
            if self.paper_balance.get("USDT", 0) >= cost:
                self.paper_balance["USDT"] -= cost
                base_asset = symbol.replace("USDT", "")
                self.paper_balance[base_asset] = self.paper_balance.get(base_asset, 0) + quantity
                status = "FILLED"
            else:
                status = "REJECTED"
                logger.warning("Insufficient balance for paper trade")
        else:
            exec_price = current_price * (1 - slippage)
            base_asset = symbol.replace("USDT", "")
            if self.paper_balance.get(base_asset, 0) >= quantity:
                self.paper_balance[base_asset] -= quantity
                self.paper_balance["USDT"] = self.paper_balance.get("USDT", 0) + (quantity * exec_price)
                status = "FILLED"
            else:
                status = "REJECTED"
                logger.warning("Insufficient balance for paper trade")
        
        self.order_counter += 1
        order = Order(
            order_id=f"PAPER_{self.order_counter}",
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=exec_price,
            status=status,
            timestamp=datetime.now(),
            filled_qty=quantity if status == "FILLED" else 0,
            avg_price=exec_price if status == "FILLED" else 0
        )
        
        self.paper_orders.append(order)
        logger.info(f"[PAPER] {side.value} {quantity:.6f} {symbol} @ {exec_price:.2f} - {status}")
        
        return order
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value in USDT"""
        total = self.paper_balance.get("USDT", 0)
        
        for asset, quantity in self.paper_balance.items():
            if asset != "USDT" and quantity > 0:
                try:
                    price = self.get_ticker_price(f"{asset}USDT")
                    total += quantity * price
                except:
                    pass
        
        return total


if __name__ == "__main__":
    # Example usage
    api_config = APIConfig(testnet=True)
    trading_config = TradingConfig()
    
    # Use paper trader for testing
    trader = PaperTrader(api_config, trading_config, initial_balance=10000)
    
    print("Paper Trading Demo")
    print("=" * 50)
    print(f"Initial Balance: ${trader.get_balance('USDT'):.2f}")
    
    # Get current BTC price
    btc_price = trader.get_ticker_price("BTCUSDT")
    print(f"Current BTC Price: ${btc_price:.2f}")
    
    # Simulate a buy order
    quantity = 0.01
    order = trader.place_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity
    )
    
    print(f"\nAfter Buy Order:")
    print(f"  USDT Balance: ${trader.get_balance('USDT'):.2f}")
    print(f"  BTC Balance: {trader.get_balance('BTC'):.6f}")
    print(f"  Portfolio Value: ${trader.get_portfolio_value():.2f}")
