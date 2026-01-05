"""
Crypto Quant Trading Bot - Main Orchestrator
Ties together data fetching, strategies, backtesting, and execution
"""
import time
import signal
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd

from config import APIConfig, TradingConfig, BacktestConfig
from data_fetcher import BinanceDataFetcher, DataPreprocessor
from strategies import *

from advanced_strategies import (

    AdaptiveStrategy, RenaissanceEnsemble, get_advanced_strategy
)
from turbo_strategies import (
    TurboQuantStrategy, ScalpStrategy, SwingTrader, UltimateQuant,

)
from hft_strategies import (
    HFTScalper, MicroMomentum, HFTCombo, get_hft_strategy
)
from pro_strategies import (
    ProfessionalQuant, VolatilityBreakout, InstitutionalQuant,

)
from elite_strategies import (
    EliteQuant, AggressiveQuant, ComboQuant, get_elite_strategy
)
from alpha_strategies import (
    AlphaTrend, MomentumSurfer, AlphaCombo, get_alpha_strategy
)
from sniper_strategies import (
    Sniper, TrendRider, UltimateSniper, get_sniper_strategy
)
from omega_strategies import (
    OmegaStrategy, ConservativeOmega, TrendOnlyOmega, get_omega_strategy
)
from ai_strategies import (
    AIQuantStrategy, AITrendFollower, get_ai_strategy
)
from macro_strategies import (
    MacroQuantStrategy, get_macro_strategy
)
from backtester import Backtester, print_backtest_results
from trader import PaperTrader, BinanceTrader, RiskManager, OrderSide
from strategies.base import BaseStrategy, Signal
from main import IchimokuTrendSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot that coordinates all components
    
    Features:
    - Multiple strategies
    - Automated trading loop
    - Risk management
    - Performance tracking
    - Paper/Live mode
    """
    
    def __init__(
        self,
        api_config: APIConfig,
        trading_config: TradingConfig,
        strategies: List[BaseStrategy],
        paper_trading: bool = True,
        initial_capital: float = 10000.0
    ):
        self.api_config = api_config
        self.trading_config = trading_config
        self.strategies = strategies
        self.paper_trading = paper_trading
        
        # Initialize components
        self.data_fetcher = BinanceDataFetcher(api_config)
        self.risk_manager = RiskManager(trading_config)
        
        # Initialize trader (paper or live)
        if paper_trading:
            self.trader = PaperTrader(api_config, trading_config, initial_capital)
            logger.info("🧪 Running in PAPER TRADING mode")
        else:
            self.trader = BinanceTrader(api_config, trading_config)
            logger.info("🔴 Running in LIVE TRADING mode - Real money at risk!")
        
        # State tracking
        self.running = False
        self.last_signals = {}
        self.trade_history = []
        self.performance_log = []
        
        # Data cache
        self.data_cache = {}
    
    def fetch_and_prepare_data(self, symbol: str, bars: int = 500) -> pd.DataFrame:
        """Fetch and prepare data with indicators"""
        try:
            df = self.data_fetcher.get_historical_data(
                symbol=symbol,
                interval=self.trading_config.timeframe,
                days=int(bars * 1.5 / 24)  # Approximate days needed
            )
            
            # Limit to required bars
            df = df.tail(bars)
            
            # Add indicators
            df = DataPreprocessor.add_indicators(df)
            df.name = symbol
            
            # Cache the data
            self.data_cache[symbol] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, symbol: str) -> dict:
        """Generate signals from all strategies for a symbol"""
        df = self.fetch_and_prepare_data(symbol)
        
        if df.empty:
            return {}
        
        signals = {}
        for strategy in self.strategies:
            try:
                signal = strategy.get_signal(df)
                signals[strategy.name] = signal
            except Exception as e:
                logger.error(f"Error generating signal for {strategy.name}: {e}")
        
        return signals
    
    def aggregate_signals(self, signals: dict) -> Signal:
        """
        Aggregate signals from multiple strategies
        Uses voting mechanism with optional weighting
        """
        if not signals:
            return Signal.HOLD
        
        buy_votes = 0
        sell_votes = 0
        
        for strategy_name, signal in signals.items():
            if signal.signal == Signal.BUY:
                buy_votes += signal.confidence
            elif signal.signal == Signal.SELL:
                sell_votes += signal.confidence
        
        threshold = len(signals) * 0.3  # 30% agreement threshold
        
        if buy_votes > threshold and buy_votes > sell_votes:
            return Signal.BUY
        elif sell_votes > threshold and sell_votes > buy_votes:
            return Signal.SELL
        
        return Signal.HOLD
    
    def execute_trade(self, symbol: str, signal: Signal):
        """Execute a trade based on the aggregated signal"""
        if signal == Signal.HOLD:
            return

        # Check risk management
        if not self.risk_manager.check_position_allowed(len(self.trader.positions)):
            logger.warning("Position limit reached, skipping trade")

        if self.risk_manager.should_stop_trading():
            logger.warning("Drawdown limit reached, trading paused")

        # Get current price
        current_price = self.data_fetcher.get_ticker_price(symbol)
        
        # Execute based on signal
        if signal == Signal.BUY:
            quantity = self.trader.calculate_position_size(symbol, OrderSide.BUY)
            if quantity > 0:
                order = self.trader.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity
                )
                
                if order and order.status in ["FILLED", "NEW"]:
                    # Set stop loss and take profit
                    stop_loss = self.risk_manager.calculate_stop_loss(current_price, "long")
                    take_profit = self.risk_manager.calculate_take_profit(current_price, "long")
                    
                    self.trade_history.append({
                        "time": datetime.now(),
                        "symbol": symbol,
                        "side": "BUY",
                        "price": current_price,
                        "quantity": quantity,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    })
                    
                    logger.info(f"✅ BUY {symbol}: {quantity:.6f} @ ${current_price:.2f}")
                    logger.info(f"   SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
                    
        elif signal == Signal.SELL:
            position = self.trader.positions.get(symbol)
            if position:
                order = self.trader.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity
                )
                
                if order and order.status in ["FILLED", "NEW"]:
                    pnl = (current_price - position.entry_price) * position.quantity
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    self.trade_history.append({
                        "time": datetime.now(),
                        "symbol": symbol,
                        "side": "SELL",
                        "price": current_price,
                        "quantity": position.quantity,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct
                    })
                    
                    logger.info(f"💰 SELL {symbol}: {position.quantity:.6f} @ ${current_price:.2f}")
                    logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    
    def check_stop_loss_take_profit(self):
        """Check and execute stop loss / take profit orders"""
        for symbol, position in list(self.trader.positions.items()):
            current_price = self.data_fetcher.get_ticker_price(symbol)
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            # Check stop loss
            if pnl_pct <= -self.trading_config.stop_loss_pct:
                logger.warning(f"⛔ STOP LOSS triggered for {symbol}")
                self.execute_trade(symbol, Signal.SELL)
            
            # Check take profit
            elif pnl_pct >= self.trading_config.take_profit_pct:
                logger.info(f"🎯 TAKE PROFIT triggered for {symbol}")
                self.execute_trade(symbol, Signal.SELL)
    
    def run_trading_cycle(self):
        """Execute one trading cycle"""
        logger.info("\n" + "=" * 50)
        logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        for symbol in self.trading_config.symbols:
            try:
                # Generate signals
                signals = self.generate_signals(symbol)
                
                # Log individual strategy signals
                for name, sig in signals.items():
                    logger.debug(f"{symbol} | {name}: {sig.signal.name}")
                
                # Aggregate signals
                aggregated = self.aggregate_signals(signals)
                
                logger.info(f"{symbol}: {aggregated.name}")
                
                # Execute trade
                self.execute_trade(symbol, aggregated)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Check stop loss / take profit for existing positions
        self.check_stop_loss_take_profit()
        
        # Log portfolio status
        self.log_portfolio_status()
    
    def log_portfolio_status(self):
        """Log current portfolio status"""
        if hasattr(self.trader, 'get_portfolio_value'):
            portfolio_value = self.trader.get_portfolio_value()
        else:
            portfolio_value = self.trader.get_balance("USDT")
        
        self.risk_manager.update_equity(portfolio_value)
        
        logger.info(f"\n📊 Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"   Drawdown: {self.risk_manager.get_current_drawdown()*100:.2f}%")
        logger.info(f"   Open Positions: {len(self.trader.positions)}")
        
        # Log positions
        if self.trader.positions:
            positions_df = self.trader.get_position_summary()
            for _, pos in positions_df.iterrows():
                logger.info(f"   {pos['Symbol']}: {pos['P&L %']:+.2f}%")
        
        # Track performance
        self.performance_log.append({
            "time": datetime.now(),
            "portfolio_value": portfolio_value,
            "positions": len(self.trader.positions),
            "drawdown": self.risk_manager.get_current_drawdown()
        })
    
    def start(self, interval_seconds: int = 3600):
        """Start the trading bot"""
        self.running = True
        
        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\n🛑 Shutting down trading bot...")
            self.running = False
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.info("\n" + "=" * 60)
        logger.info("🚀 CRYPTO QUANT TRADING BOT STARTED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        logger.info(f"Symbols: {self.trading_config.symbols}")
        logger.info(f"Strategies: {[s.name for s in self.strategies]}")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info("=" * 60 + "\n")
        
        while self.running:
            try:
                self.run_trading_cycle()
                
                logger.info(f"\n⏳ Next cycle in {interval_seconds}s...")
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("Trading bot stopped")
    
    def get_performance_report(self) -> pd.DataFrame:
        """Generate performance report"""
        if not self.performance_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_log)
        df.set_index("time", inplace=True)
        return df


def run_backtest(
    symbols: List[str],
    strategies: List[BaseStrategy],
    days: int = 90,
    initial_capital: float = 10000.0,
    timeframe: str = "1h"
):
    """Run backtest on historical data"""
    print("\n" + "=" * 60)
    print("📈 RUNNING BACKTEST")
    print("=" * 60)
    
    api_config = APIConfig(testnet=True)
    fetcher = BinanceDataFetcher(api_config)
    
    # Detect if using HFT strategies
    hft_names = ["HFT_Scalper", "MicroMomentum", "HFT_Combo"]
    pro_names = ["ProfessionalQuant", "VolatilityBreakout", "InstitutionalQuant"]
    elite_names = ["EliteQuant", "AggressiveQuant", "ComboQuant"]
    alpha_names = ["AlphaTrend", "MomentumSurfer", "AlphaCombo"]
    sniper_names = ["Sniper", "TrendRider", "UltimateSniper"]
    omega_names = ["Omega", "ConservativeOmega", "TrendOnlyOmega"]
    ai_names = ["AIQuant", "AITrendFollower"]
    macro_names = ["MacroQuant"]
    
    is_hft = any(s.name in hft_names for s in strategies)
    is_pro = any(s.name in pro_names for s in strategies)
    is_elite = any(s.name in elite_names for s in strategies)
    is_alpha = any(s.name in alpha_names for s in strategies)
    is_sniper = any(s.name in sniper_names for s in strategies)
    is_omega = any(s.name in omega_names for s in strategies)
    is_ai = any(s.name in ai_names for s in strategies)
    is_macro = any(s.name in macro_names for s in strategies)
    
    if is_hft or timeframe in ["1m", "5m"]:
        stop_loss = 0.005   # 0.5% for HFT
        take_profit = 0.01  # 1% for HFT
        position_size = 0.3  # Bigger positions for small moves
    elif is_macro:
        # Macro: Wide stops (ATR-based), big targets (1:2 minimum)
        stop_loss = 0.025   # 2.5% stop (ATRP based)
        take_profit = 0.05  # 5% take profit
        position_size = 0.20  # 20% max (macro adjusts this)
    elif is_ai:
        # AI: Wide stops, let AI decisions play out
        stop_loss = 0.025   # 2.5% stop
        take_profit = 0.05  # 5% take profit (1:2 R/R)
        position_size = 0.25  # 25% position
    elif is_omega:
        # Omega: WIDE stops (don't get shaken out), big targets
        stop_loss = 0.025   # 2.5% stop (WIDE!)
        take_profit = 0.05  # 5% take profit (1:2 R/R)
        position_size = 0.25  # 25% position
    elif is_sniper:
        # Sniper: Tight stop, HUGE target (1:5 R/R)
        stop_loss = 0.012   # 1.2% stop (tight!)
        take_profit = 0.06  # 6% take profit (1:5 R/R)
        position_size = 0.30  # 30% on A+ setups
    elif is_alpha:
        # Alpha: Let winners run, cut losers fast
        stop_loss = 0.015   # 1.5% stop
        take_profit = 0.06  # 6% take profit (1:4 R/R!)
        position_size = 0.25  # 25% position
    elif is_elite or timeframe == "1h":
        # Elite: Tighter stops, better R/R
        stop_loss = 0.015   # 1.5% stop
        take_profit = 0.03  # 3% take profit (1:2 R/R)
        position_size = 0.25  # 25% position
    elif is_pro or timeframe in ["4h", "1d"]:
        # Professional: ATR-based stops, wider ranges
        stop_loss = 0.025   # 2.5% stop (will be overridden by ATR)
        take_profit = 0.05  # 5% take profit
        position_size = 0.20  # 20% position, risk-managed
    else:
        stop_loss = 0.015
        take_profit = 0.045
        position_size = 0.25
    
    backtester = Backtester(
        initial_capital=initial_capital,
        commission=0.001,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    
    all_results = []
    
    for symbol in symbols:
        print(f"\nFetching data for {symbol} ({timeframe} candles)...")
        df = fetcher.get_historical_data(symbol, interval=timeframe, days=days)
        df = DataPreprocessor.add_indicators(df)
        
        for strategy in strategies:
            print(f"\nBacktesting {strategy.name} on {symbol}...")
            result = backtester.run(df, strategy, symbol)
            all_results.append(result)
            print_backtest_results(result)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Crypto Quant Trading Bot")
    parser.add_argument("--mode", choices=["backtest", "paper", "live"], 
                       default="paper", help="Trading mode")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                       help="Trading symbols")
    parser.add_argument("--strategy", default="macro",
                       choices=["ma_crossover", "rsi", "bollinger", "momentum", "macd", "ensemble",
                                "statarb", "ml", "multifactor", "adaptive", "renaissance",
                                "turbo", "scalp", "swing", "ultimate",
                                "hft", "micro", "hftcombo",
                                "pro", "breakout", "institutional",
                                "elite", "aggressive", "combo",
                                "alpha", "surfer", "alphacombo",
                                "sniper", "rider", "ultsniper",
                                "omega", "conservative", "trendonly",
                                "aiquant", "aitrend",
                                "macro"],
                       help="Trading strategy")
    parser.add_argument("--timeframe", default="1h",
                       choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                       help="Candle timeframe (default: 1h)")
    parser.add_argument("--ai-backend", default="ollama",
                       choices=["ollama", "groq", "grok"],
                       help="AI backend: ollama (local), groq (cloud), grok (xAI cloud)")
    parser.add_argument("--interval", type=int, default=3600,
                       help="Trading interval in seconds (default: 3600)")
    parser.add_argument("--capital", type=float, default=10000.0,
                       help="Initial capital (default: 10000)")
    parser.add_argument("--days", type=int, default=90,
                       help="Backtest days (default: 90)")
    
    args = parser.parse_args()
    
    # Setup strategies
    advanced_strategies = ["statarb", "ml", "multifactor", "adaptive", "renaissance"]
    turbo_strategies = ["turbo", "scalp", "swing", "ultimate"]
    hft_strategies = ["hft", "micro", "hftcombo"]
    pro_strategies = ["pro", "breakout", "institutional"]
    elite_strategies = ["elite", "aggressive", "combo"]
    alpha_strategies = ["alpha", "surfer", "alphacombo"]
    sniper_strategies = ["sniper", "rider", "ultsniper"]
    omega_strategies = ["omega", "conservative", "trendonly"]
    ai_strategies = ["aiquant", "aitrend"]
    macro_strategies_list = ["macro"]
    
    if args.strategy == "ensemble":
        strategies = [
            MACrossoverStrategy(),
            RSIMeanReversionStrategy(),
            BollingerBandStrategy(),
            MomentumStrategy(),
            MACDStrategy()
        ]
        strategy_list = [EnsembleStrategy(strategies)]
    elif args.strategy in advanced_strategies:
        strategy_list = [get_advanced_strategy(args.strategy)]
    elif args.strategy in turbo_strategies:
        strategy_list = [get_turbo_strategy(args.strategy)]
    elif args.strategy in hft_strategies:
        strategy_list = [get_hft_strategy(args.strategy)]
    elif args.strategy in pro_strategies:
        strategy_list = [get_pro_strategy(args.strategy)]
    elif args.strategy in elite_strategies:
        strategy_list = [get_elite_strategy(args.strategy)]
    elif args.strategy in alpha_strategies:
        strategy_list = [get_alpha_strategy(args.strategy)]
    elif args.strategy in sniper_strategies:
        strategy_list = [get_sniper_strategy(args.strategy)]
    elif args.strategy in omega_strategies:
        strategy_list = [get_omega_strategy(args.strategy)]
    elif args.strategy in ai_strategies:
        strategy_list = [get_ai_strategy(args.strategy, backend=args.ai_backend)]
    elif args.strategy in macro_strategies_list:
        strategy_list = [get_macro_strategy(args.strategy)]
    else:
        strategy_list = [get_strategy(args.strategy)]
    
    # Setup configs
    api_config = APIConfig(
        api_key="",  # Add your API key
        api_secret="",  # Add your API secret
        testnet=args.mode != "live"
    )
    
    trading_config = TradingConfig(symbols=args.symbols, timeframe=args.timeframe)
    
    if args.mode == "backtest":
        # Run backtest
        run_backtest(
            symbols=args.symbols,
            strategies=strategy_list,
            days=args.days,
            initial_capital=args.capital,
            timeframe=args.timeframe
        )
    else:
        # Run trading bot
        bot = TradingBot(
            api_config=api_config,
            trading_config=trading_config,
            strategies=strategy_list,
            paper_trading=(args.mode == "paper"),
            initial_capital=args.capital
        )
        
        bot.start(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

def get_strategy(name):
    return STRATEGY_MAP.get(name)
