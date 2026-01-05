"""
Backtesting Engine - Historical Strategy Testing
Simulates trading performance with realistic assumptions
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from strategies import BaseStrategy, Signal


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    status: str = "open"  # "open", "closed", "stopped"
    
    def close(self, exit_price: float, exit_time: datetime, commission_rate: float = 0.001):
        """Close the trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "closed"
        
        # Calculate P&L
        if self.side == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        # Deduct commission
        self.commission = (self.entry_price + exit_price) * self.quantity * commission_rate
        self.pnl -= self.commission


@dataclass
class BacktestResult:
    """Contains all backtesting results and metrics"""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime


class Backtester:
    """
    Backtesting engine for strategy evaluation
    
    Features:
    - Realistic commission and slippage modeling
    - Position sizing based on risk
    - Stop loss and take profit
    - Detailed performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        position_size: float = 0.1,  # 10% of capital per trade
        stop_loss: Optional[float] = 0.02,  # 2%
        take_profit: Optional[float] = 0.04,  # 4%
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def run(
        self, 
        df: pd.DataFrame, 
        strategy: BaseStrategy,
        symbol: str = "UNKNOWN"
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            df: OHLCV DataFrame with datetime index
            strategy: Trading strategy to test
            symbol: Trading pair symbol
        
        Returns:
            BacktestResult with trades and metrics
        """
        # Generate signals
        signals = strategy.generate_signals(df)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0.0  # Current position size
        entry_price = 0.0
        entry_time = None
        trades = []
        equity_curve = []
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_price = row["close"]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Track equity
            if position > 0:
                unrealized_pnl = (current_price - entry_price) * position
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)
            
            # Check stop loss / take profit for open positions
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Stop loss hit
                if self.stop_loss and pnl_pct <= -self.stop_loss:
                    exit_price = entry_price * (1 - self.stop_loss) * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position
                    commission = (entry_price + exit_price) * position * self.commission
                    capital += pnl - commission
                    
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        symbol=symbol,
                        side="long",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=position,
                        pnl=pnl - commission,
                        pnl_pct=pnl_pct,
                        commission=commission,
                        status="stopped"
                    )
                    trades.append(trade)
                    position = 0.0
                    continue
                
                # Take profit hit
                if self.take_profit and pnl_pct >= self.take_profit:
                    exit_price = entry_price * (1 + self.take_profit) * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position
                    commission = (entry_price + exit_price) * position * self.commission
                    capital += pnl - commission
                    
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        symbol=symbol,
                        side="long",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=position,
                        pnl=pnl - commission,
                        pnl_pct=pnl_pct,
                        commission=commission,
                        status="closed"
                    )
                    trades.append(trade)
                    position = 0.0
                    continue
            
            # Process signals
            if signal > 0 and position == 0:  # Buy signal, no position
                # Calculate position size
                trade_capital = capital * self.position_size
                entry_price = current_price * (1 + self.slippage)  # Slippage on entry
                position = trade_capital / entry_price
                entry_time = timestamp
                
            elif signal < 0 and position > 0:  # Sell signal, have position
                exit_price = current_price * (1 - self.slippage)  # Slippage on exit
                pnl = (exit_price - entry_price) * position
                pnl_pct = (exit_price - entry_price) / entry_price
                commission = (entry_price + exit_price) * position * self.commission
                capital += pnl - commission
                
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=timestamp,
                    symbol=symbol,
                    side="long",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=position,
                    pnl=pnl - commission,
                    pnl_pct=pnl_pct,
                    commission=commission,
                    status="closed"
                )
                trades.append(trade)
                position = 0.0
        
        # Close any remaining position at the end
        if position > 0:
            final_price = df["close"].iloc[-1] * (1 - self.slippage)
            pnl = (final_price - entry_price) * position
            pnl_pct = (final_price - entry_price) / entry_price
            commission = (entry_price + final_price) * position * self.commission
            capital += pnl - commission
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=df.index[-1],
                symbol=symbol,
                side="long",
                entry_price=entry_price,
                exit_price=final_price,
                quantity=position,
                pnl=pnl - commission,
                pnl_pct=pnl_pct,
                commission=commission,
                status="closed"
            )
            trades.append(trade)
        
        # Create equity curve series
        equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_series, returns, df)
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            returns=returns,
            metrics=metrics,
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=df.index[0],
            end_date=df.index[-1]
        )
    
    def _calculate_metrics(
        self, 
        trades: List[Trade], 
        equity: pd.Series, 
        returns: pd.Series,
        df: pd.DataFrame
    ) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {"error": "No trades executed"}
        
        # Basic metrics
        total_pnl = sum(t.pnl for t in trades)
        total_commission = sum(t.commission for t in trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(equity)
        
        # Sharpe Ratio (annualized, assuming hourly data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe = 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(24 * 365)
        else:
            sortino = 0
        
        # Calmar Ratio
        annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (365 * 24 / len(equity)) - 1
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Profit Factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Buy & Hold comparison
        buy_hold_return = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
        strategy_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Average trade duration
        durations = []
        for t in trades:
            if t.exit_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            # Returns
            "total_return": strategy_return,
            "total_return_pct": strategy_return * 100,
            "annual_return": annual_return,
            "buy_hold_return": buy_hold_return,
            "alpha": strategy_return - buy_hold_return,
            
            # Trade statistics
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade_duration_hours": avg_duration,
            
            # Risk metrics
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "profit_factor": profit_factor,
            
            # P&L
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "net_pnl": total_pnl,
            "final_capital": equity.iloc[-1] if len(equity) > 0 else self.initial_capital,
            
            # Expectancy
            "expectancy": (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)),
        }
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown.min()


def print_backtest_results(result: BacktestResult):
    """Pretty print backtest results"""
    m = result.metrics
    
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {result.strategy_name}")
    print(f"Symbol: {result.symbol}")
    print(f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    print("\n📈 RETURNS")
    print(f"  Total Return:      {m['total_return_pct']:>10.2f}%")
    print(f"  Annual Return:     {m['annual_return']*100:>10.2f}%")
    print(f"  Buy & Hold:        {m['buy_hold_return']*100:>10.2f}%")
    print(f"  Alpha:             {m['alpha']*100:>10.2f}%")
    
    print("\n📊 TRADE STATISTICS")
    print(f"  Total Trades:      {m['total_trades']:>10}")
    print(f"  Win Rate:          {m['win_rate']*100:>10.2f}%")
    print(f"  Avg Win:           ${m['avg_win']:>10.2f}")
    print(f"  Avg Loss:          ${m['avg_loss']:>10.2f}")
    print(f"  Profit Factor:     {m['profit_factor']:>10.2f}")
    print(f"  Expectancy:        ${m['expectancy']:>10.2f}")
    
    print("\n⚠️  RISK METRICS")
    print(f"  Max Drawdown:      {m['max_drawdown_pct']:>10.2f}%")
    print(f"  Sharpe Ratio:      {m['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:     {m['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:      {m['calmar_ratio']:>10.2f}")
    
    print("\n💰 P&L SUMMARY")
    print(f"  Starting Capital:  ${result.equity_curve.iloc[0]:>10,.2f}")
    print(f"  Final Capital:     ${m['final_capital']:>10,.2f}")
    print(f"  Total P&L:         ${m['total_pnl']:>10,.2f}")
    print(f"  Total Commission:  ${m['total_commission']:>10,.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test backtester with sample data
    from strategies import MACrossoverStrategy, RSIMeanReversionStrategy, EnsembleStrategy
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=2000, freq="1h")
    
    # Simulate realistic price movement
    returns = np.random.randn(2000) * 0.01 + 0.0001  # Slight upward bias
    price = 40000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": price * (1 + np.random.randn(2000) * 0.001),
        "high": price * (1 + np.abs(np.random.randn(2000) * 0.005)),
        "low": price * (1 - np.abs(np.random.randn(2000) * 0.005)),
        "close": price,
        "volume": np.random.randint(100, 1000, 2000)
    }, index=dates)
    
    # Test different strategies
    strategies = [
        MACrossoverStrategy(fast_period=12, slow_period=26),
        RSIMeanReversionStrategy(rsi_period=14),
    ]
    
    backtester = Backtester(
        initial_capital=10000,
        commission=0.001,
        position_size=0.2,
        stop_loss=0.03,
        take_profit=0.05
    )
    
    for strategy in strategies:
        result = backtester.run(df, strategy, symbol="BTCUSDT")
        print_backtest_results(result)
