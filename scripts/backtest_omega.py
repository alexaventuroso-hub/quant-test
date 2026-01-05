#!/usr/bin/env python3
import argparse
import inspect
from datetime import datetime, timedelta, timezone

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from backtester import Backtester, print_backtest_results
from config import APIConfig
from data_fetcher import BinanceDataFetcher
from omega_strategies import OmegaStrategy

try:
    import ccxt  # optional
except ImportError:
    ccxt = None

matplotlib.use("Agg")


def _utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_data_ccxt(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    if ccxt is None:
        raise SystemExit("Install ccxt first: python3 -m pip install ccxt")

    ex = ccxt.binance({"enableRateLimit": True})
    rows = []
    since = datetime.utcnow() - timedelta(days=days)
    cur = _utc_ms(since)
    now_ms = _utc_ms(datetime.utcnow())

    while cur < now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 1
        if len(rows) > 500000:
            break

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_data_binance(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    api_config = APIConfig(testnet=False)
    fetcher = BinanceDataFetcher(api_config)
    df = fetcher.get_historical_data(symbol=symbol, interval=timeframe, days=days)
    return df.sort_index()


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _fetch_data(symbol: str, timeframe: str, days: int, use_ccxt: bool) -> pd.DataFrame:
    if use_ccxt:
        return _fetch_data_ccxt(symbol, timeframe, days)
    return _fetch_data_binance(_normalize_symbol(symbol), timeframe, days)


def _summarize(result, label: str) -> dict:
    metrics = result.metrics

    pnls = [t.pnl for t in result.trades] if getattr(result, "trades", None) else []
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]
    avg_win = (sum(winning) / len(winning)) if winning else 0.0
    avg_loss = (sum(losing) / len(losing)) if losing else 0.0
    avg_win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss else float("inf")

    return {
        "label": label,
        "total_return_pct": metrics.get("total_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "win_rate_pct": (metrics.get("win_rate") * 100) if metrics.get("win_rate") is not None else None,
        "profit_factor": metrics.get("profit_factor"),
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "total_trades": metrics.get("total_trades"),
        "final_capital": metrics.get("final_capital"),
    }


def _plot_equity_curve(result, output_path: str) -> None:
    plt.figure(figsize=(10, 4))
    result.equity_curve.plot()
    plt.title(f"Equity Curve: {result.strategy_name} ({result.symbol})")
    plt.xlabel("Time")
    plt.ylabel("Equity (USDT)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_backtest(
    symbol: str,
    timeframe: str,
    days: int,
    capital: float,
    fee_rate: float,
    atr_trailing: bool,
    atr_mult: float,
    use_ccxt: bool,
):
    df = _fetch_data(symbol, timeframe, days, use_ccxt)
    if df is None or df.empty or len(df) < 300:
        raise SystemExit(f"Not enough candles fetched: {0 if df is None else len(df)}")

    strategy = OmegaStrategy()

    sig = inspect.signature(Backtester.__init__)
    allowed = set(sig.parameters.keys())

    bt_kwargs = {"initial_capital": capital}
    if "commission" in allowed:
        bt_kwargs["commission"] = fee_rate
    if "fee_rate" in allowed:
        bt_kwargs["fee_rate"] = fee_rate
    if "slippage" in allowed:
        bt_kwargs["slippage"] = 0.0
    if "position_size" in allowed:
        bt_kwargs["position_size"] = 1.0
    if "stop_loss" in allowed:
        bt_kwargs["stop_loss"] = None
    if "take_profit" in allowed:
        bt_kwargs["take_profit"] = None

    atr_supported = False
    if atr_trailing:
        if "atr_trailing_stop" in allowed:
            bt_kwargs["atr_trailing_stop"] = atr_mult
            atr_supported = True
        elif "atr_trailing" in allowed:
            bt_kwargs["atr_trailing"] = atr_mult
            atr_supported = True
        elif "atr_stop_mult" in allowed:
            bt_kwargs["atr_stop_mult"] = atr_mult
            atr_supported = True

        if "atr_period" in allowed:
            bt_kwargs["atr_period"] = 14
        elif "atr_length" in allowed:
            bt_kwargs["atr_length"] = 14

    if atr_trailing and not atr_supported:
        print("⚠️ ATR trailing requested, but Backtester does not support it. Running without ATR trailing.")

    backtester = Backtester(**bt_kwargs)

    result = backtester.run(df, strategy, symbol=_normalize_symbol(symbol))
    print_backtest_results(result)

    output_path = f"equity_curve_{_normalize_symbol(symbol).lower()}_{timeframe}.png"
    _plot_equity_curve(result, output_path)
    summary = _summarize(result, f"{symbol} {timeframe}")
    return summary, output_path


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Backtest OmegaStrategy on Binance data.")
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--timeframes", default="15m,5m")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--fee_rate", type=float, default=0.0004, help="0.04% per side")
    parser.add_argument("--atr_trailing", action="store_true")
    parser.add_argument("--atr_mult", type=float, default=2.5)
    parser.add_argument("--use_ccxt", action="store_true")
    args = parser.parse_args()

    summaries = []
    plots = []

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    for tf in timeframes:
        summary, plot_path = run_backtest(
            args.symbol,
            tf,
            args.days,
            args.capital,
            args.fee_rate,
            args.atr_trailing,
            args.atr_mult,
            args.use_ccxt,
        )
        summaries.append(summary)
        plots.append(plot_path)

    print("\nSummary")
    for s in summaries:
        print(
            f"{s['label']}: "
            f"Return {_fmt(s['total_return_pct'])}%, "
            f"Max DD {_fmt(s['max_drawdown_pct'])}%, "
            f"Win Rate {_fmt(s['win_rate_pct'])}%, "
            f"Profit Factor {_fmt(s['profit_factor'])}, "
            f"Avg Win/Loss {_fmt(s['avg_win_loss_ratio'])}, "
            f"Trades {s['total_trades']}"
        )

    print("\nEquity curve plots:")
    for p in plots:
        print(f"- {p}")


if __name__ == "__main__":
    main()
