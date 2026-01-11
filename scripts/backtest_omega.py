#!/usr/bin/env python3
import argparse
import inspect
import importlib.util
from datetime import datetime, timedelta, timezone

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from backtester import Backtester, print_backtest_results
from config import APIConfig
from data_fetcher import BinanceDataFetcher
try:
    from grok_quant import GrokQuantClient, GrokQuantError
except ImportError:  # optional dependency
    GrokQuantClient = None
    GrokQuantError = Exception
from omega_strategies import OmegaStrategy

ccxt = None
if importlib.util.find_spec("ccxt"):
    import ccxt  # type: ignore[no-redef]

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


def _fetch_data_binance(symbol: str, timeframe: str, days: int, market: str) -> pd.DataFrame:
    api_config = APIConfig(testnet=False)
    fetcher = BinanceDataFetcher(api_config)
    df = fetcher.get_historical_data(
        symbol=symbol,
        interval=timeframe,
        days=days,
        
    )
    return df.sort_index()


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _fetch_data(
    symbol: str,
    timeframe: str,
    days: int,
    use_ccxt: bool,
    market: str,
) -> pd.DataFrame:
    if use_ccxt:
        return _fetch_data_ccxt(symbol, timeframe, days)
    return _fetch_data_binance(_normalize_symbol(symbol), timeframe, days, market)


def _summarize(result, label: str) -> dict:
    metrics = result.metrics

    pnls = [t.pnl for t in result.trades] if getattr(result, "trades", None) else []
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]
    avg_win = (sum(winning) / len(winning)) if winning else 0.0
    avg_loss = (sum(losing) / len(losing)) if losing else 0.0
    avg_win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss else None

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


def _ensure_generate_signals(strategy):
    if hasattr(strategy, "generate_signals") and callable(strategy.generate_signals):
        return strategy

    if not hasattr(strategy, "get_signal") or not callable(strategy.get_signal):
        raise SystemExit("Strategy must implement generate_signals or get_signal.")

    class _StrategyAdapter:
        def __init__(self, base):
            self.base = base
            self.name = getattr(base, "name", base.__class__.__name__)

        def generate_signals(self, df):
            import pandas as pd

            signals = pd.Series(0, index=df.index, dtype=int)
            for i in range(len(df)):
                sig = self.base.get_signal(df.iloc[: i + 1])
                value = 0
                if hasattr(sig, "signal"):
                    value = getattr(sig.signal, "value", 0)
                else:
                    try:
                        value = int(sig)
                    except Exception:
                        value = 0
                signals.iat[i] = 1 if value > 0 else (-1 if value < 0 else 0)
            return signals

        def get_signal(self, df):
            return self.base.get_signal(df)

    return _StrategyAdapter(strategy)


def _load_macro_data(path: str, date_column: str) -> pd.DataFrame:
    macro = pd.read_csv(path)
    if date_column not in macro.columns:
        raise SystemExit(f"Macro CSV missing date column: {date_column}")
    macro[date_column] = pd.to_datetime(macro[date_column], utc=True)
    macro = macro.set_index(date_column).sort_index()
    return macro


def _apply_macro_filter(signals: pd.Series, df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return signals
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing macro columns in data: {missing}")
    mask = df[columns].ffill().fillna(0).astype(bool).all(axis=1)
    filtered = signals.copy()
    filtered.loc[~mask] = 0
    return filtered


def _macro_summary(df: pd.DataFrame, columns: list[str]) -> dict:
    if not columns:
        return {}
    summary = {}
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            summary[col] = {"latest": None, "mean": None, "std": None}
            continue
        summary[col] = {
            "latest": float(series.iloc[-1]),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    return summary


def run_backtest(
    symbol: str,
    timeframe: str,
    days: int,
    capital: float,
    fee_rate: float,
    atr_trailing: bool,
    atr_mult: float,
    use_ccxt: bool,
    market: str,
    macro_csv: str | None,
    macro_date_column: str,
    macro_filter_cols: list[str],
    grok_client,
):
    df = _fetch_data(symbol, timeframe, days, use_ccxt, market)
    if df is None or df.empty or len(df) < 300:
        raise SystemExit(f"Not enough candles fetched: {0 if df is None else len(df)}")

    if macro_csv:
        macro = _load_macro_data(macro_csv, macro_date_column)
        df = df.join(macro, how="left")

    strategy = _ensure_generate_signals(OmegaStrategy())

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

    if macro_filter_cols:
        class MacroFilteredStrategy:
            def __init__(self, base, cols):
                self.base = base
                self.name = getattr(base, "name", base.__class__.__name__)
                self.cols = cols

            def generate_signals(self, data):
                signals = self.base.generate_signals(data)
                return _apply_macro_filter(signals, data, self.cols)

            def get_signal(self, data):
                return self.base.get_signal(data)

        strategy = MacroFilteredStrategy(strategy, macro_filter_cols)

    result = backtester.run(df, strategy, symbol=_normalize_symbol(symbol))
    print_backtest_results(result)

    output_path = f"equity_curve_{_normalize_symbol(symbol).lower()}_{timeframe}.png"
    _plot_equity_curve(result, output_path)
    summary = _summarize(result, f"{symbol} {timeframe}")

    if grok_client:
        macro_info = _macro_summary(df, macro_filter_cols)
        try:
            advice = grok_client.generate_trade_review(
                symbol=_normalize_symbol(symbol),
                timeframe=timeframe,
                metrics=summary,
                macro_summary=macro_info,
            )
            print("\nGrok quant review:")
            print(advice)
        except GrokQuantError as exc:
            print(f"⚠️ Grok request failed: {exc}")
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
    parser.add_argument("--market", default="spot", choices=["spot", "futures"])
    parser.add_argument("--macro_csv", default=None)
    parser.add_argument("--macro_date_column", default="date")
    parser.add_argument("--macro_filter_cols", default="")
    parser.add_argument("--grok_api_key", default=None)
    args = parser.parse_args()

    summaries = []
    plots = []

    macro_filter_cols = [c.strip() for c in args.macro_filter_cols.split(",") if c.strip()]

    grok_client = None
    if args.grok_api_key:
        if GrokQuantClient is None:
            raise SystemExit("grok_quant.py not found. Remove --grok_api_key or add grok_quant.py.")
        grok_client = GrokQuantClient(api_key=args.grok_api_key)

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
            args.market,
            args.macro_csv,
            args.macro_date_column,
            macro_filter_cols,
            None,
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
