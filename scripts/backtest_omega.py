import argparse
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import pandas as pd

from backtester import Backtester, print_backtest_results
from config import APIConfig
from data_fetcher import BinanceDataFetcher
from omega_strategies import OmegaStrategy

try:
    import ccxt
except ImportError:
    ccxt = None

def _utc_ms(dt):
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def main():
    print('Runner installed correctly')

if __name__ == '__main__':
    main()
