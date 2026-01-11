#!/usr/bin/env python3
"""
BITGET FUTURES TRADER - RELAXED VERSION
More trades, slightly lower win rate
"""
import os
import sys
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

class BitgetTrader:
    def __init__(self, symbol="ETHUSDT", leverage=50, capital_pct=0.30):
        self.symbol = symbol
        self.product_type = "USDT-FUTURES"
        self.leverage = leverage
        self.capital_pct = capital_pct
        
        self.base_url = "https://api.bitget.com"
        
        self.api_key = os.getenv("BITGET_API_KEY")
        self.api_secret = os.getenv("BITGET_API_SECRET")
        self.passphrase = os.getenv("BITGET_PASSPHRASE", "")
        
        if not self.api_key or not self.api_secret:
            print("Missing API keys!")
            sys.exit(1)
        
        # RELAXED parameters for more trades
        self.params = {
            'zscore': 0.8,
            'rsi_low': 30,      # Was 25, now 30 (easier to trigger)
            'rsi_high': 70,     # Was 75, now 70
            'bb_low': 0.15,     # Was 0.1
            'bb_high': 0.85,    # Was 0.9
            'atr_sl': 1.5,
            'atr_tp': 6.0,
            'adx_max': 35,      # Was 20, now 35 (trade in mild trends too)
        }
        
        self.position = None
    
    def _timestamp(self):
        return str(int(time.time() * 1000))
    
    def _sign(self, timestamp, method, endpoint, body=""):
        message = timestamp + method.upper() + endpoint + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _headers(self, method, endpoint, body=""):
        timestamp = self._timestamp()
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._sign(timestamp, method, endpoint, body),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US"
        }
    
    def _request(self, method, endpoint, params=None, signed=False):
        url = self.base_url + endpoint
        body = ""
        
        try:
            if method == "GET":
                if params:
                    query = "&".join([f"{k}={v}" for k, v in params.items()])
                    endpoint_with_query = endpoint + "?" + query
                    url = self.base_url + endpoint_with_query
                    headers = self._headers(method, endpoint_with_query) if signed else {"Content-Type": "application/json"}
                else:
                    headers = self._headers(method, endpoint) if signed else {"Content-Type": "application/json"}
                r = requests.get(url, headers=headers, timeout=10)
            else:
                import json
                body = json.dumps(params) if params else ""
                headers = self._headers(method, endpoint, body) if signed else {"Content-Type": "application/json"}
                r = requests.post(url, headers=headers, data=body, timeout=10)
            
            data = r.json()
            if data.get('code') == '00000' or data.get('code') == 0:
                return data.get('data', data)
            else:
                print(f"API Error: {data}")
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def get_account(self):
        return self._request("GET", "/api/v2/mix/account/account",
                            {"symbol": "ETHUSDT", "productType": self.product_type, "marginCoin": "USDT"}, signed=True)
    
    def get_balance(self):
        account = self.get_account()
        if account:
            return float(account.get('available', 0))
        return 0
    
    def get_position(self):
        result = self._request("GET", "/api/v2/mix/position/single-position",
                              {"symbol": "ETHUSDT", "productType": self.product_type, "marginCoin": "USDT"}, signed=True)
        if result and len(result) > 0:
            for pos in result:
                size = float(pos.get('total', 0))
                if size > 0:
                    return {
                        'side': pos.get('holdSide', '').upper(),
                        'size': size,
                        'entry': float(pos.get('openPriceAvg', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPL', 0)),
                        'liquidation': float(pos.get('liquidationPrice', 0))
                    }
        return None
    
    def set_leverage(self):
        for side in ['long', 'short']:
            self._request("POST", "/api/v2/mix/account/set-leverage",
                         {"symbol": "ETHUSDT", "productType": self.product_type,
                          "marginCoin": "USDT", "leverage": str(self.leverage),
                          "holdSide": side}, signed=True)
        print(f"Leverage set to {self.leverage}x")
    
    def fetch_klines(self, limit=100):
        result = self._request("GET", "/api/v2/mix/market/candles",
                              {"symbol": "ETHUSDT", "productType": self.product_type,
                               "granularity": "15m", "limit": str(limit)})
        if not result:
            return None
        
        df = pd.DataFrame(result, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'quote_vol'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df.set_index('ts', inplace=True)
        return df
    
    def calc_indicators(self, df):
        df = df.copy()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['zscore'] = (df['close'] - df['bb_mid']) / (df['bb_std'] + 1e-10)
        
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(),
                       (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        df['plus_di'] = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['minus_di'] = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        df['adx'] = (100 * (df['plus_di'] - df['minus_di']).abs() / 
                    (df['plus_di'] + df['minus_di'] + 1e-10)).rolling(14).mean()
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        
        return df
    
    def get_signal(self, df):
        row = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check ADX (relaxed to 35)
        if row['adx'] >= self.params['adx_max']:
            return 0, f"Strong trend (ADX={row['adx']:.0f} > 35)"
        
        # LONG signal
        if (row['zscore'] < -self.params['zscore'] and 
            row['rsi'] < self.params['rsi_low'] and 
            row['bb_pct'] < self.params['bb_low']):
            return 1, f"LONG: Z={row['zscore']:.2f}, RSI={row['rsi']:.0f}, ADX={row['adx']:.0f}"
        
        # SHORT signal
        if (row['zscore'] > self.params['zscore'] and 
            row['rsi'] > self.params['rsi_high'] and 
            row['bb_pct'] > self.params['bb_high']):
            return -1, f"SHORT: Z={row['zscore']:.2f}, RSI={row['rsi']:.0f}, ADX={row['adx']:.0f}"
        
        return 0, f"Waiting (Z={row['zscore']:.2f}, RSI={row['rsi']:.0f}, ADX={row['adx']:.0f})"
    
    def place_order(self, side, size, sl_price=None, tp_price=None):
        params = {
            "symbol": "ETHUSDT",
            "productType": self.product_type,
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "side": side,
            "tradeSide": "open",
            "orderType": "market",
            "size": str(round(size, 4)),
        }
        if sl_price:
            params["presetStopLossPrice"] = str(round(sl_price, 2))
        if tp_price:
            params["presetTakeProfitPrice"] = str(round(tp_price, 2))
        
        result = self._request("POST", "/api/v2/mix/order/place-order", params, signed=True)
        if result:
            print(f"ORDER PLACED: {side.upper()} {size}")
            return result
        return None
    
    def close_position(self):
        pos = self.get_position()
        if not pos:
            return None
        
        close_side = "sell" if pos['side'] == 'LONG' else "buy"
        params = {
            "symbol": "ETHUSDT",
            "productType": self.product_type,
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "side": close_side,
            "tradeSide": "close",
            "orderType": "market",
            "size": str(pos['size']),
        }
        result = self._request("POST", "/api/v2/mix/order/place-order", params, signed=True)
        if result:
            print(f"CLOSED: {pos['side']} {pos['size']}")
        return result
    
    def run_cycle(self):
        now = datetime.now(timezone.utc)
        
        df = self.fetch_klines(100)
        if df is None or len(df) < 50:
            print("Failed to fetch data")
            return
        
        df = self.calc_indicators(df)
        row = df.iloc[-1]
        price = row['close']
        atr = row['atr']
        
        signal, reason = self.get_signal(df)
        balance = self.get_balance()
        pos = self.get_position()
        
        print(f"\n{'='*60}")
        print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC | ETHUSDT 15m")
        print(f"{'='*60}")
        print(f"Balance: ${balance:.2f} | Price: ${price:.2f}")
        print(f"RSI: {row['rsi']:.0f} | Z: {row['zscore']:.2f} | ADX: {row['adx']:.0f} | MFI: {row['mfi']:.0f}")
        print(f"Signal: {reason}")
        
        if pos:
            pct = ((price / pos['entry']) - 1) * 100 if pos['side'] == 'LONG' else ((pos['entry'] / price) - 1) * 100
            pct_lev = pct * self.leverage
            print(f"\nPOSITION: {pos['side']} {pos['size']} @ ${pos['entry']:.2f}")
            print(f"P&L: ${pos['unrealized_pnl']:.2f} ({pct_lev:+.1f}%)")
            
            if (pos['side'] == 'LONG' and signal == -1) or (pos['side'] == 'SHORT' and signal == 1):
                print(f"\nREVERSING...")
                self.close_position()
                time.sleep(1)
                pos = None
        
        if pos is None and signal != 0:
            trade_capital = balance * self.capital_pct
            position_value = trade_capital * self.leverage
            size = position_value / price
            
            if signal == 1:
                sl_price = price - self.params['atr_sl'] * atr
                tp_price = price + self.params['atr_tp'] * atr
                order_side = "buy"
            else:
                sl_price = price + self.params['atr_sl'] * atr
                tp_price = price - self.params['atr_tp'] * atr
                order_side = "sell"
            
            print(f"\n>>> OPENING {'LONG' if signal == 1 else 'SHORT'} <<<")
            print(f"Size: {size:.4f} ETH (${position_value:.2f})")
            print(f"SL: ${sl_price:.2f} | TP: ${tp_price:.2f}")
            
            self.place_order(order_side, size, sl_price, tp_price)
        
        print(f"{'─'*60}")
    
    def run(self):
        print(f"\n{'='*60}")
        print("BITGET TRADER - RELAXED VERSION")
        print(f"{'='*60}")
        print(f"ETHUSDT 15m | {self.leverage}x leverage")
        print(f"Triggers: RSI <30/>70, Z <-0.8/>0.8, ADX <35")
        print(f"{'='*60}")
        
        self.set_leverage()
        balance = self.get_balance()
        print(f"Balance: ${balance:.2f}")
        
        pos = self.get_position()
        if pos:
            print(f"Position: {pos['side']} {pos['size']} @ ${pos['entry']:.2f}")
        
        print(f"\nRunning... Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as e:
                    print(f"Error: {e}")
                
                now = datetime.now()
                mins = 15 - (now.minute % 15)
                secs = mins * 60 - now.second + 5
                print(f"Next: {mins}m {60 - now.second}s")
                time.sleep(secs)
                
        except KeyboardInterrupt:
            print(f"\n\nShutting down...")
            pos = self.get_position()
            if pos:
                print(f"Open: {pos['side']} {pos['size']}")
                if input("Close? (y/n): ").lower() == 'y':
                    self.close_position()
            print("Bye!")

if __name__ == "__main__":
    trader = BitgetTrader(symbol="ETHUSDT", leverage=50, capital_pct=0.30)
    trader.run()
