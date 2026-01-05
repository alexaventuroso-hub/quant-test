
import pandas as pd
import numpy as np

class IchimokuTrendSystem:
    name = "IchimokuTrendSystem"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        high, low, close = df["high"], df["low"], df["close"]

        # === ICHIMOKU ===
        df["tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
        df["kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
        df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
        df["senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        df["chikou"] = close.shift(-26)

        signals = pd.Series(0, index=df.index)

        # === REGIME ===
        cloud_top = df[["senkou_a", "senkou_b"]].max(axis=1)
        cloud_bot = df[["senkou_a", "senkou_b"]].min(axis=1)

        bull_regime = (
            (close > cloud_top) &
            (df["senkou_a"] > df["senkou_b"]) &
            (df["chikou"] > close.shift(26))
        )

        # === ENTRY: pullback to kijun in bull regime ===
        pullback = close <= df["kijun"]
        entry = bull_regime & pullback & (~bull_regime.shift(1).fillna(False))

        # === EXIT: lose kijun or enter cloud ===
        exit_ = (close < df["kijun"]) | ((close < cloud_top) & (close > cloud_bot))

        for i in range(len(df)):
            if entry.iloc[i]:
                signals.iloc[i] = 1
            elif exit_.iloc[i]:
                signals.iloc[i] = -1

        return signals
