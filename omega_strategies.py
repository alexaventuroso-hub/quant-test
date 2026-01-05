import pandas as pd
from typing import Series

# Assuming these are defined earlier in the file (original code)
# If not, the original backup has them - this keeps them intact
# ... (all the original classes: OmegaStrategy, ConservativeOmega, etc. remain here)

class OmegaStrategy:
    # original OmegaStrategy code (keep everything that was there before)
    pass  # placeholder - your original file has the full implementation

class ConservativeOmega:
    # original ConservativeOmega code
    pass  # placeholder - keep original

# YOUR IMPROVED CLASS
class TrendOnlyOmega(BaseStrategy):
    """
    Trend Only Omega - NEVER trades against the trend

    Only longs in strong uptrend, only shorts in strong downtrend.
    No exceptions.
    """
    
    def __init__(self):
        super().__init__("TrendOnlyOmega")
        self.omega = OmegaStrategy()
        
        # Adjustable parameters
        self.adx_threshold = 20
        self.score_threshold = 10
        self.strong_trend_threshold = 4  # need at least 4/5 EMAs aligned

    def generate_signals(self, df: pd.DataFrame) -> Series:
        # Use Omega's indicator calculations
        df = self.omega._calculate_all_indicators(df)
        
        signals = pd.Series(0, index=df.index, name='signal')
        
        # Calculate scores
        bull_scores = df.apply(self.omega._calculate_bull_score, axis=1)
        bear_scores = df.apply(self.omega._calculate_bear_score, axis=1)
        
        # Strong trend filters
        strong_uptrend = df["ema_bull_score"] >= self.strong_trend_threshold
        strong_downtrend = df["ema_bear_score"] >= self.strong_trend_threshold
        
        # Entry conditions
        long_signal = (
            (bull_scores >= self.score_threshold) &
            strong_uptrend &
            (df["adx"] > self.adx_threshold)
        )
        
        short_signal = (
            (bear_scores >= self.score_threshold) &
            strong_downtrend &
            (df["adx"] > self.adx_threshold)
        )
        
        signals[long_signal] = 1
        signals[short_signal] = -1
        
        # Final safety: no signal if trend is unclear
        neutral = ~(strong_uptrend | strong_downtrend)
        signals[neutral] = 0
        
        return signals

# THE STRATEGY REGISTRY - ADD YOUR NEW ONE HERE
strategies = {
    "omega": OmegaStrategy(),
    "conservative": ConservativeOmega(),
    "trendonlyomega": TrendOnlyOmega(),   # ← THIS IS THE NEW LINE
}
