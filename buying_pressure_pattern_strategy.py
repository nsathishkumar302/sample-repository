"""
Buying Pressure EMA Crossover Strategy with Volume/Trades Z‑Score Patterns
--------------------------------------------------------------------------

This module implements a trading strategy based on an exponential moving
average (EMA) crossover of the `BuyingPressure` series and applies
additional filters using volume and trades z-score patterns, premium
importance, and z-score category.  It can be run against a single stock
CSV file to back-test historical performance and to flag the latest
row with a pattern signal for live trading.

Key features:

* Computes 5‑period and 21‑period EMAs of `BuyingPressure` and
  generates a signal whenever the short EMA crosses above the long EMA
  (bullish crossover)【319193584195864†L189-L203】.
* Categorizes `VolumeZscore` and `TradesZscore` into four ranks:
  0 (non‑positive), 1 (lower positive tertile), 2 (middle positive
  tertile), and 3 (upper positive tertile), to form a two‑digit
  pattern code.
* Applies pattern‑specific filters based on historical analysis:
  - Pattern 02: exclude `FuturePremium_IMP == 'High'` and require
    `ZscoreCategory == 'Low / Low / Low'`.
  - Pattern 21: exclude `FuturePremium_IMP == 'Low'` and require
    `ZscoreCategory` in {`Low / Low / Low`, `Low / Medium / Low`,
    `Medium / Low / Low`}.
  - Pattern 11: exclude `FuturePremium_IMP == 'Low'` and require
    `ZscoreCategory == 'Low / Medium / Low'`.
  - Pattern 33: exclude `FuturePremium_IMP == 'Very High'` and
    automatically determine high‑success `ZscoreCategory` values by
    computing success rates from historical data.
* Back‑tests each pattern by entering a trade at the next day's open
  following a signal, exiting at a 3 % profit or after 10 trading
  days.  Records the number of trades, success ratio, average return
  and drawdown statistics.
* Adds a `PatternSignal` column to the input dataframe marking
  today's row with the pattern code if it meets all filter
  conditions, enabling live monitoring.

Usage example (run from command line):

```
python buying_pressure_pattern_strategy.py \
    --file /path/to/stockdata_AUROPHARMA_Analyzed_All.csv
```

The script prints a summary of the back‑tested performance for each
pattern and writes an updated CSV file alongside the original with
`_updated` appended to the filename.
"""

import argparse
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PatternFilter:
    """Structure defining filters for a z‑score pattern."""

    exclude_premium: str  # premium category to exclude
    allowed_categories: Optional[List[str]]  # list of allowed ZscoreCategory values


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute an exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def categorize_zscore(z: float, q1: float, q2: float) -> int:
    """Assign a categorical rank based on quantiles.

    0 for non‑positive values, 1 for values up to the 33rd percentile,
    2 for values up to the 66th percentile, 3 for anything above that.
    """
    if z <= 0:
        return 0
    if z <= q1:
        return 1
    if z <= q2:
        return 2
    return 3


def detect_signals(df: pd.DataFrame, fast: int, slow: int) -> List[int]:
    """Identify indices where the fast EMA crosses above the slow EMA."""
    ema_fast = compute_ema(df['BuyingPressure'], span=fast)
    ema_slow = compute_ema(df['BuyingPressure'], span=slow)
    macd = ema_fast - ema_slow
    # A bullish crossover occurs when the difference crosses from
    # non‑positive to positive【319193584195864†L189-L203】.
    signals = df.index[(macd.shift(1) <= 0) & (macd > 0)].tolist()
    return signals


def compute_pattern_categories(df: pd.DataFrame) -> None:
    """Compute volume/trades z‑score categories (0–3) and a pattern code.

    Adds `Vol_cat`, `Trade_cat`, and `pattern` columns to df.
    """
    # Only consider positive z‑scores when computing tertiles
    vol_pos = df.loc[df['VolumeZscore'] > 0, 'VolumeZscore']
    trade_pos = df.loc[df['TradesZscore'] > 0, 'TradesZscore']
    # Compute tertile thresholds
    vol_q1, vol_q2 = (vol_pos.quantile(0.33), vol_pos.quantile(0.66)) if not vol_pos.empty else (0.0, 0.0)
    trade_q1, trade_q2 = (trade_pos.quantile(0.33), trade_pos.quantile(0.66)) if not trade_pos.empty else (0.0, 0.0)
    # Assign categories
    df['Vol_cat'] = df['VolumeZscore'].apply(lambda z: categorize_zscore(z, vol_q1, vol_q2))
    df['Trade_cat'] = df['TradesZscore'].apply(lambda z: categorize_zscore(z, trade_q1, trade_q2))
    df['pattern'] = df['Vol_cat'].astype(str) + df['Trade_cat'].astype(str)


def backtest_trades(
    df: pd.DataFrame,
    signals: List[int],
    pattern_filters: Dict[str, PatternFilter],
    profit_target: float = 0.03,
    max_hold: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Back‑test the strategy for each pattern and return statistics.

    For each signal index, check if it matches one of the defined patterns
    under the specified filters.  If so, simulate a trade: enter at
    next day's open, exit at 3% profit or after max_hold days.  Record
    metrics such as total trades, wins, success ratio, and average return.

    Returns a dictionary keyed by pattern, each containing stats.
    """
    # Prepare a structure for per‑pattern trade logs
    results = {p: {'count': 0, 'wins': 0, 'returns': []} for p in pattern_filters}
    for idx in signals:
        entry_idx = idx + 1
        # Skip if there's no next day to enter
        if entry_idx >= len(df):
            continue
        # Determine the pattern on the signal day
        pattern_code = df.loc[idx, 'pattern']
        if pattern_code not in pattern_filters:
            continue
        flt = pattern_filters[pattern_code]
        # Filter by FuturePremium_IMP
        if df.loc[idx, 'FuturePremium_IMP'] == flt.exclude_premium:
            continue
        # Filter by allowed ZscoreCategory
        if flt.allowed_categories is not None and df.loc[idx, 'ZscoreCategory'] not in flt.allowed_categories:
            continue
        # All filters passed: simulate a trade
        entry_price = df.loc[entry_idx, 'Open']
        # Track drawdown within the holding window for informational purposes
        win = False
        exit_price = None
        for j in range(entry_idx, min(entry_idx + max_hold, len(df))):
            high_price = df.loc[j, 'High']
            if high_price >= entry_price * (1 + profit_target):
                exit_price = entry_price * (1 + profit_target)
                win = True
                break
        if not win:
            j = min(entry_idx + max_hold - 1, len(df) - 1)
            exit_price = df.loc[j, 'Close']
        trade_return = (exit_price - entry_price) / entry_price
        res = results[pattern_code]
        res['count'] += 1
        res['wins'] += int(win)
        res['returns'].append(trade_return)
    # Convert logs to summary statistics
    summary = {}
    for p, res in results.items():
        if res['count'] == 0:
            summary[p] = {'count': 0, 'success_rate': None, 'average_return': None}
        else:
            success_rate = res['wins'] / res['count']
            avg_return = float(np.mean(res['returns']))
            summary[p] = {
                'count': res['count'],
                'success_rate': success_rate,
                'average_return': avg_return,
            }
    return summary


def determine_high_success_categories(df: pd.DataFrame, signals: List[int], exclude_premium: str, pattern_code: str) -> List[str]:
    """Automatically determine ZscoreCategory values with ≥70% success.

    For the given pattern_code, exclude rows where FuturePremium_IMP
    equals exclude_premium, then compute the success rate of each
    ZscoreCategory based on the profit target.  Return those
    categories with success rate ≥ 0.7.
    """
    # Collect trades for this pattern without ZscoreCategory filtering
    records = []
    profit_target = 0.03
    max_hold = 10
    for idx in signals:
        if df.loc[idx, 'pattern'] != pattern_code:
            continue
        # Filter premium
        if df.loc[idx, 'FuturePremium_IMP'] == exclude_premium:
            continue
        entry_idx = idx + 1
        if entry_idx >= len(df):
            continue
        entry_price = df.loc[entry_idx, 'Open']
        win = False
        for j in range(entry_idx, min(entry_idx + max_hold, len(df))):
            if df.loc[j, 'High'] >= entry_price * (1 + profit_target):
                win = True
                break
        records.append((df.loc[idx, 'ZscoreCategory'], win))
    if not records:
        return []
    summary = pd.DataFrame(records, columns=['ZscoreCategory', 'win'])
    rates = summary.groupby('ZscoreCategory')['win'].mean()
    return rates[rates >= 0.7].index.tolist()


def update_dataset_with_signals(
    df: pd.DataFrame,
    signals: List[int],
    pattern_filters: Dict[str, PatternFilter],
    live_flag: bool = True,
) -> pd.DataFrame:
    """Add a column `PatternSignal` marking rows that generate a pattern signal.

    On the signal day, if all filters pass, the pattern code is written to
    `PatternSignal`.  If there is no next day (i.e. live data), the
    last row can still be marked if `live_flag` is True.  Otherwise,
    future signals at the end of the dataset are ignored.
    """
    df = df.copy()
    df['PatternSignal'] = ''
    n = len(df)
    for idx in signals:
        # Determine if a trade could be taken (next day must exist) or if live_flag allows marking
        entry_idx = idx + 1
        if entry_idx >= n and not live_flag:
            continue
        pattern_code = df.loc[idx, 'pattern']
        if pattern_code not in pattern_filters:
            continue
        flt = pattern_filters[pattern_code]
        # Apply premium filter
        if df.loc[idx, 'FuturePremium_IMP'] == flt.exclude_premium:
            continue
        # Apply ZscoreCategory filter
        if flt.allowed_categories is not None and df.loc[idx, 'ZscoreCategory'] not in flt.allowed_categories:
            continue
        df.at[idx, 'PatternSignal'] = pattern_code
    return df


def main(file_path: str) -> None:
    """Load data, compute signals, back‑test patterns, update dataset."""
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    # Ensure required columns are present
    required_cols = [
        'BuyingPressure', 'Open', 'Close', 'High',
        'VolumeZscore', 'TradesZscore', 'FuturePremium_IMP', 'ZscoreCategory'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {', '.join(missing)}")
    # Detect EMA crossover signals
    fast_period = 5
    slow_period = 21
    signals = detect_signals(df, fast=fast_period, slow=slow_period)
    # Compute pattern categories
    compute_pattern_categories(df)
    # Define pattern filters with placeholders; pattern 33's allowed categories
    # will be determined dynamically.
    pattern_filters: Dict[str, PatternFilter] = {
        '02': PatternFilter(exclude_premium='High', allowed_categories=['Low / Low / Low']),
        '21': PatternFilter(exclude_premium='Low', allowed_categories=['Low / Low / Low', 'Low / Medium / Low', 'Medium / Low / Low']),
        '11': PatternFilter(exclude_premium='Low', allowed_categories=['Low / Medium / Low']),
        '33': PatternFilter(exclude_premium='Very High', allowed_categories=None),
    }
    # Determine high‑success categories for pattern 33
    if '33' in pattern_filters:
        good_cats = determine_high_success_categories(
            df, signals, exclude_premium=pattern_filters['33'].exclude_premium, pattern_code='33'
        )
        pattern_filters['33'].allowed_categories = good_cats
    # Back‑test the strategy per pattern
    summary = backtest_trades(df, signals, pattern_filters, profit_target=0.03, max_hold=10)
    # Update dataset with live signal markers
    updated_df = update_dataset_with_signals(df, signals, pattern_filters, live_flag=True)
    # Save updated file with `_updated` suffix
    base, ext = os.path.splitext(file_path)
    out_path = f"{base}_updated{ext}"
    updated_df.to_csv(out_path, index=False)
    # Print summary results
    print("Back‑test summary by pattern:")
    for pat, stats in summary.items():
        count = stats['count']
        success = stats['success_rate']
        avg_ret = stats['average_return']
        print(f"Pattern {pat}: trades={count}, success_rate={success:.2%} if count else 'N/A', avg_return={avg_ret:.4%} if count else 'N/A'")
    print(f"Updated file written to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Buying Pressure EMA pattern strategy with live signal flags.")
    parser.add_argument('--file', required=True, help="Path to the *_all.csv file for a stock")
    args = parser.parse_args()
    main(args.file)