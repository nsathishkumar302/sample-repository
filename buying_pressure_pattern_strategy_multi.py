"""
Multi‑Stock Buying Pressure EMA Crossover Strategy
--------------------------------------------------

This script extends the single‑stock strategy implemented in
``buying_pressure_pattern_strategy.py`` to multiple stocks.  It
iterates over a list of stock symbols provided in a CSV file, applies
the 5/21‑period EMA crossover on the ``BuyingPressure`` series, filters
signals by volume/trades z‑score patterns and premium/z‑score
categories, back‑tests each qualifying trade, and records detailed
trade logs for every stock.

Key features:

* **Automatic pattern filtering** – For patterns 02, 21 and 11,
  pre-defined filters exclude specific ``FuturePremium_IMP`` values and
  restrict ``ZscoreCategory`` to high‑performing groups.  For pattern
  33, the script automatically determines which ``ZscoreCategory``
  values achieve at least a 70 % success rate after excluding
  ``FuturePremium_IMP = 'Very High'``.
* **Back‑test results** – For each stock, the script captures a
  detailed trade log containing the signal date, entry date, entry
  price, exit date, exit price, holding period, return percentage,
  maximum drawdown percentage during the trade, pattern code and
  outcome (win/lose).  These logs are saved to a CSV file named
  ``trade_log_{symbol}.csv`` in the stock’s directory.
* **Live signal marking** – The original ``*_All.csv`` file is
  updated in place by adding a ``PatternSignal`` column indicating
  which pattern triggered on each signal day.  If a signal occurs on
  the last row (live data), the pattern code is still written so you
  know whether to trade tomorrow.  No ``_updated`` file is created.

Usage example:

```
python buying_pressure_pattern_strategy_multi.py \
    --list stock_list.csv \
    --base-dir /data/stocks \
    --data-pattern stockdata_{symbol}_Analyzed_All.csv
```

Where ``stock_list.csv`` contains a column ``stock`` or ``symbol`` with
stock names.  Each stock folder under ``base-dir`` should contain the
corresponding ``*_Analyzed_All.csv`` file matching ``data-pattern``.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def categorize_zscore(z: float, q1: float, q2: float) -> int:
    if z <= 0:
        return 0
    if z <= q1:
        return 1
    if z <= q2:
        return 2
    return 3


def detect_signals(df: pd.DataFrame, fast: int, slow: int) -> List[int]:
    ema_fast = compute_ema(df['BuyingPressure'], span=fast)
    ema_slow = compute_ema(df['BuyingPressure'], span=slow)
    diff = ema_fast - ema_slow
    # bullish crossover
    return df.index[(diff.shift(1) <= 0) & (diff > 0)].tolist()


def assign_pattern_categories(df: pd.DataFrame) -> None:
    # determine quantiles for positive z‑scores
    vol_pos = df.loc[df['VolumeZscore'] > 0, 'VolumeZscore']
    trade_pos = df.loc[df['TradesZscore'] > 0, 'TradesZscore']
    vol_q1, vol_q2 = (vol_pos.quantile(0.33), vol_pos.quantile(0.66)) if not vol_pos.empty else (0.0, 0.0)
    trade_q1, trade_q2 = (trade_pos.quantile(0.33), trade_pos.quantile(0.66)) if not trade_pos.empty else (0.0, 0.0)
    df['Vol_cat'] = df['VolumeZscore'].apply(lambda z: categorize_zscore(z, vol_q1, vol_q2))
    df['Trade_cat'] = df['TradesZscore'].apply(lambda z: categorize_zscore(z, trade_q1, trade_q2))
    df['pattern'] = df['Vol_cat'].astype(str) + df['Trade_cat'].astype(str)


def determine_high_success_categories(
    df: pd.DataFrame,
    signals: List[int],
    pattern_code: str,
    exclude_premium: str,
    profit_target: float = 0.03,
    max_hold: int = 10,
) -> List[str]:
    """Identify ZscoreCategory values with >=70% success for pattern 33."""
    records: List[Tuple[str, bool]] = []
    for idx in signals:
        if df.loc[idx, 'pattern'] != pattern_code:
            continue
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
    rates = pd.DataFrame(records, columns=['ZscoreCategory', 'win']).groupby('ZscoreCategory')['win'].mean()
    return rates[rates >= 0.7].index.tolist()


def simulate_trades(
    df: pd.DataFrame,
    signals: List[int],
    pattern_filters: Dict[str, Dict[str, Optional[List[str]]]],
    profit_target: float = 0.03,
    max_hold: int = 10,
) -> pd.DataFrame:
    """Simulate trades and return a DataFrame with detailed trade records."""
    records = []
    for idx in signals:
        entry_idx = idx + 1
        if entry_idx >= len(df):
            continue
        pattern_code = df.loc[idx, 'pattern']
        if pattern_code not in pattern_filters:
            continue
        flt = pattern_filters[pattern_code]
        # premium filter
        if df.loc[idx, 'FuturePremium_IMP'] == flt['exclude']:
            continue
        # zscorecategory filter
        if flt['allowed'] is not None and df.loc[idx, 'ZscoreCategory'] not in flt['allowed']:
            continue
        # simulate trade
        entry_date = df.loc[entry_idx, 'Date']
        entry_price = df.loc[entry_idx, 'Open']
        min_price = entry_price
        win = False
        exit_date = None
        exit_price = None
        hold_days = 0
        for j in range(entry_idx, min(entry_idx + max_hold, len(df))):
            low_price = df.loc[j, 'Low'] if 'Low' in df.columns else df.loc[j, 'Close']
            if low_price < min_price:
                min_price = low_price
            high_price = df.loc[j, 'High']
            if high_price >= entry_price * (1 + profit_target):
                exit_price = entry_price * (1 + profit_target)
                exit_date = df.loc[j, 'Date']
                hold_days = j - entry_idx + 1
                win = True
                break
        if not win:
            j = min(entry_idx + max_hold - 1, len(df) - 1)
            exit_date = df.loc[j, 'Date']
            exit_price = df.loc[j, 'Close']
            hold_days = j - entry_idx + 1
        ret = (exit_price - entry_price) / entry_price * 100
        dd = (min_price - entry_price) / entry_price * 100
        records.append({
            'pattern': pattern_code,
            'signal_date': df.loc[idx, 'Date'],
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'holding_days': hold_days,
            'return_pct': ret,
            'max_drawdown_pct': dd,
            'trade_result': 'Win' if win else 'Lose',
        })
    return pd.DataFrame(records)


def update_pattern_signal_column(
    df: pd.DataFrame,
    signals: List[int],
    pattern_filters: Dict[str, Dict[str, Optional[List[str]]]],
    live_flag: bool = True,
) -> None:
    """Update df in place by marking rows that generate a pattern signal."""
    df['PatternSignal'] = ''
    n = len(df)
    for idx in signals:
        entry_idx = idx + 1
        if entry_idx >= n and not live_flag:
            continue
        pattern_code = df.loc[idx, 'pattern']
        if pattern_code not in pattern_filters:
            continue
        flt = pattern_filters[pattern_code]
        if df.loc[idx, 'FuturePremium_IMP'] == flt['exclude']:
            continue
        if flt['allowed'] is not None and df.loc[idx, 'ZscoreCategory'] not in flt['allowed']:
            continue
        df.at[idx, 'PatternSignal'] = pattern_code


def process_stock(
    file_path: str,
    fast_period: int = 5,
    slow_period: int = 21,
    profit_target: float = 0.03,
    max_hold: int = 10,
) -> None:
    """Process a single stock file: compute signals, back‑test and log trades."""
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    # verify required columns
    required = ['BuyingPressure', 'Open', 'Close', 'High', 'Low', 'VolumeZscore', 'TradesZscore', 'FuturePremium_IMP', 'ZscoreCategory']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")
    # compute patterns
    assign_pattern_categories(df)
    # detect signals
    signals = detect_signals(df, fast_period, slow_period)
    # define filters for patterns (allowed will be filled for pattern 33)
    pattern_filters: Dict[str, Dict[str, Optional[List[str]]]] = {
        '02': {'exclude': 'High', 'allowed': ['Low / Low / Low']},
        '21': {'exclude': 'Low', 'allowed': ['Low / Low / Low', 'Low / Medium / Low', 'Medium / Low / Low']},
        '11': {'exclude': 'Low', 'allowed': ['Low / Medium / Low']},
        '33': {'exclude': 'Very High', 'allowed': None},
    }
    # automatically determine high‑success ZscoreCategory for pattern 33
    if '33' in pattern_filters:
        good_cats = determine_high_success_categories(df, signals, '33', pattern_filters['33']['exclude'], profit_target, max_hold)
        pattern_filters['33']['allowed'] = good_cats
    # simulate trades
    trades_df = simulate_trades(df, signals, pattern_filters, profit_target, max_hold)
    # update pattern signal in df for live data
    update_pattern_signal_column(df, signals, pattern_filters, live_flag=True)
    # save trade log
    # trade file name is based on original file
    base_dir, filename = os.path.split(file_path)
    symbol = os.path.splitext(filename)[0]
    trade_log_path = os.path.join(base_dir, f"trade_log_{symbol}.csv")
    trades_df.to_csv(trade_log_path, index=False)
    # overwrite original file with updated PatternSignal column
    df.to_csv(file_path, index=False)
    # print summary for the stock
    print(f"Processed {symbol}: {len(trades_df)} trades logged. Pattern signals updated.")


def find_data_file(stock_dir: str, data_pattern: str, symbol: str) -> Optional[str]:
    """Find the data file for a given stock using the pattern.

    If the formatted pattern isn't found, fall back to any file containing
    'all.csv'.
    """
    candidate = data_pattern.format(symbol=symbol)
    path = os.path.join(stock_dir, candidate)
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(stock_dir, '*all.csv'))
    return matches[0] if matches else None


def main_multi(list_path: str, base_dir: str, data_pattern: str) -> None:
    stocks = pd.read_csv(list_path)
    if 'stock' in stocks.columns:
        symbols = stocks['stock'].astype(str).tolist()
    elif 'symbol' in stocks.columns:
        symbols = stocks['symbol'].astype(str).tolist()
    else:
        raise ValueError("Stock list must contain a 'stock' or 'symbol' column")
    for symbol in symbols:
        stock_dir = os.path.join(base_dir, symbol)
        if not os.path.isdir(stock_dir):
            print(f"[WARNING] directory for {symbol} not found at {stock_dir}")
            continue
        data_file = find_data_file(stock_dir, data_pattern, symbol)
        if not data_file:
            print(f"[WARNING] data file for {symbol} not found in {stock_dir}")
            continue
        try:
            process_stock(data_file)
        except Exception as e:
            print(f"[ERROR] processing {symbol}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Buying Pressure EMA pattern strategy on multiple stocks.")
    parser.add_argument('--list', required=True, help="CSV file listing stock names")
    parser.add_argument('--base-dir', required=True, help="Base directory containing stock folders")
    parser.add_argument('--data-pattern', default='stockdata_{symbol}_Analyzed_All.csv', help="Pattern for each stock's data file")
    args = parser.parse_args()
    main_multi(args.list, args.base_dir, args.data_pattern)