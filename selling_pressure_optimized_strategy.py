"""
selling_pressure_optimized_strategy
===================================

This module implements an enhanced short‑selling strategy based on
exponential moving average (EMA) crossovers of the ``SellingPressure``
series and a set of pattern and premium/category filters.  The goal
is to improve the success rate of short trades by selecting only those
signals that historically performed well and combining them with
additional context from the dataset.

Key features
------------

* Signals are generated whenever the 5‑period EMA of ``SellingPressure``
  crosses **below** the 21‑period EMA.  The next trading day’s open is
  used as the entry price for a short position.  The trade is exited
  when the price falls 3 % from the entry price or after holding for
  10 trading days.
* For each signal day, the script classifies the ``VolumeZscore`` and
  ``TradesZscore`` into four categories (0–3) based on the positive
  z‑score tertiles.  Combining these categories yields a two‑digit
  pattern string (e.g. ``'10'`` for low‑positive volume and
  non‑positive trades).
* The strategy focuses on a small set of patterns (default: ``10``,
  ``22`` and ``02``) that historically exhibited higher success
  ratios.  Additional filters can be applied per pattern to restrict
  the ``FuturePremium_IMP`` values and ``ZscoreCategory`` strings.
* For each qualifying signal, the script records a detailed trade log
  with the signal date, entry date and price, exit date and price,
  holding period, return percentage and whether the trade hit the
  profit target.
* When run in multi‑stock mode, the script loops over all symbols in
  a user‑supplied list and writes a separate trade log CSV for each
  stock.  It also prints a summary of the number of trades and
  success ratio for every symbol.

Usage example
-------------

Assuming you have a directory structure like::

    /data/stocks/
    ├── stock_list.csv         # contains a column 'stock' or 'symbol'
    ├── AUBANK/
    │   └── stockdata_AUBANK_Analyzed_All.csv
    └── AUROPHARMA/
        └── stockdata_AUROPHARMA_Analyzed_All.csv

You can run the strategy over multiple stocks and write the trade
logs to each stock’s folder as follows::

    python selling_pressure_optimized_strategy.py \
        --list /data/stocks/stock_list.csv \
        --base-dir /data/stocks \
        --data-pattern stockdata_{symbol}_Analyzed_All.csv \
        --patterns 10 22 02 \
        --premium-filter 10:Medium,High 22:Low,Medium 02:High \
        --category-filter 10:"Low / Low / Low" \
        --category-filter 22:"Low / Low / Medium","Low / Medium / Medium" \
        --category-filter 02:"Low / Medium / Low","Low / Medium / Medium","Low / Low / Medium"

The ``--premium-filter`` and ``--category-filter`` options allow you
to specify which ``FuturePremium_IMP`` and ``ZscoreCategory`` values
are acceptable for each pattern.  Multiple categories can be listed
for a single pattern by repeating the ``--category-filter`` option.

This script is intended to be flexible: you can adjust the patterns
and filters based on your own research.  See the ``main`` function at
the bottom for details on the command‑line interface.
"""

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def categorize(series: pd.Series) -> pd.Series:
    """Categorize z‑scores into 0–3 based on positive tertiles.

    ``0`` means non‑positive, ``1`` means lower third of positive
    values, ``2`` means middle third and ``3`` means the highest
    third.  Returns a new ``Series`` of integers of the same length
    as ``series``.
    """
    positive = series[series > 0]
    q1 = positive.quantile(0.33) if not positive.empty else 0
    q2 = positive.quantile(0.66) if not positive.empty else 0
    def cat(z: float) -> int:
        if z <= 0:
            return 0
        elif z <= q1:
            return 1
        elif z <= q2:
            return 2
        else:
            return 3
    return series.apply(cat)


@dataclass
class Trade:
    pattern: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    holding_days: int
    return_pct: float
    win: bool


def generate_trades(
    df: pd.DataFrame,
    patterns: Iterable[str],
    premium_filters: Dict[str, Optional[Sequence[str]]],
    category_filters: Dict[str, Optional[Sequence[str]]],
    fast_period: int = 5,
    slow_period: int = 21,
    profit_target: float = 0.03,
    max_hold: int = 10,
) -> List[Trade]:
    """Generate filtered short trades for a single stock.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing columns ``Date``, ``SellingPressure``,
        ``VolumeZscore``, ``TradesZscore``, ``FuturePremium_IMP`` and
        ``ZscoreCategory``.
    patterns : iterable of str
        Two‑digit pattern codes (e.g. '10', '22', '02') to allow.
    premium_filters : dict
        Mapping from pattern code to a sequence of acceptable
        ``FuturePremium_IMP`` values.  ``None`` means no filter for
        that pattern.
    category_filters : dict
        Mapping from pattern code to a sequence of acceptable
        ``ZscoreCategory`` strings.  ``None`` means no filter for
        that pattern.
    fast_period, slow_period : int
        Periods for the EMAs.  The fast period should be shorter than
        the slow period.
    profit_target : float
        Profit target for the short trade, expressed as a decimal (3 %
        = 0.03).  Exits occur when the price drops by this fraction.
    max_hold : int
        Maximum holding period (in trading days) for each trade.

    Returns
    -------
    list of Trade
        All qualifying trades with details.
    """
    trades: List[Trade] = []
    # Ensure required columns exist
    for col in [
        'SellingPressure', 'VolumeZscore', 'TradesZscore',
        'FuturePremium_IMP', 'ZscoreCategory', 'Open', 'High', 'Low', 'Close'
    ]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in data frame")

    # Compute EMAs for SellingPressure
    ema_fast = df['SellingPressure'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['SellingPressure'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_prev = macd.shift(1)
    # Compute pattern categories for every row
    vol_cat = categorize(df['VolumeZscore'])
    trades_cat = categorize(df['TradesZscore'])
    pattern_series = vol_cat.astype(str) + trades_cat.astype(str)
    # Identify bearish crossover indices
    signal_indices = df.index[(macd_prev >= 0) & (macd < 0)]
    for idx in signal_indices:
        pattern = pattern_series.iloc[idx]
        # Only consider patterns of interest
        if pattern not in patterns:
            continue
        premium = df.loc[idx, 'FuturePremium_IMP']
        category = df.loc[idx, 'ZscoreCategory']
        # Apply premium filter
        allowed_prem = premium_filters.get(pattern)
        if allowed_prem is not None and premium not in allowed_prem:
            continue
        # Apply category filter
        allowed_cat = category_filters.get(pattern)
        if allowed_cat is not None and category not in allowed_cat:
            continue
        # Determine entry index (next day)
        entry_idx = idx + 1
        if entry_idx >= len(df):
            continue  # no next day to enter
        entry_price = df.loc[entry_idx, 'Open']
        # Simulate trade
        hit_target = False
        exit_idx: Optional[int] = None
        for j in range(entry_idx, min(entry_idx + max_hold, len(df))):
            if df.loc[j, 'Low'] <= entry_price * (1 - profit_target):
                hit_target = True
                exit_idx = j
                break
        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold - 1, len(df) - 1)
        exit_price = entry_price * (1 - profit_target) if hit_target else df.loc[exit_idx, 'Close']
        return_pct = (entry_price - exit_price) / entry_price
        holding_days = exit_idx - entry_idx + 1
        trades.append(
            Trade(
                pattern=pattern,
                signal_date=df.loc[idx, 'Date'],
                entry_date=df.loc[entry_idx, 'Date'],
                entry_price=float(entry_price),
                exit_date=df.loc[exit_idx, 'Date'],
                exit_price=float(exit_price),
                holding_days=int(holding_days),
                return_pct=float(return_pct),
                win=hit_target,
            )
        )
    return trades


def find_data_file(stock_dir: str, data_pattern: str, symbol: str) -> Optional[str]:
    """Locate a CSV file for a stock given a directory and a filename pattern.

    If ``data_pattern`` contains ``{symbol}``, it is formatted with
    the symbol and checked.  If that file does not exist, the function
    falls back to any file ending with ``All.csv`` in the directory.
    """
    candidate = data_pattern.format(symbol=symbol)
    path = os.path.join(stock_dir, candidate)
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(stock_dir, '*All.csv'))
    return matches[0] if matches else None


def parse_filter_argument(arg: str) -> Tuple[str, List[str]]:
    """Parse a filter argument of the form ``pattern:value1,value2``.

    Returns a pair (pattern, list of values).  Patterns should be
    two‑digit strings.  Values are comma‑separated.
    """
    if ':' not in arg:
        raise argparse.ArgumentTypeError(
            f"Invalid filter format '{arg}'. Expected PATTERN:VAL1,VAL2,..."
        )
    pat, vals = arg.split(':', 1)
    if not pat.isdigit() or len(pat) != 2:
        raise argparse.ArgumentTypeError(
            f"Pattern '{pat}' must be a two‑digit code (e.g. 10, 22, 02)."
        )
    values = [v.strip() for v in vals.split(',') if v.strip()]
    return pat, values


def process_stock(
    symbol: str,
    stock_dir: str,
    data_pattern: str,
    patterns: List[str],
    premium_filters: Dict[str, Optional[List[str]]],
    category_filters: Dict[str, Optional[List[str]]],
    output_dir: str,
) -> Tuple[int, float]:
    """Process a single stock: generate trades and write trade log.

    Returns a tuple (number_of_trades, success_rate).
    """
    data_file = find_data_file(stock_dir, data_pattern, symbol)
    if not data_file:
        print(f"[WARNING] No data file found for {symbol} in {stock_dir}. Skipping.")
        return 0, 0.0
    df = pd.read_csv(data_file)
    # convert date column if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    trades = generate_trades(df, patterns, premium_filters, category_filters)
    # Write trade log
    out_file = os.path.join(output_dir, f'trade_log_{symbol}.csv')
    os.makedirs(output_dir, exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'pattern', 'signal_date', 'entry_date', 'entry_price',
            'exit_date', 'exit_price', 'holding_days', 'return_pct', 'win'
        ])
        for tr in trades:
            writer.writerow([
                tr.pattern,
                tr.signal_date,
                tr.entry_date,
                f"{tr.entry_price:.2f}",
                tr.exit_date,
                f"{tr.exit_price:.2f}",
                tr.holding_days,
                f"{tr.return_pct:.4f}",
                tr.win,
            ])
    # Compute summary stats
    if trades:
        wins = sum(t.win for t in trades)
        success_rate = wins / len(trades)
    else:
        success_rate = 0.0
    print(
        f"[INFO] {symbol}: trades={len(trades):>3}, wins={sum(t.win for t in trades):>3}, "
        f"success_rate={success_rate:.1%}"
    )
    return len(trades), success_rate


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized short trades from SellingPressure EMA crossovers."
    )
    parser.add_argument(
        '--list',
        type=str,
        required=True,
        help="CSV file listing stock symbols in a column named 'stock' or 'symbol'.",
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='.',
        help="Base directory containing subdirectories for each stock.",
    )
    parser.add_argument(
        '--data-pattern',
        type=str,
        default='stockdata_{symbol}_Analyzed_All.csv',
        help="Filename pattern for each stock's data file (default: stockdata_{symbol}_Analyzed_All.csv).",
    )
    parser.add_argument(
        '--patterns',
        nargs='+',
        required=True,
        help="Two‑digit pattern codes to consider (e.g. 10 22 02).",
    )
    parser.add_argument(
        '--premium-filter',
        action='append',
        default=[],
        metavar='PATTERN:VAL1,VAL2,...',
        help="Specify acceptable FuturePremium_IMP values for a pattern. Repeat for multiple patterns.",
    )
    parser.add_argument(
        '--category-filter',
        action='append',
        default=[],
        metavar='PATTERN:VAL1,VAL2,...',
        help="Specify acceptable ZscoreCategory values for a pattern. Repeat to add more categories.",
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help="Directory in which to write trade log CSV files.",
    )
    args = parser.parse_args()
    # Read stock list
    stocks_df = pd.read_csv(args.list)
    if 'stock' in stocks_df.columns:
        symbols = stocks_df['stock'].astype(str).tolist()
    elif 'symbol' in stocks_df.columns:
        symbols = stocks_df['symbol'].astype(str).tolist()
    else:
        raise ValueError("Stock list must contain a 'stock' or 'symbol' column")
    # Build premium and category filters
    premium_filters: Dict[str, Optional[List[str]]] = {pat: None for pat in args.patterns}
    category_filters: Dict[str, Optional[List[str]]] = {pat: None for pat in args.patterns}
    for pf_arg in args.premium_filter:
        pat, vals = parse_filter_argument(pf_arg)
        premium_filters[pat] = vals
    # Category filter arguments may specify multiple categories per pattern; aggregate
    for cf_arg in args.category_filter:
        pat, vals = parse_filter_argument(cf_arg)
        existing = category_filters.get(pat)
        if existing is None:
            category_filters[pat] = vals
        else:
            # append unique values
            for v in vals:
                if v not in existing:
                    existing.append(v)
    # Process each stock
    overall_trades = 0
    overall_wins = 0
    for symbol in symbols:
        stock_dir = os.path.join(args.base_dir, symbol)
        if not os.path.isdir(stock_dir):
            print(f"[WARNING] Directory {stock_dir} does not exist. Skipping {symbol}.")
            continue
        trades_count, success_rate = process_stock(
            symbol,
            stock_dir,
            args.data_pattern,
            args.patterns,
            premium_filters,
            category_filters,
            args.output_dir,
        )
        # update overall stats
        df_trades = pd.read_csv(os.path.join(args.output_dir, f'trade_log_{symbol}.csv')) if trades_count > 0 else pd.DataFrame()
        overall_trades += trades_count
        overall_wins += df_trades['win'].sum() if 'win' in df_trades.columns else 0
    if overall_trades:
        print(
            f"\n[SUMMARY] total trades={overall_trades}, total wins={overall_wins}, "
            f"overall success rate={overall_wins / overall_trades:.1%}"
        )
    else:
        print("\n[SUMMARY] no trades generated across all stocks.")


if __name__ == '__main__':
    main()