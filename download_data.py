import yfinance as yf
import pandas as pd
import os
import sys
import requests
from io import StringIO
from pathlib import Path
import qlib

# from qlib.data import from_csv


def get_sp500_tickers():
    # Helper to get S&P 500 tickers.
    # Use requests with headers to avoid 403
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"Failed to fetch S&P 500 list: {r.status_code}")
            return [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "NVDA",
                "TSLA",
                "META",
                "BRK-B",
                "UNH",
                "JNJ",
            ]

        df = pd.read_html(StringIO(r.text))[0]
        tickers = df["Symbol"].tolist()
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        # Fallback to top 10 for testing
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "TSLA",
            "META",
            "BRK-B",
            "UNH",
            "JNJ",
        ]


def download_data(
    tickers, start_date="2020-01-01", end_date=None, output_dir="raw_data"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Checking existing data for {len(tickers)} tickers...")

    missing_tickers = []
    update_tickers = []

    # Check status of each ticker
    import datetime

    # We'll use a conservative start date for updates (min of last dates)
    # But doing batch download for all updates might be tricky if dates vary wildly.
    # Assumption: User runs this regularly, so dates are likely clustered.
    # To be safe and efficient:
    # 1. Identify missing files -> Full Download.
    # 2. Identify existing files -> Find the oldest 'last_date'. Download from there for ALL existing.

    existing_last_dates = []

    # Optimize partial reading: only need the last line to get the date.
    def get_last_date(file_path):
        try:
            with open(file_path, "rb") as f:
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b"\n":
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
                # Assuming date is first column or we parse csv line
                # Simple split by comma
                parts = last_line.split(",")
                if parts:
                    return parts[0]  # First column is usually date in our saved format
        except:
            return None
        return None

    # Determine status
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Checking local data...")
    for t in tickers:
        csv_path = os.path.join(output_dir, f"{t}.csv")
        if os.path.exists(csv_path):
            l_date = get_last_date(csv_path)
            if l_date:
                try:
                    # Valid attempt to parse date
                    last_date = pd.to_datetime(l_date)
                    existing_last_dates.append(last_date)
                    update_tickers.append(t)
                except:
                    missing_tickers.append(t)
            else:
                missing_tickers.append(t)
        else:
            missing_tickers.append(t)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Check complete.")

    # --- Batch 1: Missing Tickers (Full Download) ---
    if missing_tickers:
        print(f"Full download for {len(missing_tickers)} missing tickers...")
        data = yf.download(
            missing_tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            threads=True,
        )
        # Process and save (Reuse logic, encapsulated in helper ideally, but inline here for simplicity)
        _save_downloaded_data(data, missing_tickers, output_dir, mode="w")

    # --- Batch 2: Update Tickers (Incremental) ---
    if update_tickers:
        if existing_last_dates:
            min_last_date = min(existing_last_dates)
            # Start from next day
            update_start = (min_last_date + datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

            # If update_start is in the future (compared to today), skip
            if update_start > datetime.datetime.now().strftime("%Y-%m-%d"):
                print("Data is up to date.")
            else:
                print(
                    f"Incremental update for {len(update_tickers)} tickers from {update_start}..."
                )
                data = yf.download(
                    update_tickers,
                    start=update_start,
                    end=end_date,
                    group_by="ticker",
                    threads=True,
                )
                _save_downloaded_data(data, update_tickers, output_dir, mode="a")
        else:
            # Should not happen if update_tickers is not empty
            pass

    return output_dir


def _save_downloaded_data(data, tickers, output_dir, mode="w"):
    valid_count = 0
    for ticker in tickers:
        try:
            # Handle single ticker case where yf doesn't return MultiIndex if len(tickers)==1?
            # yf.download(..., group_by='ticker') usually returns MultiIndex (Ticker, Field)
            # OR (Field, Ticker) ?
            # It depends on version. With group_by='ticker', it is usually top level Ticker.
            # If single ticker, it might be just (Field).

            if len(tickers) == 1:
                df = data.copy()
            else:
                if ticker not in data.columns:  # Ticker level check
                    print(f"No data for {ticker}")
                    continue
                df = data[ticker].copy()

            if df.empty:
                continue

            # Check if index is DatetimeIndex
            # df.index should be Datetime

            # Reset index to get Date as column
            df = df.reset_index()
            # If 'Date' is not in columns after reset, something is wrong
            # Column name might be 'Date' or 'Datetime'

            # Standardize columns
            df.columns = [str(c).lower() for c in df.columns]

            # Identify date column
            date_col = None
            if "date" in df.columns:
                date_col = "date"
            elif "datetime" in df.columns:
                date_col = "datetime"

            if not date_col:
                print(f"Missing date column for {ticker}")
                continue

            # Rename to 'date'
            if date_col != "date":
                df = df.rename(columns={date_col: "date"})

            # Ensure required columns
            required_cols = ["date", "open", "high", "low", "close", "volume"]
            # intersection
            if not all(col in df.columns for col in required_cols):
                # print(f"Missing columns for {ticker}: {df.columns}")
                # yf Sometimes returns Capitalized
                continue

            df["symbol"] = ticker

            # Adj Close
            if "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adjclose"})
            elif "adjclose" not in df.columns:
                df["adjclose"] = df["close"]

            # VWAP
            df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

            # Save
            csv_path = os.path.join(output_dir, f"{ticker}.csv")

            if mode == "a" and os.path.exists(csv_path):
                # Append mode: Read existing, append, drop duplicates
                existing_df = pd.read_csv(csv_path)
                combined_df = pd.concat([existing_df, df])
                # Ensure date is proper datetime for sorting
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                # Drop duplicates on date
                combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
                combined_df = combined_df.sort_values("date")
                combined_df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, index=False)

            valid_count += 1

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print(f"Processed {valid_count} tickers (Mode: {mode}).")


def convert_to_qlib_bin(source_dir, output_dir="qlib_data"):
    # Initialize Qlib (not strictly necessary for dump_bin but good practice)
    # Use qlib's dump_bin logic
    # We can run the command line tool or call internal function.
    # Calling command line is often easier to avoid internal API changes.

    python_exe = sys.executable
    cmd = f"{python_exe} scripts/dump_bin.py dump_all --data_path {source_dir} --qlib_dir {output_dir} --include_fields open,high,low,close,volume,adjclose,vwap --date_field_name date"
    print(f"Running conversion command: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    current_dir = Path(__file__).parent.absolute()
    raw_data_dir = current_dir / "raw_data"
    qlib_data_dir = current_dir / "qlib_data"

    tickers = get_sp500_tickers()
    tickers.append("^GSPC")  # Add benchmark
    # For full run, comment out the slicing
    # tickers = tickers[:50] # Limit for initial testing

    download_data(tickers, output_dir=raw_data_dir)
    convert_to_qlib_bin(raw_data_dir, output_dir=qlib_data_dir)
