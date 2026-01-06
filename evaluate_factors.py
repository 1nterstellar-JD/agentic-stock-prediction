import qlib
from qlib.data import D
import pandas as pd
import yaml
import os
import datetime
from factor_loader import get_generated_factor_expressions


def evaluate_factors():
    # Load config for Qlib init and periods
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        qlib_cfg = config.get("qlib_init")
        if "provider_uri" in qlib_cfg and not os.path.isabs(qlib_cfg["provider_uri"]):
            qlib_cfg["provider_uri"] = os.path.join(
                os.getcwd(), qlib_cfg["provider_uri"]
            )
        qlib.init(**qlib_cfg)
    except Exception as e:
        print(f"Error init qlib: {e}")
        qlib.init(provider_uri="qlib_data")

    # Define Validation Period
    # We want to evaluate on recent data to ensure robustness
    today = datetime.datetime.now()
    valid_end = today.strftime("%Y-%m-%d")
    valid_start = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"Evaluating factors on period: {valid_start} to {valid_end}")

    # Load Factors
    factors = get_generated_factor_expressions()
    if not factors:
        print("No generated factors found to evaluate.")
        return

    print(f"Found {len(factors)} factors.")

    # Prepare Fields
    fields = list(factors.values())
    names = list(factors.keys())

    # Add Label
    label_field = "Ref($close, -5)/$close - 1"
    fields.append(label_field)

    # Fetch Data
    # instruments = "csi300" # Or "all"? In config we use "all" or "csi300"
    # Ideally use S&P 500 tickers.
    # D.instruments(market='us') might return all US stocks.
    # Let's try to filter using the tickers we downloaded if possible, or just 'all'?
    # 'all' might be slow if we have many tickers.
    # S&P 500 tickers are in 'raw_data' filenames?
    # Qlib 'market' usually defines the universe.
    # Let's use 'all' for now, or filter by existing data.
    instruments = D.instruments(market="all")

    try:
        print("Fetching data (this might take a moment)...")
        df = D.features(instruments, fields, start_time=valid_start, end_time=valid_end)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print("No data fetched.")
        return

    # Rename columns to factor names
    # D.features returns MultiIndex columns if passed list?
    # Usually it returns columns named by the expression string.
    # We map expression -> name

    # Create mapping: expression -> name
    expr_to_name = {v: k for k, v in factors.items()}
    expr_to_name[label_field] = "label"

    df.columns = [expr_to_name.get(c, c) for c in df.columns]

    # Calculate IC / RankIC
    results = []

    print("Calculating metrics...")
    for name, expr in factors.items():
        if name not in df.columns or "label" not in df.columns:
            print(f"Missing data for {name}")
            continue

        # Drop NaNs for pair
        valid_data = df[[name, "label"]].dropna()
        if valid_data.empty:
            continue

        # Group by date
        ic_ts = valid_data.groupby("datetime").apply(lambda x: x[name].corr(x["label"]))
        rank_ic_ts = valid_data.groupby("datetime").apply(
            lambda x: x[name].corr(x["label"], method="spearman")
        )

        ic_mean = ic_ts.mean()
        ic_std = ic_ts.std()
        rank_ic_mean = rank_ic_ts.mean()
        rank_ic_std = rank_ic_ts.std()

        # Information Ratio (ICIR)
        icir = ic_mean / ic_std if ic_std != 0 else 0

        results.append(
            {
                "name": name,
                "expression": expr,
                "ic_mean": ic_mean,
                "rank_ic_mean": rank_ic_mean,
                "icir": icir,
            }
        )

    # Create Leaderboard
    leaderboard = pd.DataFrame(results)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("rank_ic_mean", ascending=False)
        print("\n=== Factor Leaderboard (Top 10) ===")
        print(leaderboard.head(10)[["name", "ic_mean", "rank_ic_mean", "icir"]])

        output_file = "analyzed_factors.csv"
        leaderboard.to_csv(output_file, index=False)
        print(f"\nSaved analysis to {output_file}")
    else:
        print("No valid results.")


if __name__ == "__main__":
    evaluate_factors()
