import os
import pandas as pd
import sys
import argparse

import datetime

# Config
TARGET_FACTOR_COUNT = 40
MAX_LOOPS = 5  # Prevent infinite loop
PYTHON_EXE = os.path.expanduser("~/miniforge3/envs/rdagent/bin/python")


def run_command(cmd_args, description):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] --- {description} ---")
    cmd = f"{PYTHON_EXE} {' '.join(cmd_args)}"
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    now_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if ret != 0:
        print(f"[{now_end}] Error executing {description}. Exit code: {ret}")
        # We don't raise exception for evaluating/generation failure to allow retries,
        # unless it's critical.
        return False

    print(f"[{now_end}] Finished {description}")
    return True


def count_good_factors(analysis_path="analyzed_factors.csv"):
    if not os.path.exists(analysis_path):
        return 0
    try:
        df = pd.read_csv(analysis_path)
        # We assume all listed factors are 'valid' enough to be in the list
        # (evaluate_factors.py handles calculation).
        # We could filter by RankIC > threshold here if desired.
        # For now, just count.
        return len(df)
    except Exception as e:
        print(f"Error reading analysis file: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Automated Stock Prediction Pipeline")
    parser.add_argument(
        "--max_loops",
        type=int,
        default=MAX_LOOPS,
        help="Max loops for factor generation",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_FACTOR_COUNT, help="Target factor count"
    )
    args = parser.parse_args()

    # Step 1: Update Data
    if not run_command(["download_data.py"], "Step 1: Updating Data"):
        print("Data update failed. Proceeding with existing data...")

    # Step 2: Factor Expansion Loop
    loop = 0
    while loop < args.max_loops:
        # Evaluate current state
        run_command(["evaluate_factors.py"], "Evaluating Factors")

        count = count_good_factors()
        print(f"\n[Status] Current Valid Factors: {count} / {args.target}")

        if count >= args.target:
            print("Target factor count reached!")
            break

        print(f"\n[Loop {loop+1}/{args.max_loops}] Generating new factors...")

        # Generation
        if not run_command(["agent_factor_gen.py"], "Factor Generation"):
            print("Factor generation encountered an issue. Continuing...")

        # Extraction (Append)
        run_command(["extract_factors.py"], "Extracting Factors")

        loop += 1

    if loop == args.max_loops:
        print("Warning: Max loops reached without hitting target count.")

    # Final Evaluation to ensure analyzed_factors.csv is up to date and sorted
    run_command(["evaluate_factors.py"], "Final Factor Evaluation")

    # Step 3: Prediction
    print(f"\n--- Step 3: Running Prediction with Top {args.target} Factors ---")
    run_command(["run_prediction.py"], "Prediction Pipeline")

    # Step 4: Summary
    print("\n=== Pipeline Complete ===")
    if os.path.exists("weekly_ranking.csv"):
        print("Ranking saved to: weekly_ranking.csv")
        # Print top 5
        try:
            df = pd.read_csv("weekly_ranking.csv")
            print(df.head())
        except:
            pass

    if os.path.exists("analyzed_factors.csv"):
        print("Leaderboard: analyzed_factors.csv")


if __name__ == "__main__":
    main()
