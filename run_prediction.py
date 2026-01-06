import sys
import os
import yaml
import pandas as pd
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from factor_loader import get_generated_factor_expressions
import copy
import datetime
import shutil
import json
import numpy as np

# Add the current directory to sys.path
sys.path.append(os.getcwd())


def run_prediction():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    import qlib

    qlib_config = config.get("qlib_init")
    if "provider_uri" in qlib_config:
        # Convert relative path to absolute
        if not os.path.isabs(qlib_config["provider_uri"]):
            qlib_config["provider_uri"] = os.path.join(
                os.getcwd(), qlib_config["provider_uri"]
            )

    qlib.init(**qlib_config)

    print("Checking for generated factors...")
    gen_factors = get_generated_factor_expressions(analysis_path="analyzed_factors.csv")

    run_config = copy.deepcopy(config)

    # Parse Arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_date", type=str, help="Target date YYYY-MM-DD", default=None
    )
    parser.add_argument(
        "--factor_mode",
        type=str,
        default="rdagent",
        choices=["alpha158", "rdagent", "combined", "rdagent_low_corr"],
        help="Factor set to use: alpha158, rdagent (default), combined, or rdagent_low_corr",
    )
    args, unknown = parser.parse_known_args()

    # --- Automated Date Configuration ---
    # Rolling Window: Train (4y), Valid (1y), Test (Recent 1mo)
    if args.target_date:
        try:
            today = datetime.datetime.strptime(args.target_date, "%Y-%m-%d")
            print(f"Using specified target date: {today.strftime('%Y-%m-%d')}")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return
    else:
        today = datetime.datetime.now()

    train_start = (today - datetime.timedelta(days=365 * 4)).strftime("%Y-%m-%d")
    valid_start = (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    test_start = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    end_date_str = today.strftime("%Y-%m-%d")

    # Fit window for processors (usually matches training data)
    fit_start_time = train_start
    fit_end_time = valid_start

    print(f"Auto-Configuring Dates (Rolling Window):")
    print(f"  Train: {train_start} to {valid_start}")
    print(f"  Valid: {valid_start} to {test_start}")
    print(f"  Test : {test_start} to {end_date_str}")

    # Update Task Segments
    if "task" in run_config and "dataset" in run_config["task"]:
        run_config["task"]["dataset"]["kwargs"]["segments"] = {
            "train": [train_start, valid_start],
            "valid": [valid_start, test_start],
            "test": [test_start, end_date_str],
        }

    # Update Data Handler Config (in config.yaml it might be a shared anchor, but here we modify the dict)
    # The handler config in run_config is likely nested.
    # We update the 'kwargs' passed to the handler init.
    # Note: data_handler_config anchor might be expanded by yaml.safe_load, so we modify where it's used.
    # It is used in task.dataset.kwargs.handler.kwargs

    # However, we are about to overwrite the handler below if gen_factors exist.
    # So we should store these times to use them in the new_handler.

    # Determine Handler Configuration based on Factor Mode
    new_handler = None
    factor_mode = args.factor_mode
    print(f"Configuring for Factor Mode: {factor_mode}")

    # Common Processors
    infer_processors = [{"class": "Fillna", "kwargs": {"fields_group": "feature"}}]
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]

    if factor_mode == "alpha158":
        new_handler = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": end_date_str,
                "instruments": "all",
                "infer_processors": infer_processors,
                "learn_processors": learn_processors,
            },
        }
    elif factor_mode == "combined":
        new_handler = {
            "class": "CombinedFactorHandler",
            "module_path": "custom_handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": end_date_str,
                "instruments": "all",
                "analysis_path": "analyzed_factors.csv",
                "infer_processors": infer_processors,
                "learn_processors": learn_processors,
            },
        }
    elif factor_mode == "rdagent_low_corr":
        # Ensure json exists
        low_corr_path = "low_corr_factors.json"
        if not os.path.exists(low_corr_path):
            print("low_corr_factors.json not found. Running selection script...")
            sel_res = os.system(
                "/home/ec2-user/miniforge3/envs/rdagent/bin/python select_low_corr.py"
            )
            if sel_res != 0:
                print("Factor selection failed.")
                sys.exit(1)

        new_handler = {
            "class": "LowCorrFactorHandler",
            "module_path": "custom_handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": end_date_str,
                "instruments": "all",
                "analysis_path": "analyzed_factors.csv",
                "low_corr_path": low_corr_path,
                "infer_processors": infer_processors,
                "learn_processors": learn_processors,
            },
        }

    elif factor_mode == "rdagent":
        if gen_factors:
            print(
                f"Found {len(gen_factors)} generated factors: {list(gen_factors.keys())}"
            )
            new_handler = {
                "class": "GenFactorHandler",
                "module_path": "custom_handler",
                "kwargs": {
                    "start_time": train_start,
                    "end_time": end_date_str,
                    "instruments": "all",
                    "analysis_path": "analyzed_factors.csv",
                    "infer_processors": infer_processors,
                    "learn_processors": learn_processors,
                },
            }
        else:
            print("No generated factors found for RD-Agent mode.")

    if new_handler:
        # Modify config to use the selected handler
        if "task" in run_config and "dataset" in run_config["task"]:
            ds_kwargs = run_config["task"]["dataset"].get("kwargs", {})

            # Conserve Label
            label = [
                "Ref($close, -5) / $close - 1",
                "Ref($close, -5) / $close - 1",
            ]  # Default
            if isinstance(ds_kwargs.get("handler"), dict):
                label = ds_kwargs["handler"].get("kwargs", {}).get("label", label)
            elif hasattr(ds_kwargs.get("handler"), "kwargs"):
                pass
            new_handler["kwargs"]["label"] = label

            # Conserve Instruments
            orig_handler = ds_kwargs.get("handler")
            if isinstance(orig_handler, dict) and "kwargs" in orig_handler:
                h_kwargs = orig_handler["kwargs"]
                if "instruments" in h_kwargs:
                    new_handler["kwargs"]["instruments"] = h_kwargs["instruments"]

            run_config["task"]["dataset"]["kwargs"]["handler"] = new_handler
            print(
                f"Config updated to use {factor_mode} via handler: {new_handler['class']}."
            )
        else:
            print("Config structure unexpected. Using defaults.")
    else:
        print(
            "No generated factors found. Using original config (updating dates only)."
        )
        # Update original handler kwargs with new dates
        if "task" in run_config and "dataset" in run_config["task"]:
            h_kwargs = (
                run_config["task"]["dataset"]["kwargs"]
                .get("handler", {})
                .get("kwargs", {})
            )
            h_kwargs["start_time"] = train_start
            h_kwargs["end_time"] = end_date_str
            h_kwargs["fit_start_time"] = fit_start_time
            h_kwargs["fit_end_time"] = fit_end_time
            # Also update infer processors fit times if accessible
            if "infer_processors" in h_kwargs:
                for proc in h_kwargs["infer_processors"]:
                    if proc.get("class") == "RobustZScoreNorm":
                        proc["kwargs"]["fit_start_time"] = fit_start_time
                        proc["kwargs"]["fit_end_time"] = fit_end_time

    # Save runtime config
    with open("config_runtime.yaml", "w") as f:
        yaml.dump(run_config, f)

    print("Starting Qlib workflow...")
    qrun_path = os.path.join(os.path.dirname(sys.executable), "qrun")

    # Ensure CWD is in PYTHONPATH so qlib can find custom_handler
    cwd = os.getcwd()
    cmd = f"PYTHONPATH={cwd} {qrun_path} config_runtime.yaml"
    print(f"Executing: {cmd}")
    ret = os.system(cmd)

    if ret != 0:
        print("Qlib workflow failed.")
        sys.exit(1)

    # Process Results
    try:
        exp = R.get_exp(experiment_name="workflow")  # Default name
        recorders = R.list_recorders(exp.id)
        # Sort recorders by start_time to ensure we get the latest one
        recorders_list = sorted(
            [r for r in recorders.values()],
            key=lambda x: x.info["start_time"] if "start_time" in x.info else 0,
        )
        recorder = recorders_list[-1]
        print(
            f"Using latest recorder: {recorder.id} (Started: {recorder.info.get('start_time')})"
        )

        # --- DEDUCTED INFERENCE: PREDICT ON LATEST DATA ---
        print("Running Deducted Inference (Predicting on latest available data)...")
        from qlib.utils import init_instance_by_config
        from qlib.data.dataset import DatasetH

        # Prepare inference handler config (Clone of new_handler but WITHOUT DropnaLabel)
        inf_handler_config = copy.deepcopy(new_handler)

        # Override time to ensure we cover the absolute latest data
        # Using test_start to end_date_str (today)
        # CRITICAL FIX: Shift start_time back by 60 days to allow factor lookback calculation
        # Otherwise, factors like Mean($close, 20) will be NaN at the start of the window
        dt_test_start = datetime.datetime.strptime(test_start, "%Y-%m-%d")
        inf_start_time = (dt_test_start - datetime.timedelta(days=60)).strftime(
            "%Y-%m-%d"
        )

        inf_handler_config["kwargs"]["start_time"] = inf_start_time
        inf_handler_config["kwargs"]["end_time"] = end_date_str

        # Remove DropnaLabel from learn_processors if present
        l_procs = inf_handler_config["kwargs"].get("learn_processors", [])
        inf_handler_config["kwargs"]["learn_processors"] = [
            p for p in l_procs if p.get("class") != "DropnaLabel"
        ]

        # Initialize Dataset for Inference
        inf_dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": inf_handler_config,
                "segments": {
                    "test": [test_start, end_date_str],
                },
            },
        }

        inf_dataset = DatasetH(
            handler=init_instance_by_config(inf_handler_config),
            segments={"test": (test_start, end_date_str)},
        )

        # --- DEBUG: Inspect Inference Data ---
        print("DEBUG: Inspecting Inference Data (features passed to model)...")
        try:
            df_inf_debug = inf_dataset.prepare("test")
            print(f"Inference Data Shape: {df_inf_debug.shape}")
            print(f"Inference Data Tail:\n{df_inf_debug.tail(10)}")

            # Check for constant features on last day
            last_idx = df_inf_debug.index.get_level_values("datetime").max()
            print(f"Checking features for {last_idx}:")
            df_last = df_inf_debug.loc[last_idx]
            print(df_last.head())
            print("Is Constant?", (df_last.std() == 0).all())
        except Exception as e:
            print(f"DEBUG Error: {e}")
        # -------------------------------------

        # Load Model
        model = recorder.load_object("params.pkl")

        # Predict
        # DatasetH.prepare yields (tsds, etc), usually model.predict takes dataset object directly
        # For DatasetH, we often use prepare("test")
        inf_pred_score = model.predict(inf_dataset)

        # inf_pred_score is usually a Series or DataFrame index by (datetime, instrument)
        # Convert to DataFrame if needed
        if isinstance(inf_pred_score, pd.Series):
            inf_pred_score = inf_pred_score.to_frame("score")

        print(
            f"Inference Prediction Range: {inf_pred_score.index.get_level_values('datetime').min()} to {inf_pred_score.index.get_level_values('datetime').max()}"
        )

        # Use THIS as the primary source for ranking
        pred_df = inf_pred_score

        if "datetime" in pred_df.index.names:
            last_date = pred_df.index.get_level_values("datetime").max()
            print(
                f"DEBUG: Pred DF Index Tail: {pred_df.index.get_level_values('datetime').unique().sort_values()[-5:]}"
            )
            print(f"Latest prediction date: {last_date}")

            latest_preds = pred_df[
                pred_df.index.get_level_values("datetime") == last_date
            ].copy()
            latest_preds = latest_preds.sort_values(by="score", ascending=False)

            output_file = "weekly_ranking.csv"
            latest_preds.to_csv(output_file)
            print(f"Ranking saved to {output_file}")
            print(latest_preds.head(10))

            # --- Result Archiving ---
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join("results", f"Run_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)

            # Copy Ranking
            shutil.copy(output_file, os.path.join(result_dir, output_file))
            # Copy Config
            shutil.copy("config_runtime.yaml", os.path.join(result_dir, "config.yaml"))

            print(f"Results archived to: {result_dir}")

            # --- Metrics Calculation ---
            # Calculate IC and RankIC on Valid/Test data
            if "label" in pred_df.columns:
                # Filter for test segment (dates where label is known, i.e., not the very last 5 days)
                # Actually, pred_df usually contains all requested segments.
                # We care about the test/validation performance.

                # Align prediction and label
                # Note: SignalRecord usually saves 'pred' column. Label is sometimes separate or included.
                # If label is not in pred_df, we try to load it.
                pass

            # Try to load label (from original recorder to calculate metrics on VALID data)
            # Since our pred_df is now purely inference (recent), we might not have matched labels for all of it.
            # We should try to use the ORIGINAL pred.pkl for metrics if possible, OR load labels for the inference period.
            try:
                # Load original predictions for valid/test performance evaluation
                orig_pred = recorder.load_object("pred.pkl")
                label_df = recorder.load_object("label.pkl")

                # Join
                combined = pd.concat([orig_pred, label_df], axis=1, join="inner")
                combined.columns = ["score", "label"]

                # Calc IC per day
                ic = combined.groupby("datetime").apply(
                    lambda x: x["score"].corr(x["label"])
                )
                rank_ic = combined.groupby("datetime").apply(
                    lambda x: x["score"].corr(x["label"], method="spearman")
                )

                metrics = {
                    "IC_Mean": ic.mean(),
                    "IC_Std": ic.std(),
                    "RankIC_Mean": rank_ic.mean(),
                    "RankIC_Std": rank_ic.std(),
                    "Latest_Date": str(last_date),
                }

                # Calc Weekly Return (Top 50)
                # Note: This is an approximation using the 'label' (Ref($close, -5)/$close - 1)
                # For the latest date, label is NaN, so it contributes 0 (or is skipped) to the historical average calculation.
                def get_topk_return(group, k=50):
                    # We only calculate return for dates where label exists
                    if group["label"].isnull().all():
                        return np.nan
                    return (
                        group.sort_values("score", ascending=False)
                        .head(k)["label"]
                        .mean()
                    )

                topk_ret = combined.groupby("datetime").apply(get_topk_return)
                annualized_return = topk_ret.mean() * 52

                try:
                    # Try loading Portfolio Analysis Record (calculated by Qlib backtest)
                    port_ana = recorder.load_object("port_analysis_1day.pkl")
                    # port_ana is a tuple/list: (returns_df, risk_df) usually?
                    # Actually Qlib saves it as a dict or tuple.
                    # Based on logs: "The following are analysis results...".
                    # It seems to be (report_df, risk_analysis).
                    # Let's try to load it and infer.

                    # Usually: [0] is the dataframe of daily returns/risks? [1] is the summary dict?
                    # Let's rely on the previous logic or just print the keys if it's a dict.
                    # Qlib's PortAnaRecord.generate() returns a list of results.
                    # The artifact is likely the return of `backtest`.

                    # Actually, let's look at the risk summary directly if available.
                    # Or simpler: Re-calculate from combined['score']? No, that's raw score.

                    # Let's inspect the pickle content structure via code
                    pass
                except:
                    port_ana = None

                # Calculate manually to be safe and specific about "Top 50"
                # (Qlib's default backtest might use TopK 50 if configured, but let's confirm).

                # --- Manual Detailed Metrics ---
                # 1. Strategy Return (Top 50 Equal Weight)
                # 2. Benchmark Return (^GSPC)

                # We need Benchmark data.
                from qlib.data import D

                # Load Benchmark for the same period
                # combined.index is (datetime, instrument). We need to extract the date range.
                min_date = combined.index.get_level_values("datetime").min()
                max_date = combined.index.get_level_values("datetime").max()

                # Check directly if they are already strings or Timestamps
                if hasattr(min_date, "date"):
                    min_date_str = str(min_date.date())
                else:
                    min_date_str = str(min_date).split(" ")[0]  # Fallback

                if hasattr(max_date, "date"):
                    max_date_str = str(max_date.date())
                else:
                    max_date_str = str(max_date).split(" ")[0]

                # Use Ref($close, 1) for previous day close.
                # Expression: $close / Ref($close, 1) - 1
                bench_df = D.features(
                    ["^GSPC"],
                    ["$close / Ref($close, 1) - 1"],
                    start_time=min_date_str,
                    end_time=max_date_str,
                )
                bench_df.columns = ["bench_ret"]

                # Calculate Strategy Daily Return
                # We need to simulate the strategy: hold Top 50 *Next Day*.
                # 'label' is 5-day return?
                # If label is Ref($close, -5)/$close - 1, it is 5-day forward return.
                # To get daily curve, we need daily labels.

                # Simplified Approach: Use the Annualized Return we already calculate (based on 5-day horizons)
                # And compare to Benchmark's Annualized Return over same period.

                # Benchmark Annualized (using same 5-day logic for fairness)
                # Wait, we can't easily get 5-day label for benchmark from 'combined' as it only has universe stocks.
                # We typically compare Annualized Returns.

                bench_ret_annual = 0.0
                if not bench_df.empty:
                    # mean daily * 252 (approx)
                    bench_ret_annual = bench_df["bench_ret"].mean() * 252

                # Max Drawdown
                # We need a cumulative return curve.
                # Since we only have weekly samples (approx), let's construct a weekly curve.
                strategy_weekly_ret = combined.groupby("datetime").apply(
                    get_topk_return
                )
                strategy_cum = (1 + strategy_weekly_ret).cumprod()

                # Drawdown
                running_max = strategy_cum.cummax()
                drawdown = (strategy_cum - running_max) / running_max
                max_dd = drawdown.min()
                max_dd_date = drawdown.idxmin()

                metrics = {
                    "IC_Mean": float(ic.mean()),
                    "IC_Std": float(ic.std()),
                    "RankIC_Mean": float(rank_ic.mean()),
                    "RankIC_Std": float(rank_ic.std()),
                    "Annualized_Return": float(annualized_return),
                    "Benchmark_Return": float(bench_ret_annual),
                    "Max_Drawdown": float(max_dd),
                    "Max_Drawdown_Date": (
                        str(max_dd_date) if not pd.isna(max_dd_date) else None
                    ),
                    "Latest_Date": str(last_date),
                }

                print("\nPerformance Metrics (Test Window):")
                print(f"  IC Mean: {metrics['IC_Mean']:.4f}")
                print(f"  RankIC Mean: {metrics['RankIC_Mean']:.4f}")
                print(
                    f"  Annualized Return (Top 50): {metrics['Annualized_Return']:.2%}"
                )
                print(
                    f"  Benchmark Return (S&P 500): {metrics['Benchmark_Return']:.2%} (Excess: {metrics['Annualized_Return'] - metrics['Benchmark_Return']:.2%})"
                )
                print(
                    f"  Max Drawdown: {metrics['Max_Drawdown']:.2%} (occurred on {metrics['Max_Drawdown_Date']})"
                )

                with open(os.path.join(result_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4)

            except Exception as e:
                print(f"Metrics calculation failed (label might be missing): {e}")

            print("\nMethodology:")
            print("- Loaded Generated Factors from CSV.")
            print("- Used GenFactorHandler to dynamically inject factors into Qlib.")
            print("- Trained model and generated predictions.")

        else:
            print("datetime level not found in index.")

    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_prediction()
