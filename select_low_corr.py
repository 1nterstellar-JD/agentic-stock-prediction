import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from factor_loader import get_generated_factor_expressions
import pandas as pd
import numpy as np
import json
import os
import yaml


def select_low_corr(top_k=10, period="1y"):
    print("Selecting Low-Correlation Factors...")

    # 1. Init Config (load from current config.yaml to get qlib path)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    qlib_config = config.get("qlib_init", {})
    if "provider_uri" in qlib_config:
        if not os.path.isabs(qlib_config["provider_uri"]):
            qlib_config["provider_uri"] = os.path.join(
                os.getcwd(), qlib_config["provider_uri"]
            )

    # Avoid re-init if already initialized (though script usually runs cleanly)
    try:
        qlib.init(**qlib_config)
    except:
        pass

    # 2. Get Factor Expressions
    # RD-Agent Factors
    gen_factors_dict = get_generated_factor_expressions(
        analysis_path="analyzed_factors.csv"
    )
    if not gen_factors_dict:
        print("No generated factors found. Aborting.")
        return

    # Alpha158 Factors
    tmp = Alpha158(instruments="all", start_time="2020-01-01", end_time="2020-01-02")
    alpha_config = tmp.get_feature_config()
    if isinstance(alpha_config, tuple):
        alpha_exprs, alpha_names = alpha_config
    else:
        alpha_exprs, alpha_names = alpha_config, alpha_config  # List fallback

    # 3. Load Data for Correlation Calculation
    end_time = "2025-01-01"
    start_time = "2024-01-01"

    # Get instruments list explicitly
    try:
        instruments = D.instruments(market="all")
    except Exception as e:
        print(
            f"Error getting instruments: {e}. Fallback to 'all' data might fail if not fully supported."
        )
        instruments = "all"

    print(f"Loading data for correlation ({start_time} to {end_time})...")

    # Load RD-Agent Data
    fields_rd = list(gen_factors_dict.values())
    names_rd = list(gen_factors_dict.keys())

    # Load Alpha158 Data
    fields_alpha = alpha_exprs
    names_alpha = alpha_names

    # We need to load them into a single DF to align valid indices
    all_fields = fields_rd + fields_alpha
    all_names = names_rd + names_alpha

    # Use D.features
    try:
        df = D.features(
            instruments, all_fields, start_time=start_time, end_time=end_time
        )
        df.columns = all_names
    except Exception as e:
        print(f"Error loading features: {e}")
        return

    print("Calculating Correlation Matrix...")
    corr_matrix = df.corr()

    # We want corr between {Alpha158} and {RD-Agent}
    # Rows: Alpha158, Cols: RD-Agent
    sub_corr = corr_matrix.loc[names_alpha, names_rd]

    # 4. Selection Logic
    # For each Alpha158 factor, find max absolute correlation with any RD factor.
    max_abs_corr = sub_corr.abs().max(axis=1)

    # Sort ascending (lowest max correlation)
    sorted_factors = max_abs_corr.sort_values(ascending=True)

    selected_names = sorted_factors.head(top_k).index.tolist()

    print(f"Selected {top_k} Low-Correlation Factors:")
    for name in selected_names:
        print(f"  {name}: {sorted_factors[name]:.4f}")

    # Map back to expressions
    selected_data = {}

    # Ensure aligned lookup
    if len(alpha_names) == len(alpha_exprs):
        name_to_expr = dict(zip(alpha_names, alpha_exprs))
        for name in selected_names:
            selected_data[name] = name_to_expr[name]
    else:
        print(
            "Warning: Alpha158 Names/Exprs length mismatch. Selection might be buggy."
        )

    # Save to JSON
    with open("low_corr_factors.json", "w") as f:
        json.dump(selected_data, f, indent=4)

    print("Saved to low_corr_factors.json")


if __name__ == "__main__":
    select_low_corr()
