#!/bin/bash
set -e

# Create a timestamp for this comparison run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_RESULT_DIR="results/Comparison_${TIMESTAMP}"
mkdir -p "$BASE_RESULT_DIR"

echo "Starting Factor Comparison Run..."
echo "Results will be saved to: $BASE_RESULT_DIR"

# 1. Run Alpha158
echo "--------------------------------------------------"
echo "Running Mode: Alpha158"
/home/ec2-user/miniforge3/envs/rdagent/bin/python run_prediction.py --factor_mode alpha158 || exit 1
# Move the latest result to our comparison folder
LATEST_RUN=$(ls -td results/Run_* | head -1)
cp -r "$LATEST_RUN" "$BASE_RESULT_DIR/Alpha158"
echo "Alpha158 results copied to $BASE_RESULT_DIR/Alpha158"

# 2. Run RD-Agent
echo "--------------------------------------------------"
echo "Running Mode: RD-Agent"
/home/ec2-user/miniforge3/envs/rdagent/bin/python run_prediction.py --factor_mode rdagent || exit 1
LATEST_RUN=$(ls -td results/Run_* | head -1)
cp -r "$LATEST_RUN" "$BASE_RESULT_DIR/RDAgent"
echo "RD-Agent results copied to $BASE_RESULT_DIR/RDAgent"

# 3. Run Combined
echo "--------------------------------------------------"
echo "Running Mode: Combined"
/home/ec2-user/miniforge3/envs/rdagent/bin/python run_prediction.py --factor_mode combined || exit 1
LATEST_RUN=$(ls -td results/Run_* | head -1)
cp -r "$LATEST_RUN" "$BASE_RESULT_DIR/Combined"
echo "Combined results copied to $BASE_RESULT_DIR/Combined"

# 4. Run Low-Corr
echo "--------------------------------------------------"
echo "Running Mode: Low-Corr"
/home/ec2-user/miniforge3/envs/rdagent/bin/python run_prediction.py --factor_mode rdagent_low_corr || exit 1
LATEST_RUN=$(ls -td results/Run_* | head -1)
cp -r "$LATEST_RUN" "$BASE_RESULT_DIR/LowCorr"
echo "Low-Corr results copied to $BASE_RESULT_DIR/LowCorr"

# Summary
echo "--------------------------------------------------"
echo "Comparison Run Completed."
echo "Summary of metrics:"

# Simple JSON extraction
/home/ec2-user/miniforge3/envs/rdagent/bin/python -c "
import json
import os
import pandas as pd
import sys

base_dir = '$BASE_RESULT_DIR'
modes = ['Alpha158', 'RDAgent', 'Combined', 'LowCorr']
results = []

for mode in modes:
    metrics_path = os.path.join(base_dir, mode, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                m = json.load(f)
                m['Mode'] = mode
                results.append(m)
        except Exception as e:
            print(f'Error loading {mode}: {e}')

if results:
    df = pd.DataFrame(results)
    # Reorder columns preference
    cols = ['Mode', 'IC_Mean', 'RankIC_Mean', 'Annualized_Return', 'Max_Drawdown']
    # Add other columns if they exist
    final_cols = [c for c in cols if c in df.columns]
    # Add any remaining columns
    remaining = [c for c in df.columns if c not in final_cols]
    final_cols += remaining
    
    print(df[final_cols].to_string(index=False))
    
    # Also save summary CSV
    df[final_cols].to_csv(os.path.join(base_dir, 'summary.csv'), index=False)
else:
    print('No results found.')
"
