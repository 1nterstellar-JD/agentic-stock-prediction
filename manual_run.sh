#!/bin/bash
# Manual run script for stock prediction
# Activates the rdagent environment and runs the workflow.

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Dynamically find python in the current conda environment or use standard python
PYTHON_EXE=$(which python)
if [ -z "$PYTHON_EXE" ]; then
    PYTHON_EXE="python"
fi
echo "Using Python: $PYTHON_EXE"

TARGET_DATE=$1
if [ -n "$TARGET_DATE" ]; then
    echo "Using target date: $TARGET_DATE"
    DATE_ARG="--target_date $TARGET_DATE"
else
    echo "No target date specified, using current date."
    DATE_ARG=""
fi

echo "Starting Weekly Stock Prediction..."

# 1. Download and Update Data
echo "Updating data (Internet connection required)..."
$PYTHON_EXE download_data.py
if [ $? -ne 0 ]; then
    echo "Data download failed!"
    exit 1
fi

# 1.5. Extract Factors (if any new agent runs occurred)
echo "Checking for generated factors..."
$PYTHON_EXE extract_factors.py

# 1.6. Evaluate Factors (Update Leaderboard)
echo "Evaluating factors..."
$PYTHON_EXE evaluate_factors.py

# 2. Run Prediction
echo "Running prediction workflow..."
$PYTHON_EXE run_prediction.py $DATE_ARG
if [ $? -ne 0 ]; then
    echo "Prediction failed!"
    exit 1
fi

echo "Done! Check weekly_ranking.csv for results."
