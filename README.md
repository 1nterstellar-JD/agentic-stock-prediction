# AI-Driven Stock Prediction Pipeline

This project implements an end-to-end AI stock prediction system capable of generating alpha factors, training machine learning models, and simulating weekly trading strategies for the US market (S&P 500).

Built with **Microsoft Qlib** and **CatBoost**, and enhanced by **RD-Agent** for automated factor mining.

## Features

- **Agentic Factor Generation**: Uses LLMs to automatically discover and write valid Qlib alpha factors.
- **Robust Modeling**: Implements **CatBoost** (Gradient Boosting) optimized for financial tabular data and CPU inference.
- **Automated Workflow**: One-click script (`manual_run.sh`) handles data updates, feature engineering, training, and prediction.
- **Production Metrics**: Automatically calculates **IC/RankIC**, **Annualized Returns**, **Benchmark Comparison (vs S&P 500)**, and **Max Drawdown**.
- **Time-Travel Debugging**: Support for targeting specific historical dates to verify past predictions.

## Environment & Installation

This project is designed to run within the **RD-Agent** conda environment.

```bash
# Activate the environment
conda activate rdagent

# Install dependencies
pip install qlib catboost xgboost pyyaml pandas numpy
```

## Quick Start

### 1. Run Default Prediction (Today)
To generate predictions for the current week:
```bash
bash manual_run.sh
```
*Outputs `weekly_ranking.csv` with the top stock picks.*

### 2. Run Historical Prediction (Time Travel)
To see what the model would have predicted on a specific past date (e.g., Dec 31, 2025):
```bash
bash manual_run.sh 2025-12-31
```

### 3. Advanced: Automated Factor Mining
To automatically discover more factors using the AI Agent in a loop:
```bash
python run_pipeline.py
```
*This script will:*
1. *Iteratively invoke the RD-Agent to propose new factors.*
2. *Evaluate each factor's IC and RankIC.*
3. *Add valid factors (Top 40) to `analyzed_factors.csv`.*
4. *Automatically retrain the model with the improved factor set.*

## Project Structure

- **`manual_run.sh`**: Main entry point script. Orchestrates the entire pipeline.
- **`run_prediction.py`**: Core Python script for fetching data, training the model, and running inference.
- **`agent_factor_gen.py`**: AI agent script that mines new factors using RD-Agent.
- **`factor_loader.py`**: Utility to load and filter high-quality factors (RankIC > 0.01).
- **`config.yaml`**: Configuration for Qlib initialization and model hyperparameters.
- **`weekly_ranking.csv`**: The latest output file containing stock rankings and scores.
- **`results/`**: Archive of past runs, including logs, metrics, and rankings.

## Performance

Performance metrics are printed to the console after every run:

- **IC / RankIC**: Measures the correlation between predictions and future returns.
- **Annualized Return**: Expected yearly return of a Top-50 Equal-Weight Strategy.
- **Excess Return**: Performance relative to the S&P 500 benchmark.
- **Max Drawdown**: The largest peak-to-trough decline in the test period.

## Configuration

Modify `config.yaml` to adjust:
- **`model`**: Switch between CatBoost/LightGBM or tune hyperparameters.
- **`qlib_init`**: Change data provider path or region.
