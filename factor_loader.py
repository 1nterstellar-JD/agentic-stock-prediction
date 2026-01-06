import pandas as pd
import re
import os


def get_generated_factor_expressions(
    csv_path="generated_factors.csv", analysis_path=None, top_k=40, min_rank_ic=0.01
):
    """
    Parses the generated factors.
    1. If analysis_path exists, loads top_k factors by rank_ic_mean.
    2. Fallback to csv_path extraction.
    Returns a dict {factor_name: expression}.
    """
    expressions = {}

    # Try loading from analysis leaderboard first
    try:
        if analysis_path and os.path.exists(analysis_path):
            print(f"Loading factors from analysis: {analysis_path}")
            df_ana = pd.read_csv(analysis_path)
            # Sort by RankIC (assuming col name 'rank_ic_mean')
            if "rank_ic_mean" in df_ana.columns:
                df_ana = df_ana.sort_values("rank_ic_mean", ascending=False)

            # Take Top K
            df_ana = df_ana.head(top_k)

            # Filter by Threshold (Absolute RankIC)
            # We want strong signals, whether positive or negative.
            initial_count = len(df_ana)
            df_ana = df_ana[df_ana["rank_ic_mean"].abs() >= min_rank_ic]
            filtered_count = len(df_ana)

            print(
                f"Selected Top {initial_count} factors, filtered to {filtered_count} with |RankIC| >= {min_rank_ic}"
            )

            # We need expressions.
            # If 'expression' column exists in analysis, proper.
            # evaluate_factors.py saves 'expression' column? Yes.
            if "expression" in df_ana.columns:
                for _, row in df_ana.iterrows():
                    expressions[row["name"]] = row["expression"]
                return expressions
            else:
                print(
                    " Analysis file missing 'expression' column. Falling back to generated code extraction."
                )
    except Exception as e:
        print(f"Error reading analysis file: {e}")

    # Fallback / Original Logic
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return {}

    for _, row in df.iterrows():
        name = row["name"]

        # If we already have it (from analysis fallback?), skip
        if name in expressions:
            continue

        code = row["code"]

        # Regex to find factor_expression = '...' (handling indentation)
        match = re.search(
            r"^\s*factor_expression\s*=\s*['\"]([^'\"]+)['\"]", code, re.MULTILINE
        )
        if match:
            expr = match.group(1)
            expressions[name] = expr
        else:
            print(f"Could not extract expression for {name}")

    # Blacklist check
    BLACKLIST = ["ROC", "RSI", "MACD", "EMA"]
    final_expressions = {}
    for name, expr in expressions.items():
        is_safe = True
        for op in BLACKLIST:
            # Check for op( or op ( case insensitive? Qlib usually upper
            if op + "(" in expr or op + " (" in expr:
                print(f"Skipping factor {name} due to unsupported operator: {op}")
                is_safe = False
                break
        if is_safe:
            final_expressions[name] = expr

    return final_expressions


if __name__ == "__main__":
    exprs = get_generated_factor_expressions()
    for name, expr in exprs.items():
        print(f"{name}: {expr}")
