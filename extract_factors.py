import os
import pickle
import glob
import re
import pandas as pd
import json

LOG_DIR = "log"


def extract_factors_from_latest_session():
    # Find latest session
    sessions = sorted(
        [
            os.path.join(LOG_DIR, d)
            for d in os.listdir(LOG_DIR)
            if os.path.isdir(os.path.join(LOG_DIR, d))
        ]
    )
    if not sessions:
        print("No sessions found.")
        return

    latest_session = sessions[-1]
    print(f"Analyzing latest session: {latest_session}")

    # Path to coding logs
    coding_dir = os.path.join(latest_session, "Loop_0/coding/evo_loop_0/debug_llm")

    if not os.path.exists(coding_dir):
        print(f"Coding directory not found: {coding_dir}")
        return

    factors_data = []

    # Iterate over subdirectories in debug_llm (each represents a task/factor)
    for task_id in os.listdir(coding_dir):
        task_path = os.path.join(coding_dir, task_id)
        if not os.path.isdir(task_path):
            continue

        pkl_files = sorted(glob.glob(os.path.join(task_path, "*.pkl")))
        if not pkl_files:
            continue

        # Iterate over log files to find the Code Generation step
        # Code Generation usually has "Target factor information" in 'user' prompt
        # and "code" in 'resp' (if successful).

        current_factor = {}

        # We iterate in order or reverse?
        # Reverse is better to get the latest attempt.
        for pkl_file in reversed(pkl_files):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)

                    if not isinstance(data, dict):
                        continue

                    user_content = str(data.get("user", ""))
                    resp_content = str(data.get("resp", ""))

                    if "Target factor information" in user_content:
                        # Parse Metadata
                        if "name" not in current_factor:
                            match_name = re.search(
                                r"factor_name:\s*(\w+)", user_content
                            )
                            if match_name:
                                current_factor["name"] = match_name.group(1)

                            match_desc = re.search(
                                r"factor_description:\s*(.+)", user_content
                            )
                            if match_desc:
                                current_factor["description"] = match_desc.group(
                                    1
                                ).strip()

                            match_form = re.search(
                                r"factor_formulation:\s*(.+)", user_content
                            )
                            if match_form:
                                current_factor["formulation"] = match_form.group(
                                    1
                                ).strip()

                    # Parse Code from Response
                    # The response should contain JSON with "code" key
                    if (
                        '"code":' in resp_content
                        and "final_decision" not in resp_content
                    ):
                        # Extract JSON object
                        # It might be mixed with text.
                        try:
                            # Try to find JSON block
                            match_json = re.search(r"\{.*\}", resp_content, re.DOTALL)
                            if match_json:
                                json_str = match_json.group(0)
                                json_data = json.loads(json_str)
                                if "code" in json_data:
                                    current_factor["code"] = json_data["code"]
                        except:
                            pass

                        # Fallback: simple string search if json parse fails
                        if "code" not in current_factor:
                            # Try to extract content of "code": "..."
                            pass

            except Exception as e:
                pass

            # If we found full info, break
            if "name" in current_factor and "code" in current_factor:
                factors_data.append(current_factor)
                print(f"Extracted Factor: {current_factor['name']}")
                break

    if not factors_data:
        print("No factors extracted.")
    else:
        df_new = pd.DataFrame(factors_data)

        output_csv = "generated_factors.csv"
        if os.path.exists(output_csv):
            try:
                df_old = pd.read_csv(output_csv)
                # Filter out duplicates based on name
                existing_names = set(df_old["name"].tolist())
                # Only keep new ones
                df_new = df_new[~df_new["name"].isin(existing_names)]

                if not df_new.empty:
                    df_final = pd.concat([df_old, df_new], ignore_index=True)
                    print(f" appended {len(df_new)} new factors.")
                else:
                    df_final = df_old
                    print("No new unique factors to append.")
            except Exception as e:
                print(f"Error reading existing CSV: {e}. Overwriting.")
                df_final = df_new
        else:
            df_final = df_new
            print(f"Created new factor file with {len(df_new)} factors.")

        df_final.to_csv(output_csv, index=False)
        print(f"Saved total {len(df_final)} factors to {output_csv}")


if __name__ == "__main__":
    extract_factors_from_latest_session()
