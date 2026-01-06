import sys
import os
import asyncio
import subprocess
from dotenv import load_dotenv

# Load env vars FIRST
load_dotenv()
os.environ["LITELLM_LOG"] = "DEBUG"

from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
import rdagent.scenarios.qlib.experiment.factor_experiment as factor_experiment_module
import rdagent.scenarios.qlib.experiment.utils as utils_module
import rdagent.utils.env as env_module
import rdagent.scenarios.qlib.experiment.workspace as workspace_module

# --- Patching to bypass Docker and inject local data info ---


def patched_get_data_folder_intro():
    import os

    cwd = os.getcwd()
    return f"""
The Qlib data is stored locally at `{cwd}/qlib_data`.
You MUST initialize Qlib with the following configuration in your code:
```python
import qlib
qlib.init(provider_uri='qlib_data', region='us')
```

The data covers S&P 500 stocks. 
The available fields are: `open`, `high`, `low`, `close`, `volume`, `adjclose`, `vwap`.
The data frequency is daily.

**CRITICAL IMPLEMENTATION DETAILS**:
1. When using `D.features`, you **MUST NOT** pass the string 'all' directly to `instruments`.
2. You **MUST** first get the instrument list using `D.instruments(market='all')`.
3. **Correct Usage**:
   ```python
   instruments = D.instruments(market='all')
   factor_data = D.features(instruments, fields=[...], ...)
   ```
4. **Incorrect Usage** (Will Error):
   ```python
   D.features(instruments='all', ...) # DO NOT DO THIS
   ```
5. **CRITICAL OUTPUT FORMAT**:
   - `D.features` returns index as `(instrument, datetime)`.
   - You **MUST** return index as `(datetime, instrument)`.
   - **ALWAYS** apply this transformation before returning/saving:
     ```python
     factor_data = factor_data.swaplevel().sort_index()
     ```
   - If you do not swap level, the index will be wrong.
6. **MANDATORY FACTOR EXPRESSION**:
   - You **MUST** define the Qlib expression string in a variable named `factor_expression`.
   - Example: `factor_expression = 'Mean($close, 5) / $close'`
   - **DO NOT** use pure Pandas calculation if the logic can be expressed in Qlib.
   - The system RELIES on extracting `factor_expression` for evaluation.
   - Failure to define `factor_expression` means your factor will be ignored.

**TRADING SCENARIO (WEEKLY PREDICTION)**:
- **Goal**: Predict **Weekly Returns** (5-day holding period).
- **Focus**: The Agent should prioritize factors that capture **medium-term trends** (5 to 20 days).
- **Avoid**: Purely intraday or 1-day signals (noise).
- **Suggestions**:
  - Use `Ref(x, 5)` to compare with last week.
  - Use `Mean(x, 10)` or `Mean(x, 20)` for trends.
  - Look for weekly volume accumulations or price breakouts.
"""


# Apply the patch to get_data_folder_intro
utils_module.get_data_folder_intro = patched_get_data_folder_intro
factor_experiment_module.get_data_folder_intro = patched_get_data_folder_intro

# --- Patching Environment to run locally ---


class LocalMockEnv:
    def __init__(self, conf=None):
        self.conf = conf

    def prepare(self):
        print("MockEnv: prepare() called - doing nothing (local env assumed ready)")
        pass

    def check_output(self, entry, env=None, local_path=None):
        print(f"MockEnv: Executing command locally: {entry}")

        run_env = os.environ.copy()
        # if env:
        #    for k, v in env.items():
        #        run_env[k] = str(v)

        # Explicitly ensure PYTHONPATH matches current environment or is empty (letting sys.executable decide)
        # Verify if PYTHONPATH needs filtering. Usually os.environ copy is enough.
        print(f"MockEnv: ignoring passed env vars to prevent PYTHONPATH corruption.")

        cwd = local_path if local_path else os.getcwd()

        # Clean up entry command
        entry_clean = entry.strip()

        # Remove 'qrun' prefix if present from RD-Agent (since we are running locally without qrun wrapper)
        if entry_clean.startswith("qrun "):
            entry_clean = entry_clean[5:].strip()
            print(f"MockEnv: Stripped 'qrun' prefix. New entry: {entry_clean}")

        # FORCE use of current python executable
        if entry_clean.startswith("python "):
            entry_clean = f"{sys.executable} {entry_clean[7:]}"
        elif entry_clean == "python":
            entry_clean = sys.executable

        print(f"MockEnv: Final executable entry: {entry_clean}")

        if entry_clean.lower().endswith(".yaml"):
            print(
                f"MockEnv: Detected YAML file reference: {entry_clean}. Assuming configuration generation success."
            )
            # Check if file exists, if so read it
            if os.path.exists(entry_clean):
                with open(entry_clean, "r") as f:
                    content = f.read()
                return content
            else:
                # If it doesn't exist, maybe it was just a file name string passed
                return f"Configuration file {entry_clean} accepted."

        try:
            # Use shell=True for complex commands
            output = subprocess.check_output(
                entry_clean, shell=True, env=run_env, cwd=cwd, stderr=subprocess.STDOUT
            )
            return output.decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e.output.decode('utf-8')}")
            return f"Execution Failed:\n{e.output.decode('utf-8')}"


# Patch QlibCondaEnv and QTDockerEnv with our Mock locally in module where they are used
# env_module.QlibCondaEnv = LocalMockEnv
# env_module.QTDockerEnv = LocalMockEnv
# workspace_module.QlibCondaEnv = LocalMockEnv
# workspace_module.QTDockerEnv = LocalMockEnv

# ------------------------------------------------------------


def main():
    print("Starting Agentic Factor Generation (Patched via Env + Workspace)...")

    try:
        loop = FactorRDLoop(FACTOR_PROP_SETTING)
        asyncio.run(loop.run(loop_n=1))
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
