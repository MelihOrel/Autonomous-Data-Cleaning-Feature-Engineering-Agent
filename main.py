"""
main.py

Entry point for the Autonomous Data Cleaning & Feature Engineering Agent.

Run:
    python main.py

The agent will autonomously:
  1. Explore the dirty dataset and identify missing values.
  2. Impute each missing column with the correct strategy.
  3. Compute the Gower distance matrix on the cleaned data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env (must happen before any LangChain / OpenAI import) ─────────────
load_dotenv()

# Make sure the project root is on sys.path so relative imports work when
# the script is executed from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.react_agent import build_agent_executor  # noqa: E402 – after sys.path patch

# ---------------------------------------------------------------------------
# Task query
# ---------------------------------------------------------------------------

TASK_QUERY = """\
Load the dataset at 'data/dirty_data.csv'.

Step 1 – Analyse the dataset: identify all columns that contain missing values
         and note whether each column is categorical or numerical.

Step 2 – Impute ALL missing values: for each column with missing data, call
         the appropriate imputation tool. Do not skip any column.

Step 3 – Once every column is clean, calculate the Gower distance matrix for
         the fully cleaned dataset and report the matrix dimensions and
         distance statistics.

Provide a final summary of every action taken and confirm that the cleaned
dataset and Gower matrix have been saved to disk.
"""


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  AUTONOMOUS DATA CLEANING & FEATURE ENGINEERING AGENT")
    print("=" * 70 + "\n")

    executor = build_agent_executor(verbose=True)

    print("📋  TASK SUBMITTED TO AGENT:")
    print("-" * 70)
    print(TASK_QUERY)
    print("-" * 70 + "\n")

    result = executor.invoke({"input": TASK_QUERY})

    print("\n" + "=" * 70)
    print("✅  AGENT FINAL ANSWER:")
    print("=" * 70)
    print(result["output"])
    print("=" * 70 + "\n")

    # ── Optional: print a tidy table of intermediate steps ─────────────────
    steps = result.get("intermediate_steps", [])
    if steps:
        print(f"\n📊  PIPELINE SUMMARY  ({len(steps)} steps executed)")
        print("-" * 70)
        for i, (action, observation) in enumerate(steps, start=1):
            # observation can be long; truncate for the summary table
            obs_preview = str(observation)[:120].replace("\n", " ")
            print(f"  Step {i:02d} | Tool: {action.tool:<30} | Result: {obs_preview}…")
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
