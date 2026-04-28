"""
tools/data_tools.py

Custom LangChain tools for the Autonomous Data Cleaning & Feature Engineering Agent.
Each tool is decorated with @tool and includes a comprehensive docstring so the
ReAct agent can reason about when and how to invoke it.
"""

from __future__ import annotations

import io
import os
from typing import Optional

import gower
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.impute import KNNImputer


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file and raise a clear error if the path is invalid."""
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Dataset not found at '{abs_path}'. "
            "Make sure the file path is correct relative to the project root."
        )
    return pd.read_csv(abs_path)


def _save_csv(df: pd.DataFrame, file_path: str) -> None:
    """Persist a DataFrame back to disk at the given path."""
    abs_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)


# ---------------------------------------------------------------------------
# Tool 1 – explore_data
# ---------------------------------------------------------------------------

@tool
def explore_data(file_path: str) -> str:
    """
    Explore and profile a CSV dataset to understand its structure and data quality.

    Use this tool FIRST, before any cleaning or transformation step. It gives
    you a complete picture of the dataset so you can plan the right imputation
    strategy for each column.

    What this tool returns:
    - The shape of the DataFrame (rows × columns).
    - Column names, non-null counts, and inferred dtypes (via df.info()).
    - The number of missing values per column (via df.isnull().sum()).
    - Basic descriptive statistics for numerical columns (via df.describe()).

    Args:
        file_path: Relative or absolute path to the CSV file.
                   Example: "data/dirty_data.csv"

    Returns:
        A formatted multi-section string report that can be read and reasoned
        about directly by the agent.
    """
    try:
        df = _load_csv(file_path)

        # ── Section 1: shape ─────────────────────────────────────────────
        shape_info = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"

        # ── Section 2: df.info() captured as string ──────────────────────
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        # ── Section 3: missing-value counts ──────────────────────────────
        null_counts = df.isnull().sum()
        total_missing = null_counts.sum()
        missing_lines = "\n".join(
            f"  {col}: {cnt} missing ({cnt / len(df) * 100:.1f}%)"
            for col, cnt in null_counts.items()
            if cnt > 0
        ) or "  ✔ No missing values detected."

        # ── Section 4: descriptive stats ─────────────────────────────────
        desc_str = df.describe(include="all").to_string()

        report = (
            "=" * 60 + "\n"
            "DATA EXPLORATION REPORT\n"
            "=" * 60 + "\n\n"
            f"[1] SHAPE\n{shape_info}\n"
            f"[2] COLUMN INFO\n{info_str}\n"
            f"[3] MISSING VALUES (total={total_missing})\n{missing_lines}\n\n"
            f"[4] DESCRIPTIVE STATISTICS\n{desc_str}\n"
            "=" * 60
        )
        return report

    except Exception as exc:
        return f"ERROR in explore_data: {exc}"


# ---------------------------------------------------------------------------
# Tool 2 – impute_missing_values
# ---------------------------------------------------------------------------

@tool
def impute_missing_values(file_path: str, column_name: str) -> str:
    """
    Intelligently impute (fill) missing values in a single column of a CSV dataset.

    Call this tool once per column that contains missing values. The tool
    automatically detects the column's data type and selects the most appropriate
    imputation strategy:

    • Categorical / object column  → Mode imputation (most-frequent value).
    • Numerical column with ≤ 20 % missing  → Median imputation (robust to outliers).
    • Numerical column with > 20 % missing  → KNN imputation (k=5) for better accuracy.

    After imputation the updated dataset is saved back to the same file path so
    subsequent tool calls always work with the latest version of the data.

    Args:
        file_path:   Relative or absolute path to the CSV file.
                     Example: "data/dirty_data.csv"
        column_name: Exact name of the column to impute.
                     Example: "Age"

    Returns:
        A string message describing the strategy applied and the number of
        cells that were filled, or an error message if something went wrong.
    """
    try:
        df = _load_csv(file_path)

        if column_name not in df.columns:
            available = ", ".join(df.columns.tolist())
            return (
                f"ERROR: Column '{column_name}' not found in the dataset. "
                f"Available columns: {available}"
            )

        missing_before = df[column_name].isnull().sum()
        if missing_before == 0:
            return f"Column '{column_name}' has no missing values. No action taken."

        col_dtype = df[column_name].dtype
        missing_pct = missing_before / len(df)

        # ── Categorical imputation ─────────────────────────────────────
        if col_dtype == object or pd.api.types.is_categorical_dtype(df[column_name]):
            fill_value = df[column_name].mode()[0]
            df[column_name].fillna(fill_value, inplace=True)
            strategy = f"mode imputation (fill value = '{fill_value}')"

        # ── Numerical imputation ───────────────────────────────────────
        elif pd.api.types.is_numeric_dtype(df[column_name]):

            if missing_pct <= 0.20:
                # Median is robust to skewed distributions
                fill_value = df[column_name].median()
                df[column_name].fillna(fill_value, inplace=True)
                strategy = f"median imputation (fill value = {fill_value:.4f})"

            else:
                # KNN imputation leverages correlations between columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                strategy = "KNN imputation (k=5) applied across all numeric columns"

        else:
            return (
                f"ERROR: Unsupported dtype '{col_dtype}' for column '{column_name}'. "
                "Only object/categorical and numeric columns are supported."
            )

        missing_after = df[column_name].isnull().sum()
        cells_filled = missing_before - missing_after

        _save_csv(df, file_path)

        return (
            f"SUCCESS: Column '{column_name}' imputed using {strategy}. "
            f"{cells_filled} missing value(s) filled. "
            f"Dataset saved back to '{file_path}'."
        )

    except Exception as exc:
        return f"ERROR in impute_missing_values: {exc}"


# ---------------------------------------------------------------------------
# Tool 3 – calculate_gower_distance
# ---------------------------------------------------------------------------

@tool
def calculate_gower_distance(file_path: str) -> str:
    """
    Calculate the Gower distance matrix for a mixed-type dataset (categorical +
    numerical columns) and save it as a CSV artefact.

    Gower distance is the gold-standard similarity metric for datasets that
    contain both numerical and categorical features. Unlike Euclidean distance,
    it handles mixed data types natively by normalising each feature's
    contribution to the [0, 1] range.

    Use this tool AFTER all missing values have been imputed, because the
    Gower library cannot handle NaN values.

    What this tool does:
    1. Loads the (cleaned) dataset from file_path.
    2. Drops any remaining rows that still contain NaN values and warns about them.
    3. Computes the N × N Gower distance matrix (values in [0, 1]).
    4. Saves the matrix to  <same_directory>/gower_distance_matrix.csv.
    5. Returns a summary: matrix shape, min/max/mean distances, and the save path.

    Args:
        file_path: Relative or absolute path to the CLEANED CSV file.
                   Example: "data/dirty_data.csv"

    Returns:
        A string summary of the Gower distance matrix dimensions and statistics,
        or an error message if the computation fails.
    """
    try:
        df = _load_csv(file_path)

        # ── Pre-flight check ──────────────────────────────────────────────
        rows_before = len(df)
        df_clean = df.dropna()
        rows_dropped = rows_before - len(df_clean)
        warning_msg = ""
        if rows_dropped > 0:
            warning_msg = (
                f"WARNING: {rows_dropped} row(s) still contained NaN after "
                "imputation and were dropped before computing Gower distance.\n"
            )

        if df_clean.empty:
            return "ERROR: Dataset is empty after dropping NaN rows. Impute missing values first."

        # ── Compute Gower distance matrix ─────────────────────────────────
        # gower.gower_matrix returns a numpy ndarray of shape (N, N)
        distance_matrix = gower.gower_matrix(df_clean)

        n = distance_matrix.shape[0]
        # Mask the diagonal (self-distance = 0) for meaningful statistics
        off_diag = distance_matrix[~np.eye(n, dtype=bool)]
        dist_min = float(np.min(off_diag))
        dist_max = float(np.max(off_diag))
        dist_mean = float(np.mean(off_diag))

        # ── Save the matrix ───────────────────────────────────────────────
        output_dir = os.path.dirname(os.path.abspath(file_path))
        output_path = os.path.join(output_dir, "gower_distance_matrix.csv")
        matrix_df = pd.DataFrame(
            distance_matrix,
            index=range(n),
            columns=range(n),
        )
        matrix_df.to_csv(output_path, index=True)

        result = (
            f"{warning_msg}"
            "=" * 60 + "\n"
            "GOWER DISTANCE MATRIX — SUMMARY\n"
            "=" * 60 + "\n"
            f"  Matrix shape  : {n} × {n}\n"
            f"  Min distance  : {dist_min:.6f}\n"
            f"  Max distance  : {dist_max:.6f}\n"
            f"  Mean distance : {dist_mean:.6f}\n"
            f"  Saved to      : {output_path}\n"
            "=" * 60
        )
        return result

    except Exception as exc:
        return f"ERROR in calculate_gower_distance: {exc}"
