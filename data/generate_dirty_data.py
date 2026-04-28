"""
data/generate_dirty_data.py

Run this script ONCE to create the demo dirty_data.csv used by the agent.

    python data/generate_dirty_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 200

rng = np.random.default_rng(SEED)

# ── Base clean data ──────────────────────────────────────────────────────────
ages       = rng.integers(18, 75, size=N).astype(float)
salaries   = rng.normal(55_000, 15_000, size=N).round(2)
experience = rng.integers(0, 40, size=N).astype(float)
genders    = rng.choice(["Male", "Female", "Non-binary"], size=N)
education  = rng.choice(["High School", "Bachelor", "Master", "PhD"], size=N)
department = rng.choice(["Engineering", "Marketing", "Sales", "HR", "Finance"], size=N)
satisfaction = rng.choice(["Low", "Medium", "High"], size=N)

df = pd.DataFrame({
    "Age":          ages,
    "Salary":       salaries,
    "YearsExp":     experience,
    "Gender":       genders,
    "Education":    education,
    "Department":   department,
    "Satisfaction": satisfaction,
})

# ── Inject missing values ────────────────────────────────────────────────────
def nullify(series: pd.Series, frac: float) -> pd.Series:
    idx = rng.choice(series.index, size=int(len(series) * frac), replace=False)
    series = series.copy()
    series.loc[idx] = np.nan
    return series

df["Age"]          = nullify(df["Age"],          frac=0.08)   # ~8%  missing
df["Salary"]       = nullify(df["Salary"],        frac=0.25)   # ~25% missing → KNN
df["YearsExp"]     = nullify(df["YearsExp"],      frac=0.12)   # ~12% missing
df["Gender"]       = nullify(df["Gender"],        frac=0.10)   # categorical
df["Education"]    = nullify(df["Education"],     frac=0.07)   # categorical
df["Satisfaction"] = nullify(df["Satisfaction"],  frac=0.15)   # categorical

output_path = Path(__file__).parent / "dirty_data.csv"
df.to_csv(output_path, index=False)
print(f"✔ dirty_data.csv written → {output_path}  ({N} rows, {len(df.columns)} columns)")
print(df.isnull().sum())
