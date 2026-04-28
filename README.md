# 🤖 Autonomous Data Cleaning & Feature Engineering Agent

> **An AI-powered data science pipeline built on LangChain's ReAct architecture — no human intervention required.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?logo=chainlink)](https://www.langchain.com/)
[![OpenAI GPT-4o-mini](https://img.shields.io/badge/LLM-GPT--4o--mini-blueviolet?logo=openai)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements a **fully autonomous data-cleaning and feature-engineering agent** using the **ReAct (Reasoning and Acting)** pattern from LangChain. The agent is handed a raw, dirty CSV dataset and, without any human guidance, reasons through the following pipeline:

1. **Explore** the dataset to understand its structure and locate missing values.
2. **Impute** each missing column intelligently — choosing between mode, median, or KNN imputation based on column type and missingness rate.
3. **Engineer features** by calculating the **Gower distance matrix** — the industry-standard similarity metric for mixed-type datasets.

Every decision is logged in real time as a `Thought → Action → Observation` chain, making the agent's reasoning fully transparent and auditable.

---

## ✨ Features

| Feature | Details |
|---|---|
| **ReAct Architecture** | Implements the Reasoning + Acting loop via `create_react_agent` + `AgentExecutor`. The LLM plans, acts, observes, and re-plans autonomously. |
| **Smart Imputation** | Automatically selects **mode** (categorical), **median** (numeric, ≤20% missing), or **KNN** (numeric, >20% missing) imputation per column. |
| **Gower Distance** | Computes the full N×N Gower distance matrix for mixed categorical + numerical datasets — output saved as a reusable CSV artefact. |
| **Zero Human-in-the-Loop** | A single natural-language query triggers the complete end-to-end pipeline. |
| **Transparent Reasoning** | Every `Thought`, `Action`, and `Observation` is printed to stdout — fully auditable ReAct trace. |
| **Production-Grade Code** | Full type hints, docstrings, error handling, and a clean package structure ready for CI/CD. |

---

## 🗂 Project Structure

```
autonomous-data-agent/
│
├── agents/
│   ├── __init__.py
│   └── react_agent.py        # ReAct agent + AgentExecutor factory
│
├── tools/
│   ├── __init__.py
│   └── data_tools.py         # Custom LangChain @tool decorators
│
├── data/
│   ├── generate_dirty_data.py  # Script to generate demo dirty_data.csv
│   └── dirty_data.csv          # Demo dataset (generated; 200 rows, 7 cols)
│
├── main.py                   # Entry point — run this
├── requirements.txt
├── .env.example              # Copy → .env and add your OpenAI key
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autonomous-data-agent.git
cd autonomous-data-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

```bash
cp .env.example .env
# Open .env and replace the placeholder with your real key:
# OPENAI_API_KEY=sk-...
```

### 5. Generate the demo dirty dataset

```bash
python data/generate_dirty_data.py
```

---

## 🚀 Usage

```bash
python main.py
```

The agent will process the complete pipeline autonomously. You will see the live ReAct trace in your terminal:

```
======================================================================
  AUTONOMOUS DATA CLEANING & FEATURE ENGINEERING AGENT
======================================================================

📋  TASK SUBMITTED TO AGENT:
----------------------------------------------------------------------
Load the dataset at 'data/dirty_data.csv'. Analyse it to find missing
values. Impute the missing values appropriately. Calculate the Gower
distance matrix for the cleaned dataset.
----------------------------------------------------------------------

> Entering new AgentExecutor chain...

Thought: I need to start by exploring the dataset to understand its
structure and identify all columns with missing values.

Action: explore_data
Action Input: data/dirty_data.csv

Observation:
============================================================
DATA EXPLORATION REPORT
============================================================
[1] SHAPE
Shape: 200 rows × 7 columns

[3] MISSING VALUES (total=154)
  Age: 16 missing (8.0%)
  Salary: 50 missing (25.0%)
  YearsExp: 24 missing (12.0%)
  Gender: 20 missing (10.0%)
  Education: 14 missing (7.0%)
  Satisfaction: 30 missing (15.0%)
============================================================

Thought: I can see 6 columns with missing values. I will now impute
each one. Age is numerical with 8% missing → median. Salary is
numerical with 25% missing → KNN. I'll start with Age.

Action: impute_missing_values
Action Input: {"file_path": "data/dirty_data.csv", "column_name": "Age"}

Observation: SUCCESS: Column 'Age' imputed using median imputation
(fill value = 44.0000). 16 missing value(s) filled.
Dataset saved back to 'data/dirty_data.csv'.

Thought: Age is done. Salary has 25% missing → KNN imputation.

Action: impute_missing_values
Action Input: {"file_path": "data/dirty_data.csv", "column_name": "Salary"}

Observation: SUCCESS: Column 'Salary' imputed using KNN imputation
(k=5) applied across all numeric columns. 50 missing value(s) filled.
Dataset saved back to 'data/dirty_data.csv'.

... (agent continues for all remaining columns) ...

Action: calculate_gower_distance
Action Input: data/dirty_data.csv

Observation:
============================================================
GOWER DISTANCE MATRIX — SUMMARY
============================================================
  Matrix shape  : 200 × 200
  Min distance  : 0.028571
  Max distance  : 0.857143
  Mean distance : 0.456812
  Saved to      : /path/to/data/gower_distance_matrix.csv
============================================================

Thought: I have now completed the full pipeline.
Final Answer: Successfully completed the autonomous data-cleaning
pipeline. Imputed 154 missing values across 6 columns using median,
KNN, and mode strategies. The Gower distance matrix (200×200) has
been computed and saved to data/gower_distance_matrix.csv.

> Finished chain.

======================================================================
✅  AGENT FINAL ANSWER:
======================================================================
Successfully completed the autonomous data-cleaning pipeline...
======================================================================
```

---

## 🧠 Architecture Deep-Dive

```
User Query (natural language)
        │
        ▼
┌───────────────────────────────┐
│   ReAct Prompt Template       │
│   (Thought / Action / Obs.)   │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│   GPT-4o-mini  (temp=0)       │  ← LangChain ChatOpenAI
└───────────┬───────────────────┘
            │ selects tool + input
            ▼
┌──────────────────────────────────────────┐
│            Tool Dispatcher               │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ explore_data│  │impute_missing_   │  │
│  │             │  │values            │  │
│  └─────────────┘  └──────────────────┘  │
│  ┌──────────────────────────────────┐   │
│  │ calculate_gower_distance         │   │
│  └──────────────────────────────────┘   │
└──────────────────────────────────────────┘
            │ observation
            ▼
      (loop until Final Answer)
```

### Imputation decision tree

```
Column has missing values?
        │
        ├─ YES, dtype = object/categorical
        │       └──→ Mode imputation
        │
        └─ YES, dtype = numeric
                ├─ missing ≤ 20%  ──→ Median imputation
                └─ missing > 20%  ──→ KNN imputation (k=5)
```

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `langchain` | ReAct agent framework |
| `langchain-openai` | GPT-4o-mini LLM integration |
| `pandas` | DataFrame I/O and manipulation |
| `scikit-learn` | `KNNImputer` for advanced imputation |
| `gower` | Gower distance matrix for mixed types |
| `python-dotenv` | Secure API key management |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

*Built with ❤️ using LangChain ReAct + GPT-4o-mini*
