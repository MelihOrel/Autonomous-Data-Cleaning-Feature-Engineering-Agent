"""
agents/react_agent.py

Initialises the LangChain ReAct agent that autonomously orchestrates the
data-cleaning and feature-engineering workflow.

Architecture
────────────
  LLM (GPT-4o-mini, temperature=0)
      │
      ▼
  ReAct Prompt  ──→  create_react_agent()
      │
      ▼
  AgentExecutor  ←── [explore_data, impute_missing_values,
                        calculate_gower_distance]

The ReAct loop (Thought → Action → Observation) repeats until the agent
decides it has completed the task and emits a Final Answer.
"""

from __future__ import annotations

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from tools.data_tools import (
    calculate_gower_distance,
    explore_data,
    impute_missing_values,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PERSONA = """\
You are an Autonomous Senior Data Scientist agent. Your sole responsibility is
to clean dirty datasets and engineer features WITHOUT any human intervention.

Operational rules you must follow at all times:
1. ALWAYS start by calling `explore_data` to understand the dataset structure
   and identify every column that has missing values.
2. For EACH column with missing values, call `impute_missing_values` with the
   correct column name. Never skip a column.
3. After ALL missing values have been imputed, call `calculate_gower_distance`
   to produce the distance matrix for the cleaned dataset.
4. Think step-by-step. After every tool observation, reflect on what still
   needs to be done before proceeding.
5. Be precise: use the EXACT column names returned by `explore_data`.
6. Do NOT hallucinate column names or invent data.
7. Provide a concise Final Answer summarising every action taken and the
   outcome once the full pipeline is complete.
"""

# ReAct template — follows the strict format that create_react_agent expects.
# The {tools} and {tool_names} placeholders are populated automatically by
# LangChain; {agent_scratchpad} holds the running Thought/Action/Observation log.
_REACT_TEMPLATE = """\
{system_persona}

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation cycle can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_agent_executor(verbose: bool = True) -> AgentExecutor:
    """
    Construct and return a fully configured AgentExecutor.

    Parameters
    ----------
    verbose : bool
        When True (default) the agent streams every Thought / Action /
        Observation step to stdout so you can watch the ReAct loop live.

    Returns
    -------
    AgentExecutor
        Ready to invoke with a plain-English task description.
    """
    # ── 1. LLM ────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,        # deterministic reasoning
        max_tokens=4096,
    )

    # ── 2. Tool list ──────────────────────────────────────────────────────
    tools = [explore_data, impute_missing_values, calculate_gower_distance]

    # ── 3. Prompt ─────────────────────────────────────────────────────────
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"system_persona": _SYSTEM_PERSONA},
        template=_REACT_TEMPLATE,
    )

    # ── 4. ReAct agent ────────────────────────────────────────────────────
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # ── 5. Executor ───────────────────────────────────────────────────────
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,   # gracefully recover from malformed LLM output
        max_iterations=20,            # safety ceiling for runaway loops
        return_intermediate_steps=True,
    )

    return executor
