"""
agent_engine.py
---------------
DataVibe — Production-Grade Multi-Agent Churn Analytics System
==============================================================

Three stateful agent nodes, each with a single, well-scoped responsibility:

  ┌─────────────────────┐
  │  SchemaRouterAgent  │  ── Reads CSV header + samples → produces SchemaRouteMap
  └────────┬────────────┘
           │ SchemaRouteMap (dict)
  ┌────────▼──────────────────┐
  │  ExecutionEngineerAgent   │  ── Builds & self-corrects Python analysis code
  │  (sandboxed eval loop)    │     via the tools/code_runner subprocess sandbox
  └────────┬──────────────────┘
           │ ExecutionResult (dict)
  ┌────────▼────────────────────┐
  │  ExecutiveCriticAgent       │  ── Synthesises findings into a strategic report
  │  (bounded session memory)   │     and drives the conversational follow-up chat
  └─────────────────────────────┘

All three agents share a single Google GenAI client and a common
`_generate_with_retry` helper that implements exponential back-off,
model fallback, and free-tier quota detection.
"""

import collections
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from groq import APIError

from tools.code_runner import execute_analysis_code

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PRIMARY_MODEL = "llama-3.3-70b-versatile"
_FALLBACK_MODELS = [PRIMARY_MODEL, "llama-3.1-8b-instant", PRIMARY_MODEL]


# ===========================================================================
# Shared retry helper
# ===========================================================================

def _generate_with_retry(
    contents,
    system_instruction: str | None = None,
    max_retries: int = 6,
    initial_delay: int = 4,
) -> str:
    """
    Robust LLM call wrapper.
    • Retries on transient 429 / 503 errors with exponential back-off.
    • Cycles through _FALLBACK_MODELS after every 2 consecutive failures.
    """
    delay = initial_delay
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": contents})

    for attempt in range(max_retries):
        current_model = _FALLBACK_MODELS[min(attempt // 2, len(_FALLBACK_MODELS) - 1)]
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content

        except APIError as exc:
            error_str = str(exc)
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status_code is None:
                m = re.search(r"\b(\d{3})\b", error_str)
                status_code = int(m.group(1)) if m else None

            # Hard-stop on non-transient errors
            if status_code is not None and status_code not in (429, 503):
                raise RuntimeError(f"Non-retryable API error ({status_code}): {exc}") from exc

            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Groq API severely overloaded — retries exhausted. "
                    f"Last model tried: {current_model}. Error: {exc}"
                ) from exc

            # Honour the API's suggested retryDelay if present
            m = re.search(r"'retryDelay':\s*'(\d+)s'", error_str)
            sleep_time = int(m.group(1)) + 2 if m else delay

            if (attempt + 1) % 2 == 0:
                delay = initial_delay        # reset when switching model
            else:
                delay = min(delay * 2, 60)  # cap back-off at 60 s

            time.sleep(sleep_time)

        except Exception as exc:
            raise


# ===========================================================================
# Agent 1: SchemaRouterAgent
# ===========================================================================

class SchemaRouterAgent:
    """
    Inspects an arbitrary CSV file, classifies every column into one of five
    semantic roles, and produces a structured SchemaRouteMap that downstream
    agents consume.

    Column roles
    ------------
    target              : The binary churn/exit label to predict against.
    categorical_segment : Low-cardinality dimensions for segment cross-tabs.
    behavioral_indicator: Ordinal/count columns capturing engagement patterns.
    financial_indicator : Continuous monetary or score columns.
    identifier_ignore   : Row IDs, surrogate keys, or PII — must be excluded.
    """

    _SYSTEM = (
        "You are the Schema Router Agent for the DataVibe churn analytics platform. "
        "Your sole responsibility is to inspect a dataset schema and produce a "
        "structured JSON-like column classification map. "
        "Be precise. Do not write Python code. "
        "Output only the classification map followed by a brief narrative summary."
    )

    _PROMPT_TEMPLATE = """\
You are inspecting a customer churn dataset.

Detected columns: {columns}

Sample rows (first 5):
{sample}

Classify EVERY column into exactly one of these roles:
  - target              (binary churn label)
  - categorical_segment (low-cardinality grouping dimension)
  - behavioral_indicator (ordinal/count engagement metric)
  - financial_indicator  (continuous monetary or credit score)
  - identifier_ignore    (row ID, surrogate key, or PII — must be excluded from analysis)

Output format (strictly):
COLUMN_MAP:
<ColumnName>: <role>
<ColumnName>: <role>
...

NARRATIVE_SUMMARY:
<A 3-5 sentence description of what the dataset represents, which segments carry
the highest analytical weight for churn modelling, and any data quality notes.>
"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def route(self) -> dict:
        """
        Read the CSV, classify columns, and return a SchemaRouteMap dict.

        Returns
        -------
        dict with keys:
            column_map   : dict[str, str]  — column → role
            narrative    : str             — agent's summary
            raw_response : str             — full LLM output (for debug panel)
            columns_by_role : dict[str, list[str]]
        """
        sample_df = pd.read_csv(self.csv_path, nrows=5)
        columns = list(sample_df.columns)
        sample_str = sample_df.to_string(index=False)

        prompt = self._PROMPT_TEMPLATE.format(
            columns=columns,
            sample=sample_str,
        )

        raw = _generate_with_retry(contents=prompt, system_instruction=self._SYSTEM)

        column_map, narrative = self._parse_response(raw, columns)

        # Build a convenience reverse-index: role → [col, col, …]
        columns_by_role: dict[str, list] = collections.defaultdict(list)
        for col, role in column_map.items():
            columns_by_role[role].append(col)

        return {
            "column_map": column_map,
            "narrative": narrative,
            "raw_response": raw,
            "columns_by_role": dict(columns_by_role),
        }

    @staticmethod
    def _parse_response(raw: str, columns: list) -> tuple[dict, str]:
        """Parse the structured LLM output into a column_map dict and a narrative string."""
        column_map: dict[str, str] = {}
        narrative = ""

        # Extract the COLUMN_MAP block
        map_match = re.search(r"COLUMN_MAP:\s*(.*?)(?:NARRATIVE_SUMMARY:|$)", raw, re.DOTALL)
        if map_match:
            for line in map_match.group(1).strip().splitlines():
                line = line.strip()
                if ":" in line:
                    parts = line.split(":", 1)
                    col = parts[0].strip()
                    role = parts[1].strip().lower()
                    if col in columns:
                        column_map[col] = role

        # Extract the NARRATIVE_SUMMARY block
        narr_match = re.search(r"NARRATIVE_SUMMARY:\s*(.*)", raw, re.DOTALL)
        if narr_match:
            narrative = narr_match.group(1).strip()

        # Fallback: if LLM parsing produced nothing, infer roles from column names and dtypes
        # using heuristics that work for ANY churn-style dataset.
        if not column_map:
            column_map = SchemaRouterAgent._heuristic_classify(columns, raw)

        return column_map, narrative

    @staticmethod
    def _heuristic_classify(columns: list, sample_text: str = "") -> dict[str, str]:
        """
        Dataset-agnostic heuristic fallback classifier.
        Works on any CSV by inspecting column names for semantic signals.
        """
        result: dict[str, str] = {}

        # Keyword banks — order matters (earlier = higher priority)
        _ID_KEYWORDS = (
            "id", "key", "number", "num", "row", "index", "uuid", "code",
            "identifier", "ref", "no",
        )
        _PII_KEYWORDS = (
            "name", "surname", "firstname", "lastname", "email",
            "phone", "address", "postcode", "zip",
        )
        _TARGET_KEYWORDS = (
            "exited", "churn", "churned", "left", "attrition",
            "cancelled", "converted", "defaulted", "lapsed",
        )
        _CAT_KEYWORDS = (
            "country", "geography", "region", "state", "city",
            "gender", "sex", "segment", "category", "tier",
            "type", "group", "class", "department", "channel",
        )
        _BEHAV_KEYWORDS = (
            "tenure", "products", "active", "visits", "logins",
            "complaints", "calls", "sessions", "transactions",
            "frequency", "recency", "card", "member",
        )
        _FINANCE_KEYWORDS = (
            "balance", "salary", "income", "credit", "score",
            "revenue", "amount", "payment", "debt", "spend",
            "value", "worth", "price", "fee", "charge",
        )

        for col in columns:
            key = col.lower().replace(" ", "").replace("_", "").replace("-", "")

            if any(kw in key for kw in _TARGET_KEYWORDS):
                result[col] = "target"
            elif any(key.endswith(kw) or key.startswith(kw) for kw in _ID_KEYWORDS):
                result[col] = "identifier_ignore"
            elif any(kw in key for kw in _PII_KEYWORDS):
                result[col] = "identifier_ignore"
            elif any(kw in key for kw in _CAT_KEYWORDS):
                result[col] = "categorical_segment"
            elif any(kw in key for kw in _BEHAV_KEYWORDS):
                result[col] = "behavioral_indicator"
            elif any(kw in key for kw in _FINANCE_KEYWORDS):
                result[col] = "financial_indicator"
            else:
                # Last resort: default to behavioral_indicator (safe for numeric ordinals)
                result[col] = "behavioral_indicator"

        # Guarantee exactly one target — if none was found, pick the last binary-looking column
        if "target" not in result.values():
            # Try to find a binary column by name heuristics
            candidates = [c for c in columns if result.get(c) == "behavioral_indicator"]
            if candidates:
                result[candidates[-1]] = "target"

        return result


# ===========================================================================
# Agent 2: ExecutionEngineerAgent
# ===========================================================================

class ExecutionEngineerAgent:
    """
    Receives a SchemaRouteMap and autonomously writes, executes, and self-corrects
    Python data analysis code inside a sandboxed subprocess evaluation loop.

    Analysis is fully adaptive to whatever columns the SchemaRouterAgent identified:
      • Global churn rate on the detected target column.
      • Cross-tab churn rates across all detected categorical segment columns.
      • Cohort comparison (mean values) across financial & behavioral columns.
      • A bar/count chart for the most prominent behavioral indicator.
      • A distribution chart for the most prominent financial indicator.
      • A correlation heatmap of all numeric non-identifier columns → churn_heatmap.png

    Works with ANY churn-style CSV — no column names are assumed.
    """

    _SYSTEM = """\
You are the Execution Engineer Agent inside the DataVibe churn analytics system.
Your ONLY job is to write a single, clean, self-contained Python analysis script.
Rules:
  - Wrap the entire script inside a ```python ... ``` block.
  - Do NOT call plt.show(). Save all figures with plt.savefig().
  - Print every computed metric with a clear descriptive header to stdout.
  - Assume pandas, numpy, matplotlib, seaborn, and the dataframe `df` are already loaded.
  - Do not re-import pandas/numpy/matplotlib or re-read the CSV — they are pre-injected.
"""

    _INITIAL_PROMPT_TEMPLATE = """\
You have been given a customer churn dataset with the following column roles determined by the Schema Router Agent:

TARGET COLUMN (binary churn label): {target_col}
CATEGORICAL SEGMENT COLUMNS: {cat_cols}
BEHAVIORAL INDICATOR COLUMNS: {behav_cols}
FINANCIAL INDICATOR COLUMNS: {finance_cols}
IGNORED COLUMNS (exclude from all analysis): {ignore_cols}

Dataset narrative:
{narrative}

Write a complete Python analysis script that performs ALL of the following steps.
Adapt each step EXACTLY to the column names listed above — do NOT assume any column names
beyond what is provided.

1. GLOBAL CHURN RATE
   - Compute df['{target_col}'].mean() and print:
     "=== Global Baseline Churn Rate ===" followed by the rate as a percentage.

2. SEGMENT CROSS-TABS
   - For EACH column in the categorical segment list ({cat_cols}):
     Compute churn rate per unique value using groupby(col)['{target_col}'].mean().
     Print "=== Churn Rate by <col> ===" followed by the result, sorted descending.
   - If there are 2+ categorical columns, also compute a joint cross-tab of the first two:
     pd.crosstab(df[col1], df[col2], values=df['{target_col}'], aggfunc='mean')
     Print "=== Cross-Tab Churn: <col1> × <col2> ==="

3. COHORT COMPARISON
   - For churned ({target_col}==1) vs. retained ({target_col}==0), compute the mean of
     ALL financial and behavioral columns: {finance_cols} + {behav_cols}
   - Print "=== Mean Metrics: Churned vs. Retained ===" followed by a comparison table.

4. BEHAVIORAL BAR CHART
   - Pick the FIRST behavioral indicator column: {first_behav_col}
   - Compute churn rate per unique value of that column.
   - Create a seaborn barplot, title "Churn Rate by {first_behav_col}".
   - Save to 'churn_by_{first_behav_col_safe}.png'.
   - Call plt.close() after saving.

5. FINANCIAL DISTRIBUTION CHART
   - Pick the FIRST financial indicator column: {first_finance_col}
   - Plot overlapping histograms of {first_finance_col} for churned vs. retained (alpha=0.6).
   - Set title "{first_finance_col} Distribution: Churned vs. Retained". Add legend.
   - Save to '{first_finance_col_safe}_distribution.png'.
   - Call plt.close() after saving.

6. CORRELATION HEATMAP
   - Select all strictly numeric columns (e.g. df.select_dtypes(include=['number']).columns)
   - Then remove any ignored ones: {ignore_cols}
   - Compute the correlation matrix and plot a seaborn heatmap (annot=True, cmap='coolwarm', fmt='.2f').
   - Set title "Feature Correlation Heatmap".
   - Save to 'churn_heatmap.png'.
   - Call plt.close() after saving.

Style rules:
  - sns.set_style('darkgrid') at the top of the script.
  - plt.tight_layout() before every plt.savefig() call.
  - figsize=(10, 6) for bar/distribution charts; (12, 9) for the heatmap.
  - Do NOT call plt.show().
"""

    _RETRY_PROMPT_TEMPLATE = """\
Your previous script failed during sandboxed execution.

=== STDERR (exact error) ===
{stderr}

=== STDOUT captured before crash ===
{stdout}

Diagnose the error from the stderr above. Rewrite the COMPLETE script from scratch, fixing the issue.
Output the full corrected script inside a ```python ... ``` block.
"""

    def __init__(self, csv_path: str, max_retries: int = 3):
        self.csv_path = csv_path
        self.max_retries = max_retries

    def run(self, schema_route_map: dict) -> dict:
        """
        Execute the self-correcting engineer loop.

        Returns
        -------
        dict with keys:
            success        : bool
            code_used      : str
            terminal_output: str
            artifacts      : list[str]  — image file paths written
            error          : str | None
            attempts       : int
        """
        cbr = schema_route_map.get("columns_by_role", {})

        # Resolve columns from the schema map — no hardcoded names anywhere
        target_cols  = cbr.get("target", [])
        cat_cols     = cbr.get("categorical_segment", [])
        behav_cols   = cbr.get("behavioral_indicator", [])
        finance_cols = cbr.get("financial_indicator", [])
        ignore_cols  = cbr.get("identifier_ignore", [])

        # Pick primary target; if none detected, fall back to the last column
        target_col = target_cols[0] if target_cols else list(
            schema_route_map.get("column_map", {}).keys()
        )[-1]

        # Pick primary behavioral + financial columns for single-column chart steps
        first_behav_col   = behav_cols[0]   if behav_cols   else (cat_cols[0] if cat_cols else target_col)
        first_finance_col = finance_cols[0] if finance_cols else (behav_cols[0] if behav_cols else target_col)

        # Safe filenames: strip non-alphanumeric characters
        def _safe(col: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]", "_", col).lower()

        base_prompt = self._INITIAL_PROMPT_TEMPLATE.format(
            target_col=target_col,
            cat_cols=", ".join(cat_cols) if cat_cols else "(none detected)",
            behav_cols=", ".join(behav_cols) if behav_cols else "(none detected)",
            finance_cols=", ".join(finance_cols) if finance_cols else "(none detected)",
            ignore_cols=", ".join(ignore_cols) if ignore_cols else "(none)",
            narrative=schema_route_map.get("narrative", ""),
            first_behav_col=first_behav_col,
            first_behav_col_safe=_safe(first_behav_col),
            first_finance_col=first_finance_col,
            first_finance_col_safe=_safe(first_finance_col),
        )
        current_prompt = base_prompt

        for attempt in range(1, self.max_retries + 1):
            raw_text = _generate_with_retry(
                contents=current_prompt,
                system_instruction=self._SYSTEM,
            )

            code_match = re.search(r"```python\s*(.*?)\s*```", raw_text, re.DOTALL)
            if not code_match:
                current_prompt = (
                    "Your response did not contain a valid ```python ... ``` block. "
                    "Please output ONLY the complete script inside that markdown block."
                )
                continue

            extracted_code = code_match.group(1)
            execution = execute_analysis_code(extracted_code, self.csv_path)

            if execution["success"]:
                return {
                    "success": True,
                    "code_used": extracted_code,
                    "terminal_output": execution["output"],
                    "artifacts": execution["artifacts"],
                    "error": None,
                    "attempts": attempt,
                }

            # Build targeted debug prompt for the next attempt
            current_prompt = base_prompt + "\n\n" + self._RETRY_PROMPT_TEMPLATE.format(
                stderr=execution["error"][:3000],   # cap to avoid token bloat
                stdout=execution["output"][:1000],
            )
            time.sleep(5)  # brief pause before retry

        last_error = execution["error"] if "execution" in locals() and execution else "Unknown error"
        return {
            "success": False,
            "code_used": extracted_code if "extracted_code" in dir() else "",
            "terminal_output": "",
            "artifacts": [],
            "error": f"Code execution failed after {self.max_retries} attempts. Last error:\n{last_error}",
            "attempts": self.max_retries,
        }


# ===========================================================================
# Agent 3: ExecutiveCriticAgent
# ===========================================================================

class ExecutiveCriticAgent:
    """
    Synthesises raw execution outputs into strategic business intelligence reports
    and drives a persistent short-term conversational session with stakeholders.

    Memory model
    ------------
    A bounded `collections.deque` stores the last `memory_window` turns.
    Every LLM call serialises the current deque into a [USER]/[AGENT] block
    injected as `Session Memory` in the prompt — ensuring the critic always
    has contextual grounding without unbounded token growth.
    """

    _SYSTEM = """\
You are the Executive Critic Agent for the DataVibe churn analytics platform.
You translate raw statistical outputs into precise, actionable business intelligence.
Your outputs must be:
  - Grounded strictly in the data shown — never hallucinate numbers.
  - Structured with markdown headings and bullet points.
  - Concise enough for a busy C-suite executive to scan in under 2 minutes.
"""

    _REPORT_PROMPT_TEMPLATE = """\
You have access to the following data analysis outputs produced by the Execution Engineer:

=== Schema Summary ===
{narrative}

=== Raw Execution Terminal Output ===
{terminal_output}

Produce a structured Executive Retention Intelligence Report with EXACTLY these sections:

## 🔍 Key Findings
Bullet points identifying which customer segments exhibit the highest churn rates.
Reference specific numbers from the terminal output.

## ⚠️ Risk Drivers
Bullet points identifying which behavioral or financial attributes correlate most strongly
with retention failures. Be specific.

## 🎯 Retention Action Plan
3-5 concrete, operational recommendations the retention team can implement immediately.
Prioritise by expected impact. Each recommendation must be actionable, not generic.

Keep the entire report under 500 words.
"""

    _CHAT_PROMPT_TEMPLATE = """\
You are the Executive Critic Agent in an ongoing strategy session with a business stakeholder.

The data analysis pipeline produced the following terminal output (ground truth):
{terminal_output}

=== Session Memory (last {window} turns) ===
{session_memory}

=== New Question ===
{question}

Provide a specific, accurate answer grounded ONLY in the statistical outputs shown above.
Do not guess. If the data does not support an answer, say so clearly.
"""

    def __init__(self, memory_window: int = 6):
        self.memory_window = memory_window
        self._memory: collections.deque = collections.deque(maxlen=memory_window)

    def generate_report(self, schema_route_map: dict, execution_result: dict) -> str:
        """
        Synthesise the pipeline outputs into a structured executive report.
        """
        if not execution_result.get("success"):
            return (
                f"## ❌ Analysis Pipeline Failed\n\n"
                f"**Reason:** {execution_result.get('error', 'Unknown error')}\n\n"
                "Please re-run the pipeline. Check the Agent Console for error details."
            )

        prompt = self._REPORT_PROMPT_TEMPLATE.format(
            narrative=schema_route_map.get("narrative", ""),
            terminal_output=execution_result.get("terminal_output", ""),
        )
        return _generate_with_retry(contents=prompt, system_instruction=self._SYSTEM)

    def chat(self, user_question: str, execution_result: dict) -> str:
        """
        Answer a follow-up question, injecting bounded session memory as context.
        Appends the turn to the memory deque after generating a response.
        """
        session_memory_str = ""
        for turn in self._memory:
            session_memory_str += f"[USER]: {turn['user']}\n[AGENT]: {turn['agent']}\n\n"

        if not session_memory_str:
            session_memory_str = "(No prior turns in this session.)"

        prompt = self._CHAT_PROMPT_TEMPLATE.format(
            terminal_output=execution_result.get("terminal_output", ""),
            window=self.memory_window,
            session_memory=session_memory_str,
            question=user_question,
        )

        response = _generate_with_retry(contents=prompt, system_instruction=self._SYSTEM)

        # Persist this turn into short-term memory
        self._memory.append({"user": user_question, "agent": response})
        return response

    def get_memory_snapshot(self) -> list[dict]:
        """Return a read-only snapshot of the current session memory."""
        return list(self._memory)

    def clear_memory(self):
        """Reset the session memory deque."""
        self._memory.clear()


# ===========================================================================
# Orchestrator
# ===========================================================================

class ChurnAgentOrchestrator:
    """
    Stateful coordinator that wires the three agent nodes into a sequential pipeline
    and exposes a cached-resume contract so the UI can recover from mid-run crashes.

    Pipeline contract
    -----------------
    `run_pipeline(cached_state)` accepts a dict that may already contain partial
    results. It skips any stage whose key is already present in `cached_state`,
    enabling crash recovery without re-running expensive or quota-burning stages.

    Stage keys: "schema" | "execution" | "report"
    """

    def __init__(self, csv_path: str, memory_window: int = 6):
        self.csv_path = csv_path
        self._schema_agent = SchemaRouterAgent(csv_path)
        self._engineer_agent = ExecutionEngineerAgent(csv_path)
        self._critic_agent = ExecutiveCriticAgent(memory_window=memory_window)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, cached_state: dict | None = None) -> dict:
        """
        Execute (or resume) the three-stage analysis pipeline.

        Parameters
        ----------
        cached_state : dict, optional
            Partial results from a previous interrupted run.

        Returns
        -------
        dict with keys: "schema", "execution", "report"
        """
        if cached_state is None:
            cached_state = {}

        # Stage 1 — Schema Router
        if "schema" not in cached_state:
            schema_route_map = self._schema_agent.route()
            cached_state["schema"] = schema_route_map
            time.sleep(8)  # throttle to respect free-tier rate limits

        # Stage 2 — Execution Engineer
        if "execution" not in cached_state or not cached_state["execution"].get("success"):
            execution_result = self._engineer_agent.run(cached_state["schema"])
            cached_state["execution"] = execution_result
            time.sleep(8)

        # Stage 3 — Executive Critic (initial report)
        if "report" not in cached_state or not cached_state.get("execution", {}).get("success"):
            report = self._critic_agent.generate_report(
                cached_state["schema"],
                cached_state["execution"],
            )
            cached_state["report"] = report

        return cached_state

    # ------------------------------------------------------------------
    # Conversational follow-up
    # ------------------------------------------------------------------

    def conversational_follow_up(
        self,
        user_question: str,
        execution_result: dict,
    ) -> str:
        """
        Delegate a stakeholder question to the ExecutiveCriticAgent's chat method,
        which automatically injects bounded session memory into the prompt.
        """
        return self._critic_agent.chat(
            user_question=user_question,
            execution_result=execution_result,
        )

    def get_memory_snapshot(self) -> list[dict]:
        """Expose the critic's current session memory for display in the UI."""
        return self._critic_agent.get_memory_snapshot()

    def clear_chat_memory(self):
        """Reset the critic's conversational memory."""
        self._critic_agent.clear_memory()