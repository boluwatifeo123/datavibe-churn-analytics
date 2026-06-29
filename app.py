"""
app.py
------
DataVibe — Streamlit Frontend
Premium dark-mode UI for the multi-agent churn analytics system.
"""

import os

import pandas as pd
import streamlit as st

from agent_engine import ChurnAgentOrchestrator

# ===========================================================================
# Page configuration
# ===========================================================================
st.set_page_config(
    page_title="DataVibe | Multi-Agent Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Global CSS — dark premium theme
# ===========================================================================
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* ── Root tokens ─────────────────────────────────────────── */
    :root {
        --bg-base:      #0a0b14;
        --bg-surface:   #11121f;
        --bg-card:      #181929;
        --border:       rgba(124, 58, 237, 0.25);
        --accent-violet:#7C3AED;
        --accent-cyan:  #06B6D4;
        --accent-green: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose:  #F43F5E;
        --text-primary: #F1F5F9;
        --text-muted:   #94A3B8;
        --radius:       12px;
        --shadow:       0 4px 24px rgba(0,0,0,0.45);
    }
    /* ── Base & typography ────────────────────────────────────── */
    html, body, .stApp {
        background-color: var(--bg-base) !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
    /* ── Sidebar ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    /* ── Tabs ─────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-surface);
        border-radius: var(--radius);
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        color: var(--text-muted) !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-violet), #5B21B6) !important;
        color: #fff !important;
    }
    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-violet) 0%, #5B21B6 100%);
        color: #fff !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 2px 12px rgba(124, 58, 237, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.55);
    }
    /* ── Metrics ──────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.78rem; }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--accent-cyan) !important;
        font-weight: 700;
        font-size: 1.6rem !important;
    }
    /* ── Expanders ────────────────────────────────────────────── */
    details {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 8px;
    }
    summary { font-weight: 600; padding: 10px 0; }
    /* ── Text area / code ──────────────────────────────────────── */
    .stTextArea textarea, .stCodeBlock {
        background: #0d0e1c !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.82rem !important;
    }
    /* ── Dataframe ────────────────────────────────────────────── */
    .stDataFrame { border-radius: var(--radius); overflow: hidden; }
    /* ── Chat messages ────────────────────────────────────────── */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 8px;
    }
    /* ── Alert / info boxes ───────────────────────────────────── */
    .stAlert { border-radius: var(--radius) !important; }
    /* ── Upload widget ────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 1px dashed var(--accent-violet) !important;
        border-radius: var(--radius) !important;
        padding: 16px;
    }
    /* ── Spinner ──────────────────────────────────────────────── */
    .stSpinner > div { border-color: var(--accent-violet) transparent transparent !important; }
    /* ── Status badge helper classes ──────────────────────────── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .badge-queued  { background: rgba(148,163,184,0.15); color: #94A3B8; }
    .badge-running { background: rgba(6,182,212,0.15);  color: #06B6D4; }
    .badge-done    { background: rgba(16,185,129,0.15); color: #10B981; }
    .badge-failed  { background: rgba(244,63,94,0.15);  color: #F43F5E; }
    /* ── Glass card ───────────────────────────────────────────── */
    .glass-card {
        background: rgba(24, 25, 41, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    /* ── Role pill tags ───────────────────────────────────────── */
    .pill { display:inline-block; padding:2px 9px; border-radius:99px; font-size:0.7rem; font-weight:600; margin:2px; }
    .pill-target    { background:rgba(244,63,94,0.2);   color:#F43F5E; }
    .pill-cat       { background:rgba(124,58,237,0.2);  color:#A78BFA; }
    .pill-behav     { background:rgba(6,182,212,0.2);   color:#06B6D4; }
    .pill-finance   { background:rgba(16,185,129,0.2);  color:#10B981; }
    .pill-ignore    { background:rgba(148,163,184,0.15);color:#64748B; }
    /* ── Memory indicator bar ─────────────────────────────────── */
    .memory-bar {
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# Helpers
# ===========================================================================

ROLE_PILL_CLASS = {
    "target": "pill-target",
    "categorical_segment": "pill-cat",
    "behavioral_indicator": "pill-behav",
    "financial_indicator": "pill-finance",
    "identifier_ignore": "pill-ignore",
}

ROLE_LABEL = {
    "target": "🎯 Target",
    "categorical_segment": "🗂 Segment",
    "behavioral_indicator": "📊 Behavioral",
    "financial_indicator": "💰 Financial",
    "identifier_ignore": "🚫 Ignored",
}


def badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def pill(col: str, role: str) -> str:
    css = ROLE_PILL_CLASS.get(role, "pill-ignore")
    label = ROLE_LABEL.get(role, role)
    return f'<span class="pill {css}" title="{role}">{col}</span>'


def compute_churn_prevalence(df: pd.DataFrame, target_col: str | None = None) -> float | None:
    """Return churn prevalence % using the schema-detected target column, or by name heuristics."""
    # Prefer the schema-detected column
    if target_col and target_col in df.columns:
        return float(df[target_col].mean() * 100)
    # Heuristic fallback for unknown schemas
    candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("exited", "churn", "churned", "left", "attrition", "lapsed"))
    ]
    if candidates:
        return float(df[candidates[0]].mean() * 100)
    return None


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 24px;">
            <div style="font-size:2.4rem;">📡</div>
            <div style="font-size:1.25rem; font-weight:700; color:#F1F5F9;">DataVibe</div>
            <div style="font-size:0.72rem; color:#7C3AED; font-weight:600; letter-spacing:1px;">
                MULTI-AGENT CHURN ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Configuration")
    memory_window = st.slider(
        "Critic Memory Window (turns)",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        help="Number of conversation turns the Executive Critic remembers per session.",
    )

    st.markdown("---")
    st.markdown("### 🤖 Agent Nodes")
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#94A3B8; line-height:1.8;">
        <b style="color:#A78BFA;">① Schema Router</b><br>
        Classifies every CSV column into a semantic role for downstream routing.<br><br>
        <b style="color:#06B6D4;">② Execution Engineer</b><br>
        Writes & self-corrects Python analysis code in a sandboxed subprocess eval loop.<br><br>
        <b style="color:#10B981;">③ Executive Critic</b><br>
        Synthesises findings into strategic reports with bounded session memory.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem; color:#475569;'>Powered by Groq SDK · Llama 3 · Local</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# Header
# ===========================================================================
st.markdown(
    """
    <div style="padding: 8px 0 24px;">
        <h1 style="margin:0;">📡 DataVibe</h1>
        <p style="color:#7C3AED; font-weight:600; margin:4px 0 0; font-size:0.9rem; letter-spacing:0.5px;">
            Stateful Multi-Agent Retention Analytics Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# File upload
# ===========================================================================
uploaded_file = st.file_uploader(
    "Upload your churn dataset (CSV)",
    type=["csv"],
    help="Expected format: Churn_Modelling.csv with columns for Geography, Gender, Exited, etc.",
)

# Auto-load the bundled dataset if no file is uploaded
BUNDLED_CSV = "Churn_Modelling.csv"
if uploaded_file is None and os.path.exists(BUNDLED_CSV):
    csv_path = os.path.abspath(BUNDLED_CSV)
    st.info(f"📂 Using bundled dataset: **{BUNDLED_CSV}**", icon="ℹ️")
elif uploaded_file is not None:
    csv_path = os.path.abspath(BUNDLED_CSV)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File uploaded and saved.", icon="✅")
else:
    st.warning("Please upload a churn CSV file or place `Churn_Modelling.csv` in the project root.")
    st.stop()

# Initialise or re-initialise the orchestrator when memory_window changes
if (
    "orchestrator" not in st.session_state
    or st.session_state.get("memory_window") != memory_window
):
    st.session_state["orchestrator"] = ChurnAgentOrchestrator(
        csv_path=csv_path, memory_window=memory_window
    )
    st.session_state["memory_window"] = memory_window

orchestrator: ChurnAgentOrchestrator = st.session_state["orchestrator"]

# ===========================================================================
# Tabs
# ===========================================================================
tab1, tab2, tab3 = st.tabs([
    "📋  Data Preview",
    "⚙️  Agent Console",
    "📈  Strategic Insights",
])

# ---------------------------------------------------------------------------
# TAB 1 — Data Preview
# ---------------------------------------------------------------------------
with tab1:
    df_full = pd.read_csv(csv_path)
    total_rows = len(df_full)
    total_cols = len(df_full.columns)
    null_pct = df_full.isnull().mean().mean() * 100
    churn_pct = compute_churn_prevalence(df_full)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{total_rows:,}")
    m2.metric("Columns", total_cols)
    m3.metric("Null Rate", f"{null_pct:.2f}%")
    if churn_pct is not None:
        m4.metric("Churn Prevalence", f"{churn_pct:.1f}%")

    st.markdown("#### 📊 Dataset Sample (First 10 Rows)")
    st.dataframe(df_full.head(10), use_container_width=True, height=280)

    # Show schema route map if available
    if "pipeline_results" in st.session_state:
        col_map = st.session_state["pipeline_results"]["schema"].get("column_map", {})
        if col_map:
            st.markdown("#### 🗂 Schema Route Classification")
            st.markdown(
                "<div class='glass-card'>"
                + " ".join(pill(col, role) for col, role in col_map.items())
                + "</div>",
                unsafe_allow_html=True,
            )

            # Legend
            lc1, lc2, lc3, lc4, lc5 = st.columns(5)
            lc1.markdown("<span class='pill pill-target'>🎯 Target</span>", unsafe_allow_html=True)
            lc2.markdown("<span class='pill pill-cat'>🗂 Segment</span>", unsafe_allow_html=True)
            lc3.markdown("<span class='pill pill-behav'>📊 Behavioral</span>", unsafe_allow_html=True)
            lc4.markdown("<span class='pill pill-finance'>💰 Financial</span>", unsafe_allow_html=True)
            lc5.markdown("<span class='pill pill-ignore'>🚫 Ignored</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2 — Agent Console
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### Multi-Agent Pipeline Execution Console")

    # Determine agent statuses for the status bar
    res = st.session_state.get("pipeline_results", {})
    schema_status = "done" if "schema" in res else "queued"
    exec_status   = "done" if "execution" in res else ("running" if "schema" in res else "queued")
    critic_status = "done" if "report" in res else "queued"

    # Status bar
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"<div class='glass-card' style='text-align:center;'>"
            f"<div style='font-size:1.4rem;'>🗺️</div>"
            f"<div style='font-weight:600;font-size:0.9rem;margin:6px 0 8px;'>Schema Router</div>"
            f"{badge('● ' + schema_status.upper(), schema_status)}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div class='glass-card' style='text-align:center;'>"
            f"<div style='font-size:1.4rem;'>⚙️</div>"
            f"<div style='font-weight:600;font-size:0.9rem;margin:6px 0 8px;'>Execution Engineer</div>"
            f"{badge('● ' + exec_status.upper(), exec_status)}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div class='glass-card' style='text-align:center;'>"
            f"<div style='font-size:1.4rem;'>🧠</div>"
            f"<div style='font-weight:600;font-size:0.9rem;margin:6px 0 8px;'>Executive Critic</div>"
            f"{badge('● ' + critic_status.upper(), critic_status)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_run, col_reset = st.columns([2, 1])
    with col_run:
        run_clicked = st.button("🚀 Trigger Agentic Pipeline Run", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset & Clear Cache", use_container_width=True):
            for key in ["pipeline_results", "chat_history"]:
                st.session_state.pop(key, None)
            orchestrator.clear_chat_memory()
            st.rerun()

    if run_clicked:
        with st.spinner(
            "Orchestrating agent nodes… (first run takes ~30–60s due to API rate-limit throttling)"
        ):
            try:
                cached = st.session_state.get("pipeline_results", {})
                results = orchestrator.run_pipeline(cached_state=cached)
                st.session_state["pipeline_results"] = results
                st.success("✅ Pipeline completed successfully across all three agent nodes.")
                st.rerun()
            except RuntimeError as exc:
                st.error(f"⚠️ Pipeline halted: {exc}")
                st.info(
                    "Your partial progress has been saved. Wait 1–2 minutes for the Groq API "
                    "to recover, then click **Trigger** again to resume from the last completed stage."
                )

    # Display agent outputs
    if "pipeline_results" in st.session_state:
        res = st.session_state["pipeline_results"]

        # --- Schema Router output
        with st.expander("① Schema Router Agent — Column Classification", expanded=False):
            schema = res.get("schema", {})
            if schema.get("narrative"):
                st.markdown(f"**Narrative:** {schema['narrative']}")
            col_map = schema.get("column_map", {})
            if col_map:
                st.markdown("**Column Role Map:**")
                role_df = pd.DataFrame(
                    [{"Column": k, "Assigned Role": v} for k, v in col_map.items()]
                )
                st.dataframe(role_df, use_container_width=True, hide_index=True)
            if schema.get("raw_response"):
                st.markdown("**Raw LLM Output:**")
                st.text_area("", value=schema["raw_response"], height=180, label_visibility="collapsed")

        # --- Engineer output
        with st.expander("② Execution Engineer Agent — Generated Analysis Script", expanded=False):
            exec_res = res.get("execution", {})
            if exec_res.get("success"):
                attempts = exec_res.get("attempts", 1)
                st.markdown(
                    f"{badge('✔ SUCCEEDED', 'done')} &nbsp; "
                    f"<span style='color:#94A3B8;font-size:0.82rem;'>Completed in {attempts} attempt(s)</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(badge("✘ FAILED", "failed"), unsafe_allow_html=True)
                st.error(exec_res.get("error", "Unknown error"))

            if exec_res.get("code_used"):
                st.code(exec_res["code_used"], language="python")

        # --- Console output
        with st.expander("③ Sandboxed Execution — Console (stdout)", expanded=True):
            exec_res = res.get("execution", {})
            st.text_area(
                "Console Output",
                value=exec_res.get("terminal_output", "(No output captured)"),
                height=220,
                label_visibility="collapsed",
            )

# ---------------------------------------------------------------------------
# TAB 3 — Strategic Insights
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### Executive Retention Intelligence Dashboard")

    if "pipeline_results" not in st.session_state:
        st.markdown(
            "<div class='glass-card' style='text-align:center; padding:48px;'>"
            "<div style='font-size:2.5rem;'>🔒</div>"
            "<div style='font-weight:600; margin-top:12px; color:#94A3B8;'>"
            "Run the Agentic Pipeline in the <b>Agent Console</b> tab to unlock insights."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    res = st.session_state["pipeline_results"]
    exec_res = res.get("execution", {})
    terminal_output = exec_res.get("terminal_output", "")

    # ── Top KPI strip ──────────────────────────────────────────────────────
    df_full_tab3 = pd.read_csv(csv_path)
    schema = res.get("schema", {})
    cbr = schema.get("columns_by_role", {})
    target_col_detected = (cbr.get("target") or [None])[0]
    churn_rate = compute_churn_prevalence(df_full_tab3, target_col=target_col_detected)
    artifacts = exec_res.get("artifacts", [])

    k1, k2, k3, k4 = st.columns(4)
    if churn_rate is not None:
        k1.metric("🎯 Global Churn Rate", f"{churn_rate:.1f}%")
    k2.metric("📁 Records Analysed", f"{len(df_full_tab3):,}")
    k3.metric("🖼 Charts Generated", len(artifacts))
    k4.metric("🧠 Memory Turns", len(orchestrator.get_memory_snapshot()))

    st.markdown("---")

    # ── Two-column layout: report + chart carousel ─────────────────────────
    col_report, col_charts = st.columns([3, 2])

    with col_report:
        st.markdown("#### 📋 Agentic Executive Report")
        report = res.get("report", "")
        st.markdown(
            f"<div class='glass-card'>{report}</div>",
            unsafe_allow_html=True,
        )

    with col_charts:
        st.markdown("#### 🖼 Generated Visual Artifacts")
        # Use the artifacts list from execution results — fully dynamic for any schema
        artifacts_from_run = exec_res.get("artifacts", [])
        # Also scan disk for churn_heatmap.png which is always expected
        all_charts = sorted(
            set(artifacts_from_run) | {f for f in ["churn_heatmap.png"] if os.path.exists(f)}
        )
        chart_files = [f for f in all_charts if os.path.exists(f)]
        if chart_files:
            selected_chart = st.selectbox(
                "Select chart",
                options=chart_files,
                format_func=lambda x: (
                    "📊 Feature Correlation Heatmap" if x == "churn_heatmap.png"
                    else f"📈 {x.replace('_', ' ').replace('.png', '').title()}"
                ),
                label_visibility="collapsed",
            )
            st.image(selected_chart, use_container_width=True)
        else:
            st.markdown(
                "<div class='glass-card' style='text-align:center; color:#64748B;'>"
                "No chart artifacts found on disk."
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Conversational chat ────────────────────────────────────────────────
    memory_snap = orchestrator.get_memory_snapshot()
    mem_fill = len(memory_snap) / memory_window if memory_window else 0

    st.markdown(
        f"#### 💬 Chat with Executive Critic "
        f"<span style='font-size:0.78rem; color:#94A3B8; font-weight:400;'>"
        f"({len(memory_snap)}/{memory_window} turns in memory)</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='memory-bar' style='width:{int(mem_fill * 100)}%;'></div>",
        unsafe_allow_html=True,
    )

    # Initialise chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Render historical turns
    for msg in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["agent"])

    # Chat input
    user_query = st.chat_input(
        "Ask a specific question about the analysis… e.g. 'What is the churn rate in Germany?'"
    )

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("Executive Critic is consulting session memory and data logs…"):
            reply = orchestrator.conversational_follow_up(
                user_question=user_query,
                execution_result=exec_res,
            )

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state["chat_history"].append({"user": user_query, "agent": reply})
        st.rerun()

    if st.session_state.get("chat_history"):
        if st.button("🗑 Clear Chat History"):
            st.session_state["chat_history"] = []
            orchestrator.clear_chat_memory()
            st.rerun()