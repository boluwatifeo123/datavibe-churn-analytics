# 📊 DataVibe: Multi-Agent Operational Churn Pipeline

DataVibe is an intelligent, multi-agent analytics pipeline engineered to automate customer churn and retention analysis using **Churn_Modelling.csv**. Built using the official **Google GenAI SDK**, **Gemini 2.5 Flash**, and **Streamlit**, the system orchestrates three specialized agent nodes alongside a sandboxed local code execution framework to translate raw metrics into verified strategic business reports—with 0% cloud compute costs and zero external billing dependencies.

This project was built as a Capstone submission for **Kaggle's 5-Day AI Agents: Intensive Vibe Coding Course with Google** (June 15–19, 2026).

---

## 🏗️ Multi-Agent Architecture & Data Flow

DataVibe splits analytical and reasoning tasks into distinct roles to ensure data accuracy and eliminate LLM math hallucinations:

1. **Schema Router Agent:** Inspects the structural metadata of the uploaded file ($N=3$ rows). It maps target parameters (`Exited`), flags categorical and continuous indicators, and blacklists non-predictive administrative features (`CustomerId`, `Surname`, `RowNumber`) to save token context.
2. **Code Execution Engineer Agent:** Translates the Schema Router's profile into an isolated Python data science script utilizing `pandas`, `numpy`, and `seaborn`.
3. **Local Subprocess Tool Sandbox:** Runs the generated code locally under a 30-second execution cutoff, intercepts stdout logs, catches runtime exceptions for dynamic agent self-correction loops, and exports high-resolution visual correlations (`churn_heatmap.png`).
4. **Executive Critic Agent:** Reads the verified statistical console output, synthesizes a structured Markdown business report, and powers a stateful conversational memory array for user deep-dives.
