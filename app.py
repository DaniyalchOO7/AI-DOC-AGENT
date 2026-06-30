"""
app.py
------
Streamlit front end for the Document-to-Action Agent.
Users upload a PDF directly in the browser — no command line needed.
The agent processes it live and shows the full reasoning trace + summary.

Run:
    streamlit run app.py
"""

import time
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from extraction import extract_text
from agent import run_agent_on_text
from tools import get_budgets, LOG_FILE

st.set_page_config(
    page_title="Document-to-Action Agent",
    page_icon="🧾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.title("🧾 Document-to-Action Agent")
st.caption(
    "Upload a receipt or invoice. The agent extracts the data, reasons about "
    "it, checks it against a budget, logs it, and alerts you if needed — "
    "autonomously, using real Gemini function calls."
)

with st.expander("ℹ️ How this works"):
    st.markdown(
        """
        This isn't a simple "chat with your PDF" wrapper — it's an **agent**
        that decides what actions to take based on what it finds:

        1. **Extracts** text from your uploaded PDF (with OCR fallback for scanned docs)
        2. **Reasons** about which tools to call and in what order using Gemini function calling
        3. **Acts**: checks the expense against a budget, logs it permanently, and sends an alert if over budget
        4. **Shows you** the full reasoning trace — every tool call and result, not just the final answer

        Built with: Python · Streamlit · Gemini 2.0 Flash · pdfplumber · pytesseract
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------

left, right = st.columns([1.3, 1], gap="large")

# ── LEFT: upload + agent output ──────────────────────────────────────────────
with left:
    st.subheader("1. Upload your document")

    uploaded_file = st.file_uploader(
        label="Drop a receipt or invoice here",
        type=["pdf"],
        help="Supports text-based and scanned (image) PDFs.",
    )

    if uploaded_file is not None:
        st.success(f"📎 Loaded: **{uploaded_file.name}**")

        if st.button("▶ Run agent", type="primary", use_container_width=True):

            # Save the uploaded file to a temp location so pdfplumber can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.status("Agent is working...", expanded=True) as status:

                # Step 1 — extraction
                st.write("📄 Extracting text from document...")
                try:
                    raw_text = extract_text(tmp_path)
                except ValueError as e:
                    st.error(f"Extraction failed: {e}")
                    st.stop()

                with st.expander("View extracted text"):
                    st.text(raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""))

                # Step 2 — agent
                st.write("🤖 Agent reasoning and calling tools...")
                time.sleep(0.2)

                summary, trace = run_agent_on_text(raw_text)

                status.update(label="✅ Done!", state="complete", expanded=False)

            # Reasoning trace
            st.subheader("2. Agent reasoning trace")
            with st.expander("🔍 View every tool call and result", expanded=True):
                for step in trace:
                    if step.startswith("→"):
                        st.markdown(f"**{step}**")
                    else:
                        st.code(step, language=None)

            # Final summary
            st.subheader("3. Summary")
            st.success(summary)

    else:
        st.info("👆 Upload a PDF receipt or invoice to get started.")

# ── RIGHT: budget table + live log ───────────────────────────────────────────
with right:
    st.subheader("📊 Budget limits")
    budgets = get_budgets()

    st.caption("Edit any limit and click Save to update.")

    updated_budgets = {}
    for category, limit in budgets.items():
        label = category.replace("_", " ").title()
        new_val = st.number_input(
            label=f"{label} ($)",
            min_value=0,
            max_value=10000,
            value=int(limit),
            step=10,
            key=f"budget_{category}",
        )
        updated_budgets[category] = new_val

    if st.button("💾 Save budget limits", use_container_width=True):
        import json
        from tools import BUDGET_FILE
        BUDGET_FILE.write_text(json.dumps(updated_budgets, indent=2))
        st.success("✅ Budgets updated!")
        st.rerun()

    st.subheader("🧾 Expense log")

    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)

        # Colour-code the status column
        def colour_status(val):
            if isinstance(val, str) and "over" in val.lower():
                return "color: #c92a2a; font-weight: bold"
            return "color: #2b8a3e; font-weight: bold"

        styled = log_df.tail(10).style.map(colour_status, subset=["status"])
        st.dataframe(styled, hide_index=True, use_container_width=True)

        # Metrics row
        col1, col2 = st.columns(2)
        col1.metric("Documents processed", len(log_df))
        over = log_df["status"].str.contains("over", case=False, na=False).sum()
        col2.metric("Flagged over budget", int(over))

        # Spending chart
        if len(log_df) > 1:
            st.subheader("📈 Spending by category")
            chart_data = (
                log_df.groupby("category")["amount"]
                .sum()
                .reset_index()
                .set_index("category")
            )
            st.bar_chart(chart_data)

    else:
        st.info("No expenses logged yet. Upload a document to get started.")

st.divider()
st.caption("Built by Daniyal Munir · github.com/DaniyalchOO7")