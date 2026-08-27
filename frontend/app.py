"""
FP&A Insight Agent — Streamlit Frontend
Calls the FastAPI backend at localhost:8000
"""

import streamlit as st
import requests
import uuid

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="FP&A Insight Agent",
    page_icon="📊",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 FP&A Insight Agent")
    st.markdown("Agentic AI assistant for financial planning & analysis.")
    st.divider()

    # Quick KPI snapshot
    st.subheader("📈 Live KPI Snapshot")
    if st.button("Load KPIs"):
        try:
            r = requests.get(f"{BACKEND}/kpis", timeout=10)
            if r.status_code == 200:
                st.code(r.json()["kpis"], language=None)
            else:
                st.error("Failed to load KPIs.")
        except Exception as e:
            st.error(f"Backend not reachable: {e}")

    st.divider()

    # Suggested queries
    st.subheader("💡 Try asking:")
    suggestions = [
        "Give me a full KPI dashboard overview",
        "What is our current MRR and ARR?",
        "Analyse our churn rate and which plan churns most",
        "What are our CAC and LTV numbers? Is the ratio healthy?",
        "Which countries have the most active customers?",
        "How has our MRR grown over the last 6 months?",
        "Compare our Starter vs Enterprise plan performance",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True):
            st.session_state["prefill"] = s

    st.divider()

    # Reset conversation
    if st.button("🔄 New Conversation", use_container_width=True):
        try:
            requests.post(
                f"{BACKEND}/clear",
                json={"session_id": st.session_state.session_id},
                timeout=5,
            )
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages   = []
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")


# ── Main chat area ────────────────────────────────────────────────────────────
st.header("FP&A Insight Agent 📊")
st.caption("Ask questions about your financial metrics, KPIs, churn, CAC, MRR, and more.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle sidebar suggestion prefill
prefill = st.session_state.pop("prefill", None)

# Chat input
user_input = st.chat_input("Ask about your FP&A data...") or prefill

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Analysing your FP&A data..."):
            try:
                r = requests.post(
                    f"{BACKEND}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": user_input,
                    },
                    timeout=60,
                )
                if r.status_code == 200:
                    response = r.json()["response"]
                else:
                    response = f"Error {r.status_code}: {r.text}"
            except requests.exceptions.ConnectionError:
                response = (
                    "⚠️ Cannot connect to backend. "
                    "Make sure FastAPI is running: `uvicorn backend.main:app --reload`"
                )
            except Exception as e:
                response = f"Unexpected error: {e}"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
