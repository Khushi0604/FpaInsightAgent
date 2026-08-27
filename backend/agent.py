"""
FP&A LangGraph Agent
─────────────────────
Flow per query:
  1. Retrieve relevant CSV context via RAG (FAISS)
  2. Inject context + conversation memory into system prompt
  3. LLM decides: answer directly OR call a KPI tool
  4. If tool called → run it → feed result back to LLM
  5. Final answer returned with sources

Memory: full conversation history kept in-process per session.
Sessions are identified by session_id (passed from Streamlit).
"""

import os
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from backend.rag_engine import retrieve, build_index
from backend.kpi_tools import ALL_TOOLS

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    rag_context: str                           # retrieved chunks for this query


# ── LLM setup ─────────────────────────────────────────────────────────────────
def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=api_key,
    )
    return llm.bind_tools(ALL_TOOLS)


# ── Nodes ─────────────────────────────────────────────────────────────────────
def rag_retrieval_node(state: AgentState) -> AgentState:
    """Pull relevant FP&A context before calling the LLM."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    context = retrieve(last_human, top_k=8)
    return {"rag_context": context}


def llm_node(state: AgentState) -> AgentState:
    """Call the LLM with full memory + RAG context injected into system prompt."""
    system_prompt = f"""You are an expert FP&A (Financial Planning & Analysis) AI assistant.
You help analysts and finance teams understand their SaaS business metrics.

You have access to the following real-time tools:
- get_kpi_dashboard: Full snapshot of all KPIs
- get_mrr_summary: MRR/ARR trends over N months
- get_churn_analysis: Churn rate and breakdown
- get_cac_ltv: CAC, LTV, and unit economics
- get_customer_segments: Customer breakdown by plan/country/industry

Use tools when the user asks for specific metrics or analysis.
For context and background questions, use the RAG data below.

RELEVANT DATA CONTEXT:
{state['rag_context']}

Rules:
- Always cite specific numbers when answering.
- If a tool would give a better answer than the context, call the tool.
- For multi-part questions, break your answer into sections.
- Be concise but complete. Finance teams hate vague answers.
"""
    messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
    llm = _get_llm()
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}


def tool_execution_node(state: AgentState) -> AgentState:
    """Execute any tool calls the LLM requested."""
    tool_map = {t.name: t for t in ALL_TOOLS}
    last_msg  = state["messages"][-1]
    results   = []

    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Error running {tool_name}: {e}"
        else:
            result = f"Unknown tool: {tool_name}"

        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": results}


def should_use_tools(state: AgentState) -> str:
    """Router: did the LLM request tool calls?"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build Graph ────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("rag",   rag_retrieval_node)
    graph.add_node("llm",   llm_node)
    graph.add_node("tools", tool_execution_node)

    graph.set_entry_point("rag")
    graph.add_edge("rag", "llm")
    graph.add_conditional_edges("llm", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")   # after tools → back to LLM for final answer

    return graph.compile()


# ── Session memory store ───────────────────────────────────────────────────────
_sessions: dict[str, list] = {}

compiled_graph = None

def get_graph():
    global compiled_graph
    if compiled_graph is None:
        build_index()
        compiled_graph = build_graph()
    return compiled_graph


def chat(session_id: str, user_message: str) -> str:
    """
    Main entry point. Pass a session_id to maintain memory across turns.
    Returns the agent's final text response.
    """
    graph = get_graph()

    # Load or init session history
    history = _sessions.get(session_id, [])
    history.append(HumanMessage(content=user_message))

    state = {"messages": history, "rag_context": ""}
    result = graph.invoke(state)

    # Save updated history (includes tool messages + AI responses)
    _sessions[session_id] = result["messages"]

    # Return the last AI text message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "I couldn't generate a response. Please try again."


def clear_session(session_id: str):
    """Clear memory for a session (called from Streamlit reset button)."""
    _sessions.pop(session_id, None)
