"""
FastAPI Backend
Endpoints:
  POST /chat          — send a message, get a response
  POST /clear         — clear session memory
  GET  /health        — health check
  GET  /kpis          — quick KPI dashboard (no LLM call)
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend.agent import chat, clear_session
from backend.kpi_tools import get_kpi_dashboard

app = FastAPI(
    title="FP&A Insight Agent",
    description="Agentic AI assistant for financial planning & analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

class ClearRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "FP&A Insight Agent"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        response = chat(session_id=req.session_id, user_message=req.message)
        return ChatResponse(session_id=req.session_id, response=response)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@app.post("/clear")
def clear_endpoint(req: ClearRequest):
    clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


@app.get("/kpis")
def kpis_endpoint():
    """Returns raw KPI dashboard without LLM — fast snapshot."""
    try:
        result = get_kpi_dashboard.invoke({})
        return {"kpis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
