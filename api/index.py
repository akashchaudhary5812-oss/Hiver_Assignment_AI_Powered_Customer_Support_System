import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.agent import SupportAgent

# Initialize FastAPI app
app = FastAPI(title="Hiver AI Customer Support", version="1.0.0")

# Lazy initialization of SupportAgent
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = SupportAgent()
    return agent

class ChatRequest(BaseModel):
    query: str
    history: List[str] = []

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Hiver AI Customer Support"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        res = get_agent().process(request.query, history=request.history)
        return {
            "draft_reply": res.get("draft_reply"),
            "generation_engine": res.get("generation_engine"),
            "intent": res.get("intent"),
            "intent_confidence": res.get("intent_confidence"),
            "escalate": res.get("escalate"),
            "escalation_status": res.get("escalation_status"),
            "risk_level": res.get("risk_level"),
            "escalation_reason": res.get("escalation_reason"),
            "next_action": res.get("next_action"),
            "grounded_sources": res.get("grounded_sources"),
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
