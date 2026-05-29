from __future__ import annotations

from pydantic import BaseModel
from fastapi import FastAPI
from app.agent_backend import ask_agent

app = FastAPI(title="Portfolio Risk Agent API")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(answer=ask_agent(req.question))
