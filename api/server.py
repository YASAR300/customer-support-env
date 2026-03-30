# api/server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add the parent directory to sys.path to allow importing from 'env'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType

app = FastAPI(
    title="Customer Support Triage OpenEnv",
    description="Real-world customer support ticket management environment for AI agents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global environment instance
env = CustomerSupportEnv()


class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class ActionRequest(BaseModel):
    action_type: str
    ticket_id: str
    priority: Optional[str] = None
    department: Optional[str] = None
    response_text: Optional[str] = None
    reasoning: Optional[str] = None


@app.get("/")
def root():
    return {
        "name": "Customer Support Triage OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": ["task_easy", "task_medium", "task_hard"]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    try:
        task_id = "task_easy"
        if request and request.task_id:
            task_id = request.task_id
        obs = env.reset(task_id=task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: ActionRequest):
    try:
        action = Action(
            action_type=ActionType(request.action_type),
            ticket_id=request.ticket_id,
            priority=request.priority,
            department=request.department,
            response_text=request.response_text,
            reasoning=request.reasoning
        )
        result = env.step(action)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    from env.tasks import TASKS
    return {
        task_id: {
            "name": task.name,
            "difficulty": task.difficulty,
            "description": task.description,
            "max_steps": task.max_steps,
            "ticket_count": task.ticket_count
        }
        for task_id, task in TASKS.items()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
