# env/models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    RETURNS = "returns"
    GENERAL = "general"
    ESCALATION = "escalation"


class ActionType(str, Enum):
    CLASSIFY_PRIORITY = "classify_priority"
    ROUTE_TICKET = "route_ticket"
    DRAFT_RESPONSE = "draft_response"
    RESOLVE_TICKET = "resolve_ticket"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_name: str
    customer_email: str
    created_at: str
    sla_deadline_hours: int          # Task completion deadline in hours
    true_priority: Priority           # Ground truth (used by grader)
    true_department: Department       # Ground truth (used by grader)
    keywords: List[str]
    sentiment: str                    # angry/neutral/happy
    
    # To be populated after agent actions
    assigned_priority: Optional[Priority] = None
    assigned_department: Optional[Department] = None
    response_drafted: Optional[str] = None
    is_resolved: bool = False
    resolution_time_minutes: Optional[float] = None


class Observation(BaseModel):
    """Data model for observations provided to the agent at each step."""
    inbox: List[Dict[str, Any]]           # Pending tickets (simplified view)
    current_ticket: Optional[Dict[str, Any]] = None  # Active ticket
    resolved_tickets: List[str] = []      # Resolved ticket IDs
    sla_violations: int = 0               # Number of SLA misses
    correct_routings: int = 0             # Number of correctly routed tickets
    correct_priorities: int = 0           # Number of correctly prioritized tickets
    total_tickets: int = 0
    steps_taken: int = 0
    task_id: str = ""
    instructions: str = ""               # Task instructions
    last_action_result: str = ""         # Last action ka result
    last_action_error: Optional[str] = None


class Action(BaseModel):
    """Data model for actions taken by the agent."""
    action_type: ActionType
    ticket_id: str
    priority: Optional[Priority] = None
    department: Optional[Department] = None
    response_text: Optional[str] = None
    reasoning: Optional[str] = None      # Agent's reasoning (for logging/debugging)


class Reward(BaseModel):
    """Data model for rewards provided after each step."""
    value: float = Field(ge=-1.0, le=1.0)
    reason: str
    breakdown: Dict[str, float] = {}     # Detailed components of the reward


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}
