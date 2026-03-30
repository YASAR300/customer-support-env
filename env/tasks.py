# env/tasks.py
from dataclasses import dataclass
from typing import List
from .data_generator import generate_ticket_batch


@dataclass
class Task:
    task_id: str
    name: str
    difficulty: str
    description: str
    instructions: str
    max_steps: int
    ticket_count: int
    seed: int


TASKS = {
    "task_easy": Task(
        task_id="task_easy",
        name="Ticket Priority Classification",
        difficulty="easy",
        description="Classify 5 customer support tickets by priority level",
        instructions="""
You are a customer support triage agent. 
TASK: Classify each ticket's priority correctly.
- CRITICAL: System down, data loss, security breach
- HIGH: Major feature broken, billing issues, wrong items
- MEDIUM: Feature partially working, minor billing questions  
- LOW: General questions, documentation requests

For each ticket, use action: classify_priority
Available priorities: low, medium, high, critical
""",
        max_steps=10,
        ticket_count=5,
        seed=42
    ),
    
    "task_medium": Task(
        task_id="task_medium",
        name="Multi-Department Routing",
        difficulty="medium",
        description="Classify priority AND route 10 tickets to correct departments",
        instructions="""
You are a customer support triage agent.
TASK: For each ticket, you must:
1. classify_priority (low/medium/high/critical)
2. route_ticket to correct department (billing/technical/returns/general/escalation)

Routing rules:
- billing: payment, invoice, charge, refund issues
- technical: bugs, errors, crashes, login, API issues
- returns: wrong items, damaged goods, return requests
- general: how-to questions, feature requests, general info
- escalation: VIP customers, legal threats, media mentions

Complete BOTH actions for each ticket.
""",
        max_steps=25,
        ticket_count=10,
        seed=123
    ),
    
    "task_hard": Task(
        task_id="task_hard",
        name="Full Resolution Pipeline",
        difficulty="hard",
        description="Complete triage, routing, response drafting, and resolution for 15 tickets within SLA",
        instructions="""
You are a senior customer support agent managing a full ticket queue.
TASK: For each ticket you must:
1. classify_priority 
2. route_ticket to correct department
3. draft_response (professional, empathetic, actionable response)
4. resolve_ticket (after response is drafted)

CRITICAL: 
- Critical tickets MUST be processed first
- High priority tickets must be resolved within SLA
- Responses must address the customer's specific problem
- Do NOT resolve without drafting a response first

Your score depends on: accuracy, SLA compliance, and response quality.
""",
        max_steps=70,
        ticket_count=15,
        seed=777
    )
}
