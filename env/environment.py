# env/environment.py
import copy
from typing import Optional, Dict, Any, List
from .models import (
    Observation, Action, Reward, StepResult, 
    Ticket, ActionType, Priority, Department
)
from .tasks import TASKS, Task
from .data_generator import generate_ticket_batch
from .graders import get_grader
from .reward import RewardCalculator


class CustomerSupportEnv:
    """
    Main OpenEnv Environment: Customer Support Ticket Management
    
    An AI agent manages a customer support inbox:
    1. Classify ticket priority
    2. Route to correct department  
    3. Draft appropriate response
    4. Resolve ticket
    
    The environment tracks accuracy, SLA compliance, and efficiency.
    """
    
    def __init__(self):
        self.current_task: Optional[Task] = None
        self.tickets: List[Ticket] = []
        self.current_ticket_idx: int = 0
        self.steps_taken: int = 0
        self.done: bool = False
        self.reward_calculator = RewardCalculator()
        self._state_history: List[Dict] = []
    
    def reset(self, task_id: str = "task_easy") -> Observation:
        """Reset the environment to a fresh start."""
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
        
        self.current_task = TASKS[task_id]
        self.tickets = generate_ticket_batch(
            count=self.current_task.ticket_count,
            seed=self.current_task.seed
        )
        self.current_ticket_idx = 0
        self.steps_taken = 0
        self.done = False
        self._state_history = []
        self.reward_calculator.reset()
        
        return self._build_observation(
            last_action_result="Environment reset. Task started.",
            last_action_error=None
        )
    
    def step(self, action: Action) -> StepResult:
        """Take an action and return the result."""
        if self.done:
            return StepResult(
                observation=self._build_observation("Episode already done.", None),
                reward=Reward(value=0.0, reason="Episode already complete"),
                done=True,
                info={"warning": "Episode already done"}
            )
        
        self.steps_taken += 1
        error = None
        result_message = ""
        reward_value = 0.0
        reward_reason = ""
        
        # Locate the ticket
        target_ticket = self._find_ticket(action.ticket_id)
        
        if target_ticket is None:
            error = f"Ticket {action.ticket_id} not found"
            reward_value = -0.1
            reward_reason = "Invalid ticket ID"
        else:
            # Execute the action
            reward_value, reward_reason, result_message, error = self._execute_action(
                action, target_ticket
            )
        
        # Check if done
        self.done = self._check_done()
        
        # Final score if done
        if self.done:
            final_score = self._compute_final_score()
            reward_value += final_score * 0.3  # Add final bonus
            reward_reason += f" | Final score: {final_score:.3f}"
        
        reward = Reward(
            value=max(-1.0, min(1.0, reward_value)),
            reason=reward_reason,
            breakdown={"base": reward_value}
        )
        
        observation = self._build_observation(result_message, error)
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=self.done,
            info={
                "steps_taken": self.steps_taken,
                "max_steps": self.current_task.max_steps,
                "task_id": self.current_task.task_id,
                "tickets_processed": self._count_processed(),
                "current_score": self._compute_final_score()
            }
        )
    
    def state(self) -> Dict[str, Any]:
        """Return the current full state of the environment."""
        return {
            "task_id": self.current_task.task_id if self.current_task else None,
            "steps_taken": self.steps_taken,
            "done": self.done,
            "tickets": [t.model_dump() for t in self.tickets],
            "current_score": self._compute_final_score() if self.current_task else 0.0
        }
    
    def _execute_action(self, action: Action, ticket: Ticket):
        """Execute the specific action and return results."""
        reward = 0.0
        reason = ""
        message = ""
        error = None
        
        if action.action_type == ActionType.CLASSIFY_PRIORITY:
            if action.priority is None:
                error = "Priority required for classify_priority action"
                reward = -0.05
            else:
                ticket.assigned_priority = action.priority
                if action.priority == ticket.true_priority:
                    reward = 0.3
                    reason = f"\u2705 Correct priority: {action.priority}"
                    message = f"Priority set to {action.priority} for {ticket.ticket_id}"
                else:
                    # Grant partial credit for similar priorities
                    priority_order = {Priority.LOW: 0, Priority.MEDIUM: 1, 
                                     Priority.HIGH: 2, Priority.CRITICAL: 3}
                    dist = abs(priority_order[action.priority] - 
                              priority_order[ticket.true_priority])
                    reward = max(-0.1, 0.15 - dist * 0.1)
                    reason = f"\u274c Wrong priority. Got {action.priority}, expected {ticket.true_priority}"
                    message = f"Priority set to {action.priority} (incorrect)"
        
        elif action.action_type == ActionType.ROUTE_TICKET:
            if action.department is None:
                error = "Department required for route_ticket action"
                reward = -0.05
            else:
                ticket.assigned_department = action.department
                if action.department == ticket.true_department:
                    reward = 0.35
                    reason = f"\u2705 Correct department: {action.department}"
                    message = f"Routed to {action.department}"
                else:
                    reward = -0.1
                    reason = f"\u274c Wrong department. Got {action.department}, expected {ticket.true_department}"
                    message = f"Routed to {action.department} (incorrect)"
        
        elif action.action_type == ActionType.DRAFT_RESPONSE:
            if not action.response_text or len(action.response_text) < 20:
                error = "Response text too short (min 20 chars)"
                reward = -0.05
            else:
                ticket.response_drafted = action.response_text
                # Response quality score
                quality = self._evaluate_response_quality(ticket, action.response_text)
                reward = quality * 0.4
                reason = f"Response drafted (quality: {quality:.2f})"
                message = f"Response drafted for {ticket.ticket_id}"
        
        elif action.action_type == ActionType.RESOLVE_TICKET:
            if not ticket.response_drafted:
                error = "Cannot resolve without drafting response first"
                reward = -0.15
                reason = "Missing response before resolution"
            elif ticket.is_resolved:
                error = "Ticket already resolved"
                reward = -0.05
            else:
                ticket.is_resolved = True
                # Bonus for priority ordering (critical first)
                if ticket.true_priority == Priority.CRITICAL:
                    reward = 0.4
                elif ticket.true_priority == Priority.HIGH:
                    reward = 0.25
                else:
                    reward = 0.15
                reason = f"\u2705 Ticket resolved: {ticket.ticket_id}"
                message = f"Ticket {ticket.ticket_id} resolved"
        
        elif action.action_type == ActionType.ESCALATE:
            ticket.assigned_department = Department.ESCALATION
            if ticket.sentiment == "angry" or ticket.true_priority == Priority.CRITICAL:
                reward = 0.2
                reason = "Appropriate escalation"
            else:
                reward = -0.05
                reason = "Unnecessary escalation (wasted resources)"
            message = f"Ticket {ticket.ticket_id} escalated"
        
        # Efficiency penalty (incentivize fewer steps)
        step_penalty = -0.01
        reward += step_penalty
        
        return reward, reason, message, error
    
    def _evaluate_response_quality(self, ticket: Ticket, response: str) -> float:
        """Calculate response quality score (0.0 to 1.0)."""
        score = 0.0
        response_lower = response.lower()
        
        if len(response) > 100: score += 0.25
        elif len(response) > 50: score += 0.15
        
        keywords_hit = sum(1 for kw in ticket.keywords if kw in response_lower)
        score += 0.35 * (keywords_hit / max(1, len(ticket.keywords)))
        
        if any(c in response_lower for c in ["regards", "sincerely", "thank you", "best"]):
            score += 0.2
        
        if ticket.customer_name.lower() in response_lower:
            score += 0.1
        
        if ticket.sentiment == "angry" and any(w in response_lower for w in 
                                               ["apologize", "sorry", "understand", "inconvenience"]):
            score += 0.1
        
        return min(1.0, score)
    
    def _find_ticket(self, ticket_id: str) -> Optional[Ticket]:
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None
    
    def _check_done(self) -> bool:
        if self.steps_taken >= self.current_task.max_steps:
            return True
        all_resolved = all(t.is_resolved for t in self.tickets)
        return all_resolved
    
    def _count_processed(self) -> int:
        return sum(1 for t in self.tickets if t.assigned_priority is not None)
    
    def _compute_final_score(self) -> float:
        if not self.current_task:
            return 0.0
        grader = get_grader(self.current_task.task_id)
        return grader.grade(self.tickets, self.steps_taken, self.current_task.max_steps)
    
    def _build_observation(self, last_action_result: str, last_action_error: Optional[str]) -> Observation:
        inbox_view = []
        for t in self.tickets:
            if not t.is_resolved:
                inbox_view.append({
                    "ticket_id": t.ticket_id,
                    "subject": t.subject,
                    "body": t.body[:300],  # Truncated
                    "sentiment": t.sentiment,
                    "sla_hours": t.sla_deadline_hours,
                    "assigned_priority": t.assigned_priority,
                    "assigned_department": t.assigned_department,
                    "response_drafted": t.response_drafted is not None,
                    "is_resolved": t.is_resolved
                })
        
        resolved_ids = [t.ticket_id for t in self.tickets if t.is_resolved]
        correct_priorities = sum(1 for t in self.tickets 
                                if t.assigned_priority == t.true_priority)
        correct_routings = sum(1 for t in self.tickets 
                              if t.assigned_department == t.true_department)
        
        return Observation(
            inbox=inbox_view,
            resolved_tickets=resolved_ids,
            sla_violations=0,
            correct_priorities=correct_priorities,
            correct_routings=correct_routings,
            total_tickets=len(self.tickets),
            steps_taken=self.steps_taken,
            task_id=self.current_task.task_id if self.current_task else "",
            instructions=self.current_task.instructions if self.current_task else "",
            last_action_result=last_action_result,
            last_action_error=last_action_error
        )
    
    def close(self):
        """Cleanup environment (required for OpenEnv standard)"""
        self.tickets = []
        self.done = True
