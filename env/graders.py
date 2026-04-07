# env/graders.py
from typing import List, Dict
from .models import Ticket, Priority


def strict_clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — judge rejects exact 0.0 or 1.0."""
    return round(min(0.999, max(0.001, score)), 4)


class TaskGrader:
    """Base grader class"""
    
    def grade(self, tickets: List[Ticket], steps_taken: int, max_steps: int) -> float:
        raise NotImplementedError


class EasyGrader(TaskGrader):
    """
    Task Easy: Priority Classification
    Score = correct_priorities / total_tickets
    Partial credit for "close" priorities
    """
    
    PRIORITY_ORDER = {Priority.LOW: 0, Priority.MEDIUM: 1, 
                      Priority.HIGH: 2, Priority.CRITICAL: 3}
    
    def grade(self, tickets: List[Ticket], steps_taken: int, max_steps: int) -> float:
        if not tickets:
            return 0.0
        
        total_score = 0.0
        graded_count = 0
        
        for ticket in tickets:
            if ticket.assigned_priority is None:
                continue  # Not attempted
            
            graded_count += 1
            
            if ticket.assigned_priority == ticket.true_priority:
                total_score += 1.0  # Perfect match
            else:
                # Partial credit for adjacent priority
                true_level = self.PRIORITY_ORDER[ticket.true_priority]
                pred_level = self.PRIORITY_ORDER[ticket.assigned_priority]
                distance = abs(true_level - pred_level)
                
                if distance == 1:
                    total_score += 0.5   # One level off
                elif distance == 2:
                    total_score += 0.2   # Two levels off
                else:
                    total_score += 0.0   # Completely wrong
        
        if graded_count == 0:
            return 0.0
        
        # Coverage penalty: Apply penalty if not all tickets are classified.
        coverage = graded_count / len(tickets)
        accuracy = total_score / graded_count
        
        final_score = accuracy * coverage
        return strict_clamp(final_score)


class MediumGrader(TaskGrader):
    """
    Task Medium: Priority + Routing
    Score = 0.4 * priority_accuracy + 0.6 * routing_accuracy
    """
    
    PRIORITY_ORDER = {Priority.LOW: 0, Priority.MEDIUM: 1,
                      Priority.HIGH: 2, Priority.CRITICAL: 3}
    
    def grade(self, tickets: List[Ticket], steps_taken: int, max_steps: int) -> float:
        if not tickets:
            return 0.0
        
        priority_scores = []
        routing_scores = []
        
        for ticket in tickets:
            # Priority scoring
            if ticket.assigned_priority is not None:
                if ticket.assigned_priority == ticket.true_priority:
                    priority_scores.append(1.0)
                else:
                    distance = abs(
                        self.PRIORITY_ORDER[ticket.assigned_priority] - 
                        self.PRIORITY_ORDER[ticket.true_priority]
                    )
                    priority_scores.append(max(0.0, 1.0 - distance * 0.4))
            else:
                priority_scores.append(0.0)
            
            # Routing scoring
            if ticket.assigned_department is not None:
                if ticket.assigned_department == ticket.true_department:
                    routing_scores.append(1.0)
                else:
                    routing_scores.append(0.0)
            else:
                routing_scores.append(0.0)
        
        avg_priority = sum(priority_scores) / len(priority_scores)
        avg_routing = sum(routing_scores) / len(routing_scores)
        
        # Efficiency bonus: fewer steps = small bonus
        step_efficiency = max(0, 1.0 - (steps_taken / (len(tickets) * 3)))
        efficiency_bonus = step_efficiency * 0.05
        
        combined = 0.4 * avg_priority + 0.6 * avg_routing + efficiency_bonus
        return strict_clamp(combined)


class HardGrader(TaskGrader):
    """
    Task Hard: Full Pipeline
    Score = priority(20%) + routing(25%) + response_quality(30%) + sla_compliance(25%)
    """
    
    PRIORITY_ORDER = {Priority.LOW: 0, Priority.MEDIUM: 1,
                      Priority.HIGH: 2, Priority.CRITICAL: 3}
    
    def grade(self, tickets: List[Ticket], steps_taken: int, max_steps: int) -> float:
        if not tickets:
            return 0.0
        
        priority_score = self._score_priorities(tickets)
        routing_score = self._score_routing(tickets)
        response_score = self._score_responses(tickets)
        sla_score = self._score_sla(tickets)
        resolution_score = self._score_resolution(tickets)
        
        # Weighted combination
        final = (
            0.20 * priority_score +
            0.25 * routing_score +
            0.25 * response_score +
            0.20 * sla_score +
            0.10 * resolution_score
        )
        
        return strict_clamp(final)
    
    def _score_priorities(self, tickets):
        scores = []
        for t in tickets:
            if t.assigned_priority == t.true_priority:
                scores.append(1.0)
            elif t.assigned_priority is not None:
                dist = abs(self.PRIORITY_ORDER[t.assigned_priority] - 
                          self.PRIORITY_ORDER[t.true_priority])
                scores.append(max(0, 1.0 - dist * 0.35))
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0
    
    def _score_routing(self, tickets):
        correct = sum(1 for t in tickets 
                     if t.assigned_department == t.true_department)
        return correct / len(tickets) if tickets else 0.0
    
    def _score_responses(self, tickets):
        scores = []
        for ticket in tickets:
            if ticket.response_drafted is None:
                scores.append(0.0)
                continue
            
            response = ticket.response_drafted.lower()
            score = 0.0
            
            # Minimum length check
            if len(ticket.response_drafted) > 50:
                score += 0.3
            
            # Personalization: customer name mention
            if ticket.customer_name.lower() in response or "dear customer" in response:
                score += 0.2
            
            # Keywords from ticket addressed
            keywords_addressed = sum(
                1 for kw in ticket.keywords if kw.lower() in response
            )
            keyword_score = min(1.0, keywords_addressed / max(1, len(ticket.keywords)))
            score += 0.3 * keyword_score
            
            # Professional closing
            closings = ["regards", "sincerely", "thank you", "best", "support team"]
            if any(c in response for c in closings):
                score += 0.2
            
            scores.append(min(1.0, score))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _score_sla(self, tickets):
        # Critical tickets resolved = full points, others proportional
        critical_tickets = [t for t in tickets if t.true_priority == Priority.CRITICAL]
        high_tickets = [t for t in tickets if t.true_priority == Priority.HIGH]
        
        scores = []
        for t in critical_tickets:
            scores.append(1.0 if t.is_resolved else 0.0)
        for t in high_tickets:
            scores.append(0.8 if t.is_resolved else 0.0)
        for t in tickets:
            if t.true_priority not in [Priority.CRITICAL, Priority.HIGH]:
                scores.append(0.4 if t.is_resolved else 0.1)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _score_resolution(self, tickets):
        resolved = sum(1 for t in tickets if t.is_resolved)
        return resolved / len(tickets) if tickets else 0.0


def get_grader(task_id: str) -> TaskGrader:
    graders = {
        "task_easy": EasyGrader(),
        "task_medium": MediumGrader(),
        "task_hard": HardGrader()
    }
    return graders.get(task_id, EasyGrader())
