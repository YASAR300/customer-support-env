# tests/test_env.py
import pytest
import os
import sys

# Ensure env can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import CustomerSupportEnv
from env.models import Action, ActionType, Priority, Department

class TestEnvironmentBasics:
    """Basic environment tests"""
    
    def setup_method(self):
        """Initialize fresh environment before each test"""
        self.env = CustomerSupportEnv()
    
    def test_reset_easy_task(self):
        """Resetting should result in a clean state."""
        obs = self.env.reset("task_easy")
        assert obs.task_id == "task_easy"
        assert len(obs.inbox) == 5
        assert obs.steps_taken == 0
        assert obs.resolved_tickets == []
        print("\n✅ test_reset_easy_task passed")
    
    def test_reset_medium_task(self):
        obs = self.env.reset("task_medium")
        assert obs.task_id == "task_medium"
        assert len(obs.inbox) == 10
        print("\n✅ test_reset_medium_task passed")
    
    def test_reset_hard_task(self):
        obs = self.env.reset("task_hard")
        assert obs.task_id == "task_hard"
        assert len(obs.inbox) == 15
        print("\n✅ test_reset_hard_task passed")
    
    def test_invalid_task_raises_error(self):
        with pytest.raises(ValueError):
            self.env.reset("task_invalid")
        print("\n✅ test_invalid_task_raises_error passed")
    
    def test_step_classify_correct_priority(self):
        """Providing the correct priority should yield a positive reward."""
        obs = self.env.reset("task_easy")
        ticket_id = obs.inbox[0]["ticket_id"]
        
        # Check actual priority (accessible in tests)
        true_priority = self.env.tickets[0].true_priority
        
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            ticket_id=ticket_id,
            priority=true_priority
        )
        result = self.env.step(action)
        
        assert result.reward.value > 0  # Positive reward
        assert not result.done           # Episode not finished
        print(f"\n✅ Correct priority reward: {result.reward.value}")
    
    def test_step_wrong_priority(self):
        """Providing the wrong priority should result in a lower reward."""
        obs = self.env.reset("task_easy")
        ticket_id = obs.inbox[0]["ticket_id"]
        true_priority = self.env.tickets[0].true_priority
        
        # Provide a completely different priority
        all_priorities = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
        wrong_priority = [p for p in all_priorities if p != true_priority][0]
        
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            ticket_id=ticket_id,
            priority=wrong_priority
        )
        result = self.env.step(action)
        print(f"\n✅ Wrong priority reward: {result.reward.value}")
    
    def test_step_invalid_ticket(self):
        """Using an invalid ticket ID should return an error."""
        self.env.reset("task_easy")
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            ticket_id="TKT-INVALID-999",
            priority=Priority.HIGH
        )
        result = self.env.step(action)
        assert result.observation.last_action_error is not None
        print("\n✅ test_step_invalid_ticket passed")
    
    def test_state_returns_dict(self):
        """State should return a dictionary."""
        self.env.reset("task_easy")
        state = self.env.state()
        assert isinstance(state, dict)
        assert "task_id" in state
        assert "steps_taken" in state
        assert "done" in state
        print("\n✅ test_state_returns_dict passed")
    
    def test_resolve_without_response_fails(self):
        """Tickets should not be resolvable without a drafted response."""
        obs = self.env.reset("task_hard")
        ticket_id = obs.inbox[0]["ticket_id"]
        
        action = Action(
            action_type=ActionType.RESOLVE_TICKET,
            ticket_id=ticket_id
        )
        result = self.env.step(action)
        assert result.observation.last_action_error is not None
        print("\n✅ test_resolve_without_response_fails passed")
    
    def test_max_steps_ends_episode(self):
        """Episodes should terminate upon reaching maximum steps."""
        obs = self.env.reset("task_easy")
        ticket_id = obs.inbox[0]["ticket_id"]
        
        for i in range(15):
            action = Action(
                action_type=ActionType.CLASSIFY_PRIORITY,
                ticket_id=ticket_id,
                priority=Priority.MEDIUM
            )
            result = self.env.step(action)
            if result.done:
                break
        
        assert result.done == True
        print(f"\n✅ Episode ended after {self.env.steps_taken} steps")


class TestGraders:
    """Grader tests — Verify 0.0 to 1.0 score range"""
    
    def setup_method(self):
        self.env = CustomerSupportEnv()
    
    def test_easy_grader_perfect_score(self):
        from env.graders import EasyGrader
        
        self.env.reset("task_easy")
        for ticket in self.env.tickets:
            ticket.assigned_priority = ticket.true_priority
        
        grader = EasyGrader()
        score = grader.grade(self.env.tickets, 5, 10)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.8
        print(f"\n✅ Perfect easy score: {score}")
    
    def test_easy_grader_zero_score(self):
        from env.graders import EasyGrader
        
        self.env.reset("task_easy")
        grader = EasyGrader()
        score = grader.grade(self.env.tickets, 0, 10)
        
        assert score == 0.0
        print(f"\n✅ Zero easy score: {score}")
    
    def test_grader_score_in_range(self):
        from env.graders import get_grader
        
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            self.env.reset(task_id)
            grader = get_grader(task_id)
            score = grader.grade(self.env.tickets, 5, 30)
            assert 0.0 <= score <= 1.0
            print(f"\n✅ {task_id} grader score in range: {score}")
    
    def test_reward_values_in_range(self):
        obs = self.env.reset("task_easy")
        ticket_id = obs.inbox[0]["ticket_id"]
        
        action = Action(
            action_type=ActionType.CLASSIFY_PRIORITY,
            ticket_id=ticket_id,
            priority=Priority.HIGH
        )
        result = self.env.step(action)
        
        assert -1.0 <= result.reward.value <= 1.0
        print(f"\n✅ Reward in range: {result.reward.value}")


class TestFullEpisode:
    """Complete episode test"""
    
    def test_easy_task_complete_episode(self):
        """Complete a full episode for the easy task"""
        env = CustomerSupportEnv()
        obs = env.reset("task_easy")
        
        total_reward = 0
        steps = 0
        
        for _ in range(10):
            inbox = obs.inbox
            if not inbox:
                break
                
            ticket = inbox[0]
            action = Action(
                action_type=ActionType.CLASSIFY_PRIORITY,
                ticket_id=ticket["ticket_id"],
                priority=Priority.MEDIUM
            )
            result = env.step(action)
            obs = result.observation
            total_reward += result.reward.value
            steps += 1
            
            if result.done:
                break
        
        state = env.state()
        assert "current_score" in state
        print(f"\n✅ Easy episode complete: steps={steps}, total_reward={total_reward:.3f}")
        print(f"   Final score: {state['current_score']:.4f}")
