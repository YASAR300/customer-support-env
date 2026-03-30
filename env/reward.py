# env/reward.py
class RewardCalculator:
    """
    Helper class for tracking and calculating cumulative rewards.
    """
    def __init__(self):
        self.total_reward = 0.0
        self.history = []

    def reset(self):
        self.total_reward = 0.0
        self.history = []

    def add_reward(self, value: float, reason: str = ""):
        self.total_reward += value
        self.history.append({"value": value, "reason": reason})
        return value
