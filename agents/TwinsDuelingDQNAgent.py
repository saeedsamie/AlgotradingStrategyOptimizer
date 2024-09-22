from agents.DuelingDQNAgent import DuelingDQNAgent


class TwinsDuelingDQNAgent(DuelingDQNAgent):
    """
    Twins Dueling DQN agent that inherits from DuelingDQNAgent.
    Can include twin-specific functionality.
    """
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, memory_size=2000):
        super().__init__(state_size, action_size, gamma, epsilon, epsilon_min, epsilon_decay, lr, batch_size, memory_size)
        # Further initialization specific to twins if necessary
