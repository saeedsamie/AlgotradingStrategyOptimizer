import numpy as np

from AgentComparator import AgentComparator
from Env import Env
from agents.DDPGAgent import DDPGAgent
from agents.DQNAgent import DQNAgent
from agents.DuelingDQNAgent import DuelingDQNAgent
from agents.TwinsDuelingDQNAgent import TwinsDuelingDQNAgent

if __name__ == "__main__":
    # Initialize the environment
    file_path = "rsi_final_merged_dataset.csv"
    dataset = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    env = Env(dataset, test_ratio=0.2)

    # Initialize the agents
    dqn_agent = DQNAgent(state_size=env.train.shape[1] - 2, action_size=2)
    ddpg_agent = DDPGAgent(state_size=env.train.shape[1] - 2, action_size=2)
    dueling_dqn_agent = DuelingDQNAgent(state_size=env.train.shape[1] - 2, action_size=2)
    twins_dueling_dqn_agent = TwinsDuelingDQNAgent(state_size=env.train.shape[1] - 2, action_size=2)

    # Compare agents using AgentComparator
    comparator = AgentComparator(agents=[dqn_agent, ddpg_agent, dueling_dqn_agent, twins_dueling_dqn_agent],
                                 environment=env)
    comparator.run_simulation(episodes=10)

    # Compare rewards
    comparator.compare_rewards()

    # Print average rewards
    comparator.average_rewards()
