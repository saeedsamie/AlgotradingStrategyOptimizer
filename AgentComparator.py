import numpy as np
import matplotlib.pyplot as plt


class AgentComparator:
    """
    A class to compare the performance of multiple agents in the same environment.
    """

    def __init__(self, agents, environment):
        """
        Initialize the comparator with agents and the environment.

        Parameters:
        agents (list): List of agents to be compared.
        environment (TradingEnvironment): The environment where the agents will interact.
        """
        self.agents = agents
        self.environment = environment
        self.performance = {agent.__class__.__name__: [] for agent in
                            agents}  # Store performance metrics for each agent

    def run_simulation(self, episodes=100):
        """
        Run a simulation for each agent and collect performance metrics.

        Parameters:
        episodes (int): Number of episodes to run for each agent.
        """
        for agent in self.agents:
            total_rewards = []
            for e in range(episodes):
                state = self.environment.reset()
                state = np.reshape(state, [1, self.environment.num_state()])
                total_reward = 0
                done = False

                while not done:
                    action = agent.act(state)  # Get action from the agent
                    next_state, reward, done = self.environment.step(action)  # Step the environment
                    next_state = np.reshape(next_state, [1, self.environment.num_state()])

                    agent.remember(state, action, reward, next_state, done)  # Store the experience
                    agent.replay()  # Replay experiences and train

                    state = next_state  # Move to the next state
                    total_reward += reward

                total_rewards.append(total_reward)  # Track the total reward for this episode
                print(f"{agent.__class__.__name__} - Episode {e + 1}/{episodes}, Reward: {total_reward}")

            # Store the total rewards of this agent over all episodes
            self.performance[agent.__class__.__name__] = total_rewards

    def compare_rewards(self):
        """
        Compare the cumulative rewards of all agents over time.
        """
        plt.figure(figsize=(10, 6))
        for agent_name, rewards in self.performance.items():
            plt.plot(rewards, label=f"{agent_name} Reward")

        plt.title("Agent Performance Comparison")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.show()

    def average_rewards(self):
        """
        Print the average rewards of each agent over all episodes.
        """
        for agent_name, rewards in self.performance.items():
            avg_reward = np.mean(rewards)
            print(f"Average Reward for {agent_name}: {avg_reward}")
