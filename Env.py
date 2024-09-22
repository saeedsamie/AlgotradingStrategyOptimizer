import numpy as np


class Env:
    """
    Environment class for trading simulation.
    """

    def __init__(self, dataset, test_ratio):
        """
        Initializes the environment with training and testing data.

        Parameters:
        dataset (np.array): The dataset to use for the environment.
        test_ratio (float): The ratio of the dataset to use for testing.
        """
        np.random.shuffle(dataset)
        self.datas = dataset
        self.train = self.datas[:int(len(self.datas) * test_ratio), :]  # Training data
        self.test = self.datas[int(len(self.datas) * test_ratio):, :]  # Test data
        self.counter = -1  # Counter for the current step
        self.counter_reward = 0  # Accumulates rewards

    def calculate_reward(self, action):
        """
        Calculates the reward based on the agent's action.

        Parameters:
        action (int): The action taken by the agent.

        Returns:
        reward (int): The reward based on whether the action was correct.
        """
        if self.counter == -1:
            return 0
        else:
            if self.train[self.counter, -1] == action:
                self.counter_reward += 2
                return 2  # Reward for correct action
            else:
                return -1  # Penalty for incorrect action

    def step(self, action):
        """
        Takes a step in the environment.

        Parameters:
        action (int): The action taken by the agent (0 or 1).

        Returns:
        state (np.array): The next state.
        reward (int): The reward obtained from the action.
        done (bool): Whether the episode is finished.
        """
        if action < 0:
            action = 0.0  # Handle invalid actions
        else:
            action = 1.0

        reward = self.calculate_reward(action)
        self.counter = np.random.randint(low=0, high=len(self.train) - 1, size=1)[0]  # Random next state
        done = False

        if self.counter_reward > len(self.train):
            done = True
        if self.counter_reward < -1000:
            done = True

        state = self.train[self.counter, :-2]  # The next state (excluding the label)
        state = state.reshape(1, state.shape[0])  # Reshape state to a 2D array
        return state, reward, done

    def reset(self):
        """
        Resets the environment for a new episode.

        Returns:
        state (np.array): The initial state.
        """
        self.counter = -1
        self.counter_reward = 0
        return self.train[0, :-2].reshape(1, self.train.shape[1] - 2)

    def render(self):
        """
        Switches to testing mode by replacing the training data with test data.
        """
        self.train = self.test
        self.counter = -1
        self.counter_reward = 0

    def num_state(self):
        """
        Returns the number of features (excluding labels) in the environment.

        Returns:
        int: The number of state features.
        """
        return self.train.shape[1] - 2
