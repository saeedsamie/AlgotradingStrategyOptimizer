import tensorflow as tf

from agents.DQNAgent import DQNAgent


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent with separate value and advantage streams.
    Inherits from DQNAgent but modifies the network architecture.
    """

    def build_model(self):
        """
        Build the dueling architecture model with separate streams.
        """
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        dense_1 = tf.keras.layers.Dense(24, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(24, activation='relu')(dense_1)

        # Value stream
        value_fc = tf.keras.layers.Dense(24, activation='relu')(dense_2)
        value = tf.keras.layers.Dense(1)(value_fc)

        # Advantage stream
        advantage_fc = tf.keras.layers.Dense(24, activation='relu')(dense_2)
        advantage = tf.keras.layers.Dense(self.action_size)(advantage_fc)

        # Combine value and advantage into the final Q-values
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

        model = tf.keras.models.Model(inputs=state_input, outputs=q_values)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
