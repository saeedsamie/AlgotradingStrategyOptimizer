import numpy as np
import tensorflow as tf
from collections import deque


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent implementation.
    """

    def __init__(self, state_size, action_size, gamma=0.99, tau=0.001, actor_lr=0.001, critic_lr=0.002,
                 memory_size=2000, batch_size=64):
        """
        Initialize the DDPG agent.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Initialize actor and critic models
        self.actor_model = self.build_actor_model()
        self.target_actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()
        self.target_critic_model = self.build_critic_model()

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Soft update of target models
        self.update_target_models()

    def build_actor_model(self):
        """
        Build the actor model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='tanh'))
        return model

    def build_critic_model(self):
        """
        Build the critic model.
        """
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        concat = tf.keras.layers.Concatenate()([state_input, action_input])

        dense_1 = tf.keras.layers.Dense(24, activation='relu')(concat)
        dense_2 = tf.keras.layers.Dense(24, activation='relu')(dense_1)
        output = tf.keras.layers.Dense(1)(dense_2)

        model = tf.keras.models.Model(inputs=[state_input, action_input], outputs=output)
        return model

    def update_target_models(self):
        """
        Soft update target models.
        """
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Use the actor model to determine the action for a given state.
        """
        state = np.reshape(state, [1, self.state_size])
        return self.actor_model.predict(state)[0]

    def train(self):
        """
        Train the actor and critic networks using a random batch from memory.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.array(random.sample(self.memory, self.batch_size))

        states, actions, rewards, next_states, dones = map(np.vstack, zip(*minibatch))
        target_actions = self.target_actor_model.predict(next_states)
        future_qs = self.target_critic_model.predict([next_states, target_actions])

        targets = rewards + self.gamma * future_qs * (1 - dones)

        self.critic_model.train_on_batch([states, actions], targets)

        with tf.GradientTape() as tape:
            pred_actions = self.actor_model(states)
            critic_value = self.critic_model([states, pred_actions])
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

    def soft_update(self):
        """
        Soft update target networks based on the main networks' weights.
        """
        new_actor_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(self.actor_model.get_weights(), self.target_actor_model.get_weights())
        ]
        self.target_actor_model.set_weights(new_actor_weights)

        new_critic_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in
            zip(self.critic_model.get_weights(), self.target_critic_model.get_weights())
        ]
        self.target_critic_model.set_weights(new_critic_weights)
