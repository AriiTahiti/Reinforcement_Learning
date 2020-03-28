from trading_environment import RLTradingEnvironment

import numpy as np

import tensorflow as tf
from tensorflow import keras

from collections import deque


class TradingAgent:

    def __init__(self, nb_features_available: int, nb_possible_action: int = 3, replay_buffer: int = 480,
                 batch_size: int = 460, discount_factor: int = 0.99):
        """
        Args:
            nb_features_available: this variable will represent the input_shape of the Deep Learning model. which
            is the number of feature we can observe in the environment.
            nb_possible_action: The number of possible action in the environment
            replay_buffer: it's an integer that represent the size of the replay buffer. Which is just list of
            experiences following the FIFO concept. we will sample a random training batch at each training iteration.
            batch_size: represents the batch size we are going to use to train the Deep Learning Model.
            discount_factor: it's the discounted factor of the future rewards (higher discount means that future
            rewards contribute more to the expected Q-value.
        """

        self.nb_features_available = nb_features_available
        self.nb_possible_action = nb_possible_action
        self.replay_buffer = deque(maxlen=replay_buffer)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = keras.optimizers.Adam(lr=0.0005)
        self.loss_fn = keras.losses.mean_squared_error
        self.model = self.agent_model()

    def agent_model(self):
        """
        Returns: This function return a keras model the agent will use to learn how to maximize reward in t
        """

        model = keras.models.Sequential(
            [
                keras.layers.Dense(128, activation="relu", input_shape=[self.nb_features_available]),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.nb_possible_action),
            ]
        )
        return model

    def epsilon_greedy_policy(self, observation, epsilon=0):
        """
        To choose an action with DQN, we select the largest predicted Q-Value. To be sure also that the agent explore
        the environment, we will need an epsilon-policy.

        Args:
            observation: represent the current observation the agent is seeing and well compute the best action
            based on the model trained
            epsilon: is the probability to take a random action to be able to explore the possibilities

        Returns: This function output the action taken by the agent

        """

        if np.random.rand() < epsilon:
            return np.random.randint(3)

        q_value = self.agent_model().predict(observation[np.newaxis])
        return np.argmax(q_value[0])

    def sample_experiences(self):
        """
        This function is used to sample experiences in the replay buffer. The sample size is equal to the batch size,
        which represent the number of experiences we are going to use in other to feed the Deep Learning Model.

        Returns: The function return all the elements that composed the sample of experiences :
            - the observations
            - the actions
            - the rewards resulting
            - the next observations
            - done

        comments: the function np.random.randint allows you to select multiple times the same observation.

        """
        # select random indices
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        # create the batch of experiences based on the random indices selected
        batch = [self.replay_buffer[index] for index in indices]
        # for each experience, extract the 5 elements
        observations, actions, rewards, next_observations, done = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return observations, actions, rewards, next_observations, done

    def play_one_step_action_augmentation(self, env: RLTradingEnvironment, observation, epsilon):

        """
        The function will play a single step using epsilon greedy policy and we will put the result
        in the replay buffer.

        Args:
            env: the environment class RLTradingEnvironment where
            observation: is the current observation the agent can see in the environment
            epsilon: is the probability to take a random action to be able to explore the possibilities

        Returns: this function returns the next observation, the reward associated to the action and the boolean that
        indicates if we when through all the observations in the time series

        """

        action = self.epsilon_greedy_policy(observation, epsilon)
        # we can add to the replay buffer the results given by the other actions {action augmentation loss}

        action_augmented = [0, 1, 2]
        action_augmented.remove(action)

        next_observation, reward, done = env.step(action)

        # put the action augmentation into the replay_buffer
        # replay_buffer.append((state, action_augmented[0], reward, next_state, done))
        # replay_buffer.append((state, action_augmented[1], reward, next_state, done))

        # put the results of the action in the replay_buffer
        self.replay_buffer.append((observation, action, reward, next_observation, done))

        return next_observation, reward, done

    def training_step(self):

        """
        This function will be used to train the model

        Returns:

        """
        # first thing we take the experience from the replay_buffer with the batch_size
        observations, actions, rewards, next_observations, done = self.sample_experiences()
        # after that we compute the estimated Q_values for the
        next_q_values = self.model.predict(next_observations)
        max_next_q_values = np.max(next_q_values, axis=1)
        # here we compute the target Q-value
        target_q_values = rewards + (1 - done) * self.discount_factor * max_next_q_values

        # the DQN will also output the Q-value for all actions but we only want the one chosen by the agent
        mask = tf.one_hot(actions, self.nb_possible_action)
        with tf.GradientTape() as tape:
            all_q_values = self.model(next_observations)
            # this part give us the Q_values only for the action chosen by the agent
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_q_values, q_values))
        # here we compute the loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        # here we perform a gradient descent step to minimize the loss
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))





