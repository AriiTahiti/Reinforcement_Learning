import tensorflow
from tensorflow import keras
from collections import deque
import random
import numpy as np


class warren_buffet:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # create a replay buffer that store all experiences and sample a random training bach from it
        self.memory = deque(maxlen=2000)

        # discount rate
        self.gamma = 0.95

        # exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    """
    Deep Learning model that take state observations as input and output a 
    """

    # model has to be fit using the state given by the environment and the reward_value
    # model.fit(state, reward_value, epochs=1, verbose=0)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.models.Sequential(
            [
                keras.layers.Dense(32, activation="relu", input_shape=[50]),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(3),
            ]
        )
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
