# import the environment and the agent

from trading_environment import RL_trading_environment

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from collections import deque
from sklearn.model_selection import train_test_split

data = pd.read_csv("/Users/ariisichoix/Desktop/data_source/data_complet_5min.csv")

list_to_drop_fast = [
    "Timestamp",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "next_Open",
    "next_Close",
]

# train test split
data_train, data_test, = train_test_split(data, test_size=0.20, random_state=42, shuffle=False)

env_train = RL_trading_environment(
    observation=data_train,
    next_open_variable="next_Open",
    next_close_variable="next_Close",
    list_to_drop=list_to_drop_fast,
    initial_balance=100000,
    spread_param=0.0005,
    transaction_cost_param=0.00002,
)

env_test = RL_trading_environment(
    observation=data_test,
    next_open_variable="next_Open",
    next_close_variable="next_Close",
    list_to_drop=list_to_drop_fast,
    initial_balance=100000,
    spread_param=0.0005,
    transaction_cost_param=0.00002,
)


n_inputs = len(env_train.observation_state)
n_output = len(env_train.action_space)


model = keras.models.Sequential(
            [
                keras.layers.Dense(128, activation='relu', input_shape=[n_inputs]),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(n_output),
            ]
        )

env_train.observation_state

"""
To select an action with DQN, we select the largest predicted Q-Value. To be sure also that the agent explore the
environment, we will need an epsilon-policy. 

We code a function that take state and the epsilon value as input and ouput the action
"""


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(3)

    q_value = model.predict(state[np.newaxis])
    return np.argmax(q_value[0])


"""
Next step in Deep Q-Learning, we need to create a 'replay_buffer' so that we can store all experiences. 
In this replay_buffer, we will sample a random training batch at each training iteration
"""

replay_buffer = deque(maxlen=2000)

"""
Now  we need to code a function that will register a single experience. An experience is composed of 5 
elements : 
- the state
- the action
- the reward resulting
- the next state
- done -> used to know if the episode ended at that point
"""


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones


"""
We also need as function that will play a single step using epsilon greedy policy. then we put the result in the replay
buffer
"""


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done = env.step(action)
    # put the results of the action in the replay_buffer
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done


"""
Finally we can create a function that sample a batch of experiences from the replay buffer and train the DQN by
performing a Gradient Descent Step
"""

batch_size = 512
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=0.001)
loss_fn = keras.losses.mean_squared_error


def training_step(batch_size):
    # first thing we take the experience from the replay_buffer with the batch_size
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # after that we compute the estimated Q_values for the
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    # here we compute the target Q-value
    target_Q_values = rewards + (1 - dones) * discount_factor * max_next_Q_values

    # the DQN will also output the Q-value for all actions but we only want the one chosen by the agent
    mask = tf.one_hot(actions, n_output)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        # this part give us the Q_values only for the action chosen by the agent
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    # here we compute the loss
    grads = tape.gradient(loss, model.trainable_variables)
    # here we perform a gradient descent step to minimize the loss
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


"""
First we can not train because the replay_buffer is empty. So we need to fill the replay_buffer with 
some experiences before using it
"""

for episode in range(1):
    obs = env_train.observation_state
    done = False
    step = 0
    while not done:
        step += 1
        print("episode done ", episode, "step done ", env_train.index)
        epsilon = max(1 - step / 100000, 0.008)
        obs, reward, done = play_one_step(env_train, obs, epsilon)
        print(env_train.done)
        if step % 1000 == 0:
            training_step(batch_size)


for episode in range(1):
    obs = env_test.observation_state
    done = False
    step = 0
    while not done:
        step += 1
        print("episode done ", episode, "step done ", env_test.index)
        epsilon = 0
        obs, reward, done = play_one_step(env_test, obs, epsilon)
        print(env_test.done)
        # if step % 1000 == 0:
        #     training_step(batch_size)







model.get_weights()

model.summary()

env_train.number_of_transactions

env_train.number_of_long_position

env_train.number_of_short_position

env_train.cumulative_rewards




list_position = env.position_realized
list_actions = env.actions_realized

from collections import Counter

Counter(list_position).keys()  # equals to list(set(words))
Counter(list_position).values()  # counts the elements' frequency


# Save the weights
model.save_weights("./checkpoints/Warren_Buffet_AI")

# Create a new model instance
mode_copy = keras.models.Sequential(
    [
        keras.layers.Dense(32, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(n_output),
    ]
)
mode_copy.get_weights()


mode_copy.load_weights("./checkpoints/Warren_Buffet_AI")

mode_copy.get_weights()
