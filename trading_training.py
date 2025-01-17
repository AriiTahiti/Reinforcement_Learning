from trading_environment import RLTradingEnvironment
from trading_agent import TradingAgent
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# import data that will be used to build the environment
data = pd.read_csv("/Users/ariisichoix/Desktop/data_source/data_complet_1h.csv")

# Create the list of variables that will be dropped
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

# train test split data
data_train, data_test, = train_test_split(
    data, test_size=0.20, random_state=42, shuffle=False
)

# reset the index for the test data
data_test = data_test.reset_index(drop=True)

# create the training environment
env_train = RLTradingEnvironment(
    observation=data_train,
    next_open_variable="next_Open",
    next_close_variable="next_Close",
    list_to_drop=list_to_drop_fast,
    initial_balance=10000,
    trade_size=10000,
    spread_param=0.0005,
    transaction_cost=0.00002
)

# create your agent
agent = TradingAgent(
    nb_features_available=len(env_train.observation_state),
    nb_possible_action=len(env_train.action_space),
    replay_buffer=480,
    batch_size=460,
    discount_factor=0.99
)

# Training Loop of the agent
for episode in range(1):
    obs = env_train.observation_state
    done = False
    step = 0
    while not done:
        step += 1
        print("episode done ", episode, "step done ", env_train.index)
        epsilon = max(0.9 - episode / 1000, 0.00001)
        obs, reward, done = agent.play_one_step_action_augmentation(env_train, obs, epsilon)
        print(env_train.done)
        if step % 460 == 0:
            agent.training_step()


# create test environment
env_test = RLTradingEnvironment(
    observation=data_test,
    next_open_variable="next_Open",
    next_close_variable="next_Close",
    list_to_drop=list_to_drop_fast,
    initial_balance=10000,
    trade_size=10000,
    spread_param=0.0005,
    transaction_cost=0.00002
)

# Testing Loop
for episode in range(1):
    obs = env_test.observation_state
    done = False
    step = 0
    while not done:
        step += 1
        print("episode done ", episode, "step done ", env_test.index)
        epsilon = 0
        obs, reward, done = agent.play_one_step_action_augmentation(env_train, obs, epsilon)
        print(env_test.done)


cumulative_rewards = np.array(env_train.all_step_rewards)
cumulative_rewards.min()
cumulative_rewards.mean()

cumulative_rewards = np.array(env_train.all_step_rewards).cumsum()

"""
Create some plots 
"""

import matplotlib.pyplot as plt

time = np.arange(0.0, 17000.0, 1.0)

fig, ax = plt.subplots()
ax.plot(time, cumulative_rewards)

ax.set(xlabel='time (s)', ylabel='reward',
       title='plot AI rewards')
ax.grid()
plt.show()

