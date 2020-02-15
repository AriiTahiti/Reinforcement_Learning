import numpy as np

"""
Deep Reinforcement Learning Chapter
"""

import gym

env = gym.make("CartPole-v1")

# the reset gives you the original state of the game
state = env.reset()

# set an action
# action = 1

# give you how the game looks like
# env.render()

# code a simple policy
def basic_policy(state):
    angle = state[0]
    return 0 if angle < 0 else 1


# create a list that will contain all the results for all the episodes we ask to run
totals = []


# episode represent the number of time we play the game

# an episode is a
for episodes in range(500):

    episode_rewards = 0
    # initialize the game
    state = env.reset()
    for step in range(200):
        action = basic_policy(state)
        state, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

"""
if we check the results given by all the episodes that 
"""

# Do a simple Deep Learning model to try to evaluate the best action possibl

import tensorflow as tf
from tensorflow import keras

n_inputs = env.observation_space.shape[0]

model_mind = keras.models.Sequential(
    [
        keras.layers.Dense(4, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

"""
The following function will be used to play a single step for an episode
"""


def play_one_step(env, state, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(state[np.newaxis])
        action = tf.random.uniform([1, 1]) > left_proba
        y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    state, reward, done, info = env.step(int(action[0, 0].numpy()))
    return state, reward, done, grads


"""
Now we need to create a function that will play multiple episodes
"""


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        state = env.reset()
        for step in range(n_max_steps):
            state, reward, done, grads = play_one_step(env, state, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


"""
We need a function that will discount the future rewards
"""


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


"""
one important thing to do is the normalize the all the discounted rewards
"""


def discounted_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [
        discount_rewards(rewards, discount_factor) for rewards in all_rewards
    ]

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [
        (discounted_rewards - reward_mean) / reward_std
        for discounted_rewards in all_discounted_rewards
    ]


# verification
discount_rewards([10, 0, -50, 6, 9, -10], discount_factor=0.8)
discounted_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_factor=0.8)

"""
Now lets define the hyper parameters
"""

n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95

"""
We also need an optimizer for the model and a loss function
"""

optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy

"""
finally we can train the training loop
"""

for iteration in range(n_iterations):

    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model_mind, loss_fn
    )

    all_final_rewards = discounted_and_normalize_rewards(all_rewards, discount_factor)

    all_mean_grads = []

    for var_index in range(len(model_mind.trainable_variables)):

        mean_grads = tf.reduce_mean(
            [
                final_reward * all_grads[episode_index][step][var_index]
                for episode_index, final_reward in enumerate(all_final_rewards)
                for step, final_reward in enumerate(final_reward)
            ],
            axis=0,
        )

        all_mean_grads.append(mean_grads)

    optimizer.apply_gradients(zip(all_mean_grads, model_mind.trainable_variables))


"""
Now that we tried a Policy Gradient method to optimize the decision making, we will implement a better algorithm called
Deep Q-Learning algorithm.

To keep it short, this algorithm try to estimate the expected value for each state
"""

"""
We first need an Neural Network that takes a state and outputs one approximate Q-Value for each possible action
"""

from RL_code.trading_environment import RL_trading_environment

env = gym.make("CartPole-v0")

n_inputs = env.observation_space.shape[0]
n_output = env.action_space.n

model = keras.models.Sequential(
    [
        keras.layers.Dense(32, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(n_output),
    ]
)


"""
To select an action with DQN, we select the largest predicted Q-Value. To be sure also that the agent explore the
environment, we will need an epsilon-policy. 

We code a function that take state and the epsilon value as input and ouput the action
"""


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_value = model.predict(state[np.newaxis])
        return np.argmax(Q_value[0])


"""
Next step in Deep Q-Learning, we need to create a 'replay_buffer' so that we can store all experiences. 
In this replay_buffer, we will sample a random training batch at each training iteration
"""

from collections import deque

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
    next_state, reward, done, info = env.step(action)
    # put the results of the action in the replay_buffer
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


"""
Finally we can create a function that sample a batch of experiences from the replay buffer and train the DQN by
performing a Gradient Descent Step
"""

batch_size = 32
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

for episode in range(800):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        training_step(batch_size)

"""
At the end of this part, we have a DQL algorithm that works fine but is not able to perform very well for 
a long period of time. This is due to the problem of catastrophic forgetting.

to stabilize the model, we will need to implement different tweak in the algorithm
"""

"""
First tweak is : Fixed Q-Value Targets

So we have 2 DQNs : 
- the first one is used to take the action and learn in an online mode
- the second is only updated when a certain number of episode is run. he is only used to estimate the target Q-Value

The target model is just a clone of the online model
"""

# we create the same structure for target_model than model
target_model = keras.models.clone_model(model)

# we copy the parameter weights of the model to the target model
target_model.set_weights(model.get_weights())


def training_step_fixed_Q_value(batch_size):
    # first thing we take the experience from the replay_buffer with the batch_size
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # after that we compute the estimated Q_values for the
    next_Q_values = target_model.predict(next_states)
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


for episode in range(800):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode % 50 == 0:
        target_model.set_weights(model.get_weights())
    if episode > 50:
        training_step_fixed_Q_value(batch_size)

"""
Double DQN. we use another online model to estimate the target Q_value
"""


def training_step_double_DQN(batch_size):
    # first thing we take the experience from the replay_buffer with the batch_size
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_output).numpy()

    next_best_Q_values = (target_model.predict(next_states) * next_mask).sum(axis=1)

    # here we compute the target Q-value
    target_Q_values = rewards + (1 - dones) * discount_factor * next_best_Q_values

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
Importance Sampling (IS) or Prioritized Experience Replay (PER)

Need to find a way to implement it in our training loop. 
"""


"""
Dueling DQN
"""


K = keras.backend
input_states = keras.layers.Intput(Shape=[4])
hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
state_values = keras.layers.Dense(1)(hidden2)
raw_advantages = keras.layers.Dense(n_output)(hidden2)
advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
Q_values = state_values + advantages
model = keras.Model(inputs=[input_states], oututs=Q_values)


def training_step_Dueling_DQN(batch_size):
    # first thing we take the experience from the replay_buffer with the batch_size
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_output).numpy()

    next_best_Q_values = (target_model.predict(next_states) * next_mask).sum(axis=1)

    # here we compute the target Q-value
    target_Q_values = rewards + (1 - dones) * discount_factor * next_best_Q_values

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
What is tf-agent? Exploration of tf-agent library
"""

from tf_agents.environments import suite_gym

env = suite_gym.load("Breakout-v4")

from tf_agents.environments.wrappers import ActionRepeat

repeating_env = ActionRepeat(env, times=4)

from gym.wrappers import TimeLimit

"""
Create an environment with the specific parameter that you want directly during the import
"""
limited_repeating_env = suite_gym.load(
    "Breakout-v4",
    gym_env_wrappers=[lambda env: TimeLimit(env, max_episode_steps=10000)],
    env_wrappers=[lambda env: ActionRepeat(env, times=4)],
)


from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4


max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"


env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4],
)


from tf_agents.environments.tf_py_environment import TFPyEnvironment

tf_env = TFPyEnvironment(env)


eval_env = TFPyEnvironment(env)

"""
Create the Deep Q Network with tf_agent
"""

from tf_agents.networks.q_network import QNetwork


preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.0)

conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
)

"""
Creating the DQN Agent 
"""

from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0)
update_period = 4
optimizer = keras.optimizers.RMSprop(
    lr=0.00025, rho=0.95, momentum=0.0, epsilon=0.0001, centered=True
)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=25000 // update_period,
    end_learning_rate=0.01,
)

agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000,
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.99,
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step),
)
agent.initialize()

"""
creating a replay buffer with tf_agent
"""

from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=tf_env.batch_size, max_length=1000000
)

replay_buffer_observer = replay_buffer.add_batch


"""
We can create a class that will help us to show the progress
"""


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


"""
Creating the replay buffer and the Corresponding Observer
"""

from tf_agents.metrics import tf_metrics


train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]


"""
We can check the result of those values at any time
"""

from tf_agents.eval.metric_utils import log_metrics

# import logging
# logging.get_logger().set_level(logging.INFO)
log_metrics(train_metrics)

"""
Creating the Collect Driver
"""

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver


collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period,
)

"""
create a loop to chos the progress of the agent
"""

from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000,
)

final_time_step, final_policy_state = init_driver.run()

"""
Dataset Creation
"""

trajectories, buffer_info = replay_buffer.get_next(sample_batch_size=2, num_steps=3)

from tf_agents.trajectories.trajectory import to_transition

time_steps, action_steps, next_time_steps = to_transition(trajectories)


dataset = replay_buffer.as_dataset(
    sample_batch_size=64, num_steps=2, num_parallel_calls=3
).prefetch(3)


"""
Final Training loop using tf-agents
"""

from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)


"""
Train the agent
"""

train_agent(100000)

"""
visualize the agent playing
"""

import IPython
import base64
import imageio


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>""".format(
        b64.decode()
    )

    return IPython.display.HTML(tag)


num_episodes = 3
video_filename = "imageio.mp4"

with imageio.get_writer(video_filename) as video:
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        video.append_data(env.render())
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(env.render())

embed_mp4(video_filename)
