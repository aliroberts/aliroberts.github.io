---
layout: post
title: Deep Q-Learning with PyTorch - Part 1
---

This is the first part of a series of three posts exploring deep Q-learning (DQN) which is a fundamental reinforcement learning algorithm. This first part will walk through a basic Python implementation of DQN to solve the cart-pole problem, using the PyTorch library. This initial implementation is based on the algorithm as described in Deepmind's paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/pdf/1312.5602.pdf). Over the next two parts we'll ramp up the level of sophistication and end with a DQN implementation for the Atari game [Breakout](<https://en.wikipedia.org/wiki/Breakout_(video_game)>), that closely follows the details of DeepMind's paper ["Human-level control through deep reinforcement learning"](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) which appeared in [Nature](http://www.nature.com/) in 2015.

This post is intended to be sort of a compliment to OpenAI's ["Spinning Up"](https://spinningup.openai.com/en/latest/user/introduction.html) which is an excellent resource if you've got some familiarity with matrix operations, calculus and programming and want to get started with deep reinforcement learning. If a lot of the terminology here is new to you, I highly recommend reading through [part one](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html), [part two](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) and [part three](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) of their introduction to RL.

## The problem

The environment that we'll tackle in this first post is known as cart-pole. In particular, we'll be using OpenAI's implementation (`CartPole-v0`) from their Python library `gym`:

**TODO** Add GIF of CartPole

> _A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center._  
>  _[https://gym.openai.com/envs/CartPole-v0/](https://gym.openai.com/envs/CartPole-v0/)_

The environment is considered 'solved' when the average return (in this case the sum of rewards from each timestep) over 100 consecutive trials (episodes) for an agent is greater than or equal to 195.

<!-- Let's take a moment to frame this problem more formally. Our cart-pole environment can be described by a Markov Decision Process (MDP) which is defined as a 5-tuple $\langle S, A, R, P, \psi_0 \rangle$ where

- $S$ is the set of possible states
- $A$ is the set of possible actions
- $R : S \times A \times S \to \mathbb{R}$ is the reward function with $r_t = R(s_t, a_t, s_{t+1})$
- $P: S \times A \to \mathcal{P}(S)$ is the transition probability function where $P(s'|s, a)$ gives the probability of transitioning to  -->

### States

Let's take a moment to frame this problem more formally. Let's first consider the **state space** $S$ (I will use the terms 'states' and 'observations' interchangably). The state space $S$ is a set whose elements describe all possible configurations of the environment. In part two we'll modify our approach to use the rendered images of the environment to derive our observations. For now, however, we'll focus on simply using the observations provided out-of-the-box by the `gym` library which essentially take the form of a 4-dimensional NumPy array of continuous-valued real numbers which have been wrapped in a custom `Box` class that gives us a few helper methods:

```python
>>> import gym
>>> env = gym.make('CartPole-v0') # Instantiate the environment
>>> env.observation_space # Data structure used to hold env states
Box(4,)
>>> env.observation_space.sample() # Randomly sample a state
array([-0.01381209, -0.00833267, -0.04027887, -0.00729395])
```

These observation arrays have the following structure:

```
Index   Observation               Min                 Max
0       Cart Position             -4.8                4.8
1       Cart Velocity             -Inf                Inf
2       Pole Angle                -0.418 rad          0.418 rad
3       Pole Angular Velocity     -Inf                Inf
```

### Actions

According to the description of the cart-pole environment above, the **action space** $A$ is discrete and contains two possible values.

```python
>>> env.action_space
Discrete(2,)
>>> env.action_space.n # The size of the action space
2
>>> env.action_space.sample() # Randomly sample an action
1
>>> env.action_space.sample()
0
```

Note that $A = \\{0, 1\\}$ with 0 corresponding to a left action and 1 corresponding to a right action rather than -1 and 1 as in the description above.

### Rewards

The cart-pole environment returns a reward of 1 for each action that results in the pole staying upright and the cart staying within a certain distance from the centre.

```python
>>> env.reset() # Initialise an episode state
array([0.01337562, 0.02386517, 0.04556581, 0.01944452])
>>> observation, reward, done, info = env.step(1) # Perform action 1 (move right)
>>> reward
1
>>> done # Has the episode terminated?
False
```

For more details about `gym` environments/installation the [official docs](https://gym.openai.com/docs/) are a quick read.

### The optimal action-value function $Q^*$

For a given envronment, the goal of RL is to choose a **policy**, that is, a function of the current state that selects the next action, that maximises the **expected return** of the agent that acts according to it.

- Introduce formally what a return function is and what it is in the context of cart-pole
- Discuss the action-value function mathematically

Suppose for a moment that we had access to the true function $Q^* $ for a given environment with a finite set of actions, how would we use it? For a given state $s$ we could enumerate all the possible actions $(a_1, a_2, ..., a_n)$, calculate $Q^* (s, a_i)$ and simply adopt a policy of choosing the action that maximises the expected return for the given state, setting $a = \max _a Q^* (s, a)$. The goal of **Q-learning** is to approximate the function $Q^*$ and adopt such a policy.

Note here that this depends on calculating Q for every action and therefore restricts this method to choosing actions for environments with discrete action spaces.

## Approximating the $Q^*$ function

- Use a FFNN to approximate Q
- What to learn

> **NOTE**: It is worth mentioning that DQN can be adapted for environments with continuous action spaces, where it goes by the name [Deep Deterministic Policy Gradient (DDPG)](https://spinningup.openai.com/en/latest/algorithms/ddpg.html). The principal modification here is the introduction of a function that chooses an action that maximises Q which is learned as part of the training process.

- Mention briefly that the algorithm can be modified to handle continuous action spaces (DDQN: https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

- Focus will be on environments with discrete action spaces since we have to search for the best action

### The complete algorithm

Our implementation will closely follow the DQN algorithm as described in Vlad Mnih et al.'s 2013 paper ["Playing Atari with Deep Reinforcement Learning"](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning). Although the authors focused on a high-dimensional observation space which consisted of processed images of the Atari games, we'll work our way up to this environment over these three posts and stick for now to the cart-pole environment and the 4-D observations that OpenAI's `gym` implementation provides us with.

![Playing Atari with Deep Reinforcement Learning, Vlad Mnih, Koray Kavukcuoglu, et al. NIPS 2013](/assets/dqn-1/mnih-algo-2013.png)

Note that the implementation described here will deviate from this in a few minor details. Firstly, rather than iterating over a fixed number of episodes and timesteps for each episode, we'll simply iterate over a fixed number of timesteps. We'll update the weights of our $Q$ network and reset the environment when each episode reaches a terminal state or exit when the environment is 'solved' (when we have reached an average return of > 195 over 100 consecutive episodes).

Secondly, the authors of the paper perform some preprocessing over a sequence of images to form a fixed-size representation of their environment's states using some function $\phi_t = \phi(s_t)$. We will use the states $s_t$ from the cart-pole environment as-is for now (we can ignore the setting of $s_t = s_t, a_t, x_\{t+1}$ too; these details will be addressed in the next post).

### Implementation

A complete implementation is given below. We first define a few helper functions for generating our Q network (`ffnn`, which I pinched from the 'Simplest Policy Gradient' implementation in [part three](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) of 'Spinning Up'), choosing the optimal action according to our Q network (`best_action`) and computing the loss (`compute_loss`). The rest of the code more-or-less devoted to implementing the `dqn` function that takes care of the complete DQN training process as described above.

The algorithm uses an $\epsilon$-greedy policy which means that it'll randomly sample an action from the action space with probability $\epsilon$ or otherwise choose the best action from the current approximation of $Q^* $ (with probability 1 - $\epsilon$). The value of $\epsilon$ essentially controls the degree to which the agent will explore the environment vs the degree to which it will exploit its current ability to maximise its expected return. When we start training our agent, our $Q$ network is unlikely to provide a reasonable approximation of $Q^* $. In order to discover which actions lead to a higher accumulation of rewards over time we opt for an initial period of data collection via random actions. Hopefully some of these random actions will have resulted in a high return over some of the episodes and so training our network will nudge the weights of our $Q$ network to predict a higher value (expected return) under those circumstances if we follow the high-return actions.

After the initial exploration period, the implementation below begins to linearly decay the value of $\epsilon$ so that more and more of the agent's actions are those which the $Q$ network predicts will maximise our reward. While $Q$ it may still not be a particularly good approximation of $Q^* $, it is hoped that it provides some reasonable estimates under some circumstances and then by mixing in some further random exploration we will gradually nudge the weights to push the average episode return up even more.

Most of the hyperparameters (learning rate, network size, timesteps and batchsize) were taken from the DQN implementation in the [Stable Baselines](https://github.com/hill-a/stable-baselines) repo and seem to work well over a range of random seeds (see below for experimental results).

```python
import argparse
import random

from collections import deque

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


def ffnn(sizes, activation=nn.Tanh, output_activaton=nn.Identity):
    # Build a feedforward neural network
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activaton
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def best_action(q_func, obs, n_acts):
    # Create n inputs for our Q net and choose the best action
    # Use one-hot encoding for actions and concatenate the observation:
    # obs = [o_1, o_2,..., o_k], n_acts = 2 give us two candidate inputs
    # for the Q function:

    # [1, 0, o_1, o_2,..., o_k], [0, 1, o_1, o_2,..., o_k]
    with torch.no_grad():
        one_hot = torch.eye(n_acts)
        expanded_obs = torch.as_tensor([obs],
                                       dtype=torch.float32).expand(n_acts, -1)
        q_in = torch.cat((one_hot, expanded_obs), 1)
        best_act = torch.argmax(q_func(q_in)).item()
    return best_act


def compute_loss(q_func, x, y):
    return ((y - q_func(x))**2).mean()


def dqn(
        timesteps=100000,
        bs=32,
        hidden=[64, 64],
        replay_buffer_len=100000,
        lr=1e-3,
        epsilon_start=1,
        explore_until=10000,  # Choose a random action for this many timesteps
        gamma=1,  # Discount factor when computing returns from rewards
        seed=42,
        render=False):

    # Instantiate environment
    env = gym.make('CartPole-v0')

    # Set random seeds
    env.seed(seed)
    torch.manual_seed(seed)

    # Construct our Q network that we'll use to approximate Q*
    n_acts = env.action_space.n
    obs_size = env.observation_space.shape[0]
    q_net = ffnn([n_acts + obs_size] + hidden + [1])

    # Use Adam for our optimizer
    optimiser = torch.optim.Adam(q_net.parameters(), lr=lr)

    def epsilon_schedule(t, timesteps):
        # We'll use a linear decay function here
        return epsilon_start * min((-t / timesteps + 1), 1.0)

    # Initialise the experience replay buffer.
    # When the replay buffer reaches the max length discard oldest items
    # so we don't exceed the max length
    xp_replay = deque([], maxlen=replay_buffer_len)

    # Create some empty buffers for tracking episode-level details.
    # These will be used for logging and determining when we've finished.
    ep_returns = []
    ep_lens = []  # The number of timesteps per episode

    rews = []  # Track rewards for each timestep and reset at end of episode
    done = False

    obs = env.reset()

    loss = None

    for i in range(timesteps):
        # Sample an action randomly with prob (1 - epsilon)
        eps = epsilon_schedule(i - explore_until, timesteps)
        act = env.action_space.sample(
        ) if random.random() < eps else best_action(q_net, obs, n_acts)

        # Perform the action in the envrionment and record
        # an experience tuple
        obs_next, rew, ep_done, _ = env.step(act)
        xp = (obs.copy(), act, rew, obs_next.copy(), ep_done)

        # Record a reward of 0 for the terminal state
        if ep_done:
            rew = 0

        rews.append(rew)
        xp_replay.append(xp)

        # Make sure we update the current observation for the next iteration!
        obs = obs_next

        targets = []
        inputs = []

        if ep_done or i == timesteps - 1:
            # Cartpole is considered solved when the average return is >= 195
            # over 100 consecutive trials
            ep_returns.append(sum(rews))
            if len(ep_returns) >= 100 and np.mean(ep_returns[-100:]) >= 195:
                print(f'SOLVED! timesteps: {i} \t episodes: {len(ep_returns)}')
                return

            # Randomly sample a minibatch from our experience replay buffer
            minibatch = random.sample(xp_replay, min(bs, len(xp_replay)))

            # Construct the targets and inputs for our loss function
            for obs, act, rew, obs_next, done in minibatch:
                # Find the best action (predicted by the current q_net)
                # and construct our input vals
                best_act = best_action(q_net, obs_next, n_acts)

                best_act_one_hot = np.zeros(n_acts)
                best_act_one_hot[best_act] = 1

                q_input = np.concatenate((best_act_one_hot, obs_next))

                with torch.no_grad():
                    y_i = rew if done else rew + gamma * q_net(
                        torch.as_tensor(q_input, dtype=torch.float32)).item()
                targets.append([y_i])

                # One-hot encoding of action
                act_one_hot = np.zeros(n_acts)
                act_one_hot[act] = 1
                inputs.append(np.concatenate((act_one_hot, obs)))

            optimiser.zero_grad()
            loss = compute_loss(q_net,
                                torch.as_tensor(inputs, dtype=torch.float32),
                                torch.as_tensor(targets, dtype=torch.float32))
            loss.backward()
            optimiser.step()

            ep_returns.append(sum(rews))
            ep_lens.append(len(rews))

            rews = []

            obs = env.reset()

        if i > 0 and i % 500 == 0:
            print(
                f'timestep: {i} \t epsilon: {eps}' \
                    f'\t last loss: {loss}' \
                        f'\t return (last 100): {np.mean(ep_returns[-100:])}'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--replay-buffer-len', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon_start', type=int, default=1)
    parser.add_argument('--explore_until', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render', type=bool, default=True)
    args = parser.parse_args()

    dqn(
        timesteps=args.timesteps,
        bs=args.bs,
        replay_buffer_len=args.replay_buffer_len,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        explore_until=args.explore_until,
        render=args.render,
    )

```

- Generating a FFNN
- The main training loop

### Results

- A note about choosing hyperparameters (these were borrowed from the stable-baselines repo)
- pyplot graphs
- Multiple random seeds
- Discussion about the warm-up period that's usually associated with Q-learning
- Stability
