"""
File of functions we used in lab. Called 'utils', since I couldn't think of anything better and didn't want them in
main script.
"""

import torch
from torch import optim
import numpy as np

# Smoothing function for nicer plots
def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    # YOUR CODE HERE
    done = False
    state = env.reset()
    while not done:
        # Get action using policy.
        action = policy.sample_action(torch.Tensor(state))  # .item()
        next_state, reward, done, _ = env.step(action)
        # Append to lists
        states.append(state), actions.append(action), rewards.append(reward), dones.append(done)
        # Update to next state.
        state = next_state

    states, actions, rewards = torch.Tensor(states), \
                               torch.LongTensor(actions).unsqueeze(dim=1), \
                               torch.Tensor(rewards).unsqueeze(dim=1)

    dones = torch.Tensor(dones).unsqueeze(dim=1)
    return states, actions, rewards, dones


def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized
    # while the loss should be minimized so you need a - somewhere

    # YOUR CODE HERE
    states, actions, rewards, dones = episode

    # Calculate rewards.
    rewards = rewards.squeeze()
    G = torch.zeros_like(rewards)
    # Need to calculate loss using formula G_{t+1} = r_t + \gamma G_{t+1}
    # If statement makes sure that there isn't an error when t=0. Otherwise, we'd get an error
    #  since there's no negative time step.
    for t in reversed(range(rewards.shape[0])):
        G[t] = rewards[t] + ((discount_factor * G[t + 1]) if t + 1 < rewards.shape[0] else 0)

    # Calculate loss.
    action_probs = torch.log(policy.get_probs(states, actions)).squeeze()
    loss = - (action_probs * G).sum()
    return loss


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate,
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)

    episode_durations = []
    rewards = []
    for i in range(num_episodes):
        # YOUR CODE HERE
        episode = sample_episode(env, policy)
        loss = compute_reinforce_loss(policy, episode, discount_factor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        rewards.append(sum(episode[1]))

    return episode_durations, rewards