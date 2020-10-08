"""
File of functions we used in lab. Called 'utils', since I couldn't think of anything better and didn't want them in
main script.
"""

import torch
from torch import optim
import numpy as np
import sys

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

    states, actions, rewards, dones = episode
    rewards = rewards.squeeze()
    G = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        G[t] = rewards[t] + ((discount_factor * G[t + 1]) if t + 1 < rewards.shape[0] else 0)

    action_probs = torch.log(policy.get_probs(states, actions)).squeeze()
    loss = - (action_probs * G[0]).sum()
    return loss


def compute_gpomdp_loss(policy, episode, discount_factor):
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


def eval_policy(policy, env, discount_factor):

    episode_gradients = dict()
    for _ in range(1000):
        episode = sample_episode(env, policy)
        policy.zero_grad()  # We need to reset the optimizer gradients for each new run.
        loss = compute_reinforce_loss(policy, episode, discount_factor)
        loss.backward()

        # Extracting gradients from policy network.
        for name, param in policy.named_parameters():
            if name not in episode_gradients:
                episode_gradients[name] = []
            episode_gradients[name].append(param.grad.cpu().detach().view(-1))
    episode_gradients = {key: torch.stack(episode_gradients[key], dim=0).numpy() for key in episode_gradients}

    return episode, loss, episode_gradients


def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate,
                                 loss_function, sampling_freq=10,
                                 sampling_function=sample_episode):

    optimizer = optim.Adam(policy.parameters(), learn_rate)
    episode_durations, rewards, losses, gradients = list(), list(), list(), list()

    # Setting policy to be trained.
    policy.train()
    for i in range(num_episodes):

        episode = sample_episode(env, policy)
        optimizer.zero_grad()  # We need to reset the optimizer gradients for each new run.
        # The loss_function is a variable meant for loss functions. With the way it's currently coded, we need the same
        #   input and outputs for this to work.
        loss = loss_function(policy, episode, discount_factor)
        loss.backward()
        optimizer.step()

        # Validating (or "freezing" training of the model).
        if i % sampling_freq == 0:
            # Calling separate function to do validation. No gradients are taken, and
            episode, loss, episode_gradients = eval_policy(policy, env, discount_factor)

            # Printing something just so we know what's going on.
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))


            # Save episode durations, rewards, and losses to plot later.
            episode_durations.append(len(episode[0])), rewards.append(sum(episode[1])), losses.append(float(loss))
            print("Done. Things works this far.")
            sys.exit(1)
            # todo: Fix me. This isn't correct anymore!
            gradients.append(torch.cat(episode_gradients))


    return episode_durations, rewards, losses, torch.stack(gradients)