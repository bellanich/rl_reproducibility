"""
File of functions we used in lab. Called 'utils', since I couldn't think of anything better and didn't want them in
main script.
"""

import torch
from torch import optim
import numpy as np
import sys
import os

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


def compute_reinforce_loss(policy, episode, discount_factor, baseline=None):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Same as gpomdp function, only there's a slight difference in loss equation.
    states, actions, rewards, dones = episode
    rewards = rewards.squeeze()
    G = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        G[t] = rewards[t] + ((discount_factor * G[t + 1]) if t + 1 < rewards.shape[0] else 0)

    # Also called "whitening" the gradients.
    if baseline == "normalized_baseline":
        G = (G - G.mean()) / G.std()

    # Use random policy as baseline.
    if baseline == "random_baseline":
        print("Save G for untrained model and subtract from G calculated above.")
        raise NotImplemented

    action_probs = torch.log(policy.get_probs(states, actions)).squeeze()
    loss = - (action_probs * G[0]).sum()
    return loss


def compute_gpomdp_loss(policy, episode, discount_factor, baseline=None):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # YOUR CODE HERE
    states, actions, rewards, dones = episode

    # Calculate rewards.
    rewards = rewards.squeeze()
    G = torch.zeros_like(rewards)
    # Need to calculate loss using formula G_{t+1} = r_t + \gamma G_{t+1}. If statement makes sure that there isn't
    # an error when t=0. Otherwise, we'd get an error since there's no negative time step.
    for t in reversed(range(rewards.shape[0])):
        G[t] = rewards[t] + ((discount_factor * G[t + 1]) if t + 1 < rewards.shape[0] else 0)

    # Also called "whitening" the gradients.
    if baseline == "normalized_baseline":
        G = (G - G.mean()) / G.std()

    # Use random policy as baseline.
    if baseline == "random_baseline":
        print("Save G for untrained model and subtract from G calculated above.")
        raise NotImplemented

    # Calculate loss.
    action_probs = torch.log(policy.get_probs(states, actions)).squeeze()
    loss = - (action_probs * G).sum()
    return loss


def eval_policy(policy, env, config, loss_function):

    # Return gradients per episode
    episode_gradients, losses = dict(), list()
    for _ in range(config["num_episodes"]):
        episode = sample_episode(env, policy)
        policy.zero_grad()  # We need to reset the optimizer gradients for each new run.
        loss = loss_function(policy, episode, config["discount_factor"], config["baseline"])
        loss.backward()

        # Save losses as a list.
        losses.append(loss.item())

        # Extracting gradients from policy network.
        for name, param in policy.named_parameters():
            if name not in episode_gradients:
                episode_gradients[name] = []
            episode_gradients[name].append(param.grad.cpu().detach().view(-1))

    episode_gradients = {key: torch.stack(episode_gradients[key], dim=0).numpy() for key in episode_gradients}
    average_loss = np.asarray(losses).mean()

    return episode, average_loss, episode_gradients


def run_episodes_policy_gradient(policy, env, config):

    # Define loss function using policy_name.
    policy_name = config["policy"]
    loss_function = compute_reinforce_loss if "reinforce" in policy_name else compute_gpomdp_loss
    baseline = config["baseline"]

    # Setting up for training.
    optimizer = optim.Adam(policy.parameters(), config["learning_rate"])
    episode_durations, rewards, losses = list(), list(), list()
    policy_description = "{}_seed_{}_lr_{}_discount_{}_sampling_freq_{}".format(config["environment"].replace('-', '_'),
                                                                                  config["seed"],
                                                                                  config["learning_rate"],
                                                                                  config["discount_factor"],
                                                                                  config["sampling_freq"])

    # Setting policy to be trained.
    policy.train()
    for i in range(config["num_episodes"]):

        episode = sample_episode(env, policy)
        optimizer.zero_grad()  # We need to reset the optimizer gradients for each new run.
        # With the way it's currently coded, we need the same input and outputs for this to work.
        loss = loss_function(policy, episode, config["discount_factor"], baseline)
        loss.backward()
        optimizer.step()

        # Validating (or "freezing" training of the model).
        if i % config["sampling_freq"] == 0:
            # Calling separate function to do validation. No gradients are taken, and
            episode, avg_loss, current_gradients = eval_policy(policy, env, config, loss_function)

            # Printing something just so we know what's going on.
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))


            # Save episode durations, rewards, and losses to visualize later.
            episode_durations.append(len(episode[0])), rewards.append(sum(episode[1])), losses.append(float(avg_loss))

            # Saving policy gradients per 'validation' iteration.
            gradients_path = os.path.join('outputs', 'policy_gradients', config["policy"], policy_description)
            # Create dir if doesn't already exist.
            if not os.path.exists(gradients_path):
                os.mkdir(gradients_path)
            np.savez_compressed(os.path.join(gradients_path, "timestep_{}_gradients".format(i)), current_gradients)

    return episode_durations, rewards, losses