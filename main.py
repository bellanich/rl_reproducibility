import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import sys
import gym
import time
import argparse
import os
import pickle

# Personal imports.
from model import NNPolicy
from utils import run_episodes_policy_gradient, smooth
from configurations import grid_search_configurations

"""
TODO List:
(1) Figure out if tqdm is necessary and what is does.
 > tqdm is a progress bar, it's purely aesthetic but nice to have if your code takes long to run. https://tqdm.github.io/
(2) Figure out to measure model performance, so we can save best model/results. (i.e., What metric do we use?)
 > Jochem suggests accumulated reward per episode.
(3) Subtract out baseline from REINFORCE AND GPOMDP. Do additional tricks if necessary to stabilize performance.
"""

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

""" TODO remove this
# Arg parser. This will make it easier when we eventually have to write a bash script to send things to Lisa.
parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_argument('--num-episodes', dest='num_episodes', default=50, type=int) #500
parser.add_argument('--discount-factor', dest='discount_factor', default=0.99, type=float)
parser.add_argument('--seed', dest='seed', default=42,  type=int)
parser.add_argument('--learning-rate', dest='learn_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--hidden-layers', dest='num_hidden', default=128, type=int, help='number of hidden layers')
args = parser.parse_args()

# List of environment we'll be applying our model to.
my_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
"""

# Directories to save output files in.
figures_path, models_path = os.path.join('outputs', 'figures'), os.path.join('outputs', 'models')
# Dictionaries we'll use to save results.
best_performance = {}
policy_gradients = {}
rewards = {}

for config in grid_search_configurations():
    env_name = config["environment"]

    best_performance[env_name] = -1
    policy_gradients[env_name] = None
    rewards[env_name] = 0

    # Make environment.
    env = gym.make(env_name)
    print("Initialize Network for {}.".format(env_name))
    # Now seed both the environment and network.
    torch.manual_seed(config["seed"])
    env.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Finally, initialize network. (Needs to be reinitalized, because input dim size varies with environment.
    if config["policy"] == "gpomdb":
        policy = NNPolicy(input_size=env.observation_space.shape[0],
                        output_size=env.action_space.n,
                        num_hidden=config["hidden_layer"])
    else:
        raise NotImplementedError

    print("Training for {} episodes.".format(config["num_episodes"]))
    # Simulate N episodes. (Code from lab.)
    episode_durations_policy_gradient, episode_rewards, episode_losses = run_episodes_policy_gradient(policy,
                                                                                                      env,
                                                                                                      config["num_episodes"],
                                                                                                      config["discount_factor"],
                                                                                                      config["learning_rate"])

    # The policy gradients will be saved and used to generate plots later.
    smooth_policy_gradients = smooth(episode_durations_policy_gradient, 10)
    smooth_rewards = smooth(episode_rewards, 10)
    # todo: add if statement to check model performance before saving
    policy_gradients[env_name] = smooth_policy_gradients
    rewards[env_name] = smooth_rewards

    # Save best policy.
    # todo: add if statement to check model performance before saving
    torch.save(policy.state_dict(), os.path.join(models_path, "{}_model.pt".format(env_name.replace('-','_'))))

# Save results.
filename = os.path.join('outputs', 'policy_gradients','best_policy_gradients.pickle')
with open(filename, 'wb') as handle:
    pickle.dump(policy_gradients, handle, protocol=pickle.HIGHEST_PROTOCOL)

filename = os.path.join('outputs', 'rewards','best_rewards.pickle')
with open(filename, 'wb') as handle:
    pickle.dump(rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

