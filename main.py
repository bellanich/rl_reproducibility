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
(1) B: Debug sending tensors to gpu.

"""

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# Directories to save output files in.
figures_path, models_path = os.path.join('outputs', 'figures'), os.path.join('outputs', 'models')
# Check if gpu is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for config in grid_search_configurations():
    # Make environment.
    env_name = config["environment"]
    env = gym.make(env_name)

    # Add device to config. (Device needs to be passed to any module where we initialize Torch tensors. Use config to
    #   do so, since it means making less changes to the code.
    config['device'] = device

    print("Initializing the network for configuration:")
    for key, value in config.items():
        print(f'    {key:<15} {value}')

    # Now seed both the environment and network.
    torch.manual_seed(config["seed"])
    env.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Finally, initialize network. (Needs to be reinitalized, because input dim size varies with environment.
    acceptable_policies = ["gpomdp", "reinforce"]
    if config["policy"] in acceptable_policies:
        policy = NNPolicy(input_size=env.observation_space.shape[0],
                        output_size=env.action_space.n,
                        num_hidden=config["hidden_layer"]).to(device)
    else:
        raise NotImplementedError

    print("Training for {} episodes.".format(config["num_episodes"]))
    # Simulate N episodes. (Code from lab.)
    episodes_data = run_episodes_policy_gradient(policy, 
                                                 env,
                                                 config)
    durations, rewards, losses = episodes_data

    # The policy gradients will be saved and used to generate plots later.
    smooth_policy_gradients = smooth(durations, 10)
    smooth_rewards = smooth(rewards, 10)

    # Save trained policy. We save the policy under the name of its hyperparameter values.
    baseline_name = "_".join(config['baseline'].split('_')[:-1])
    policy_description = "{}_baseline_{}_{}_seed_{}_lr_{}_discount_{}_sampling_freq_{}".format(config["policy"],
                                                                                baseline_name,
                                                                                config["environment"].replace('-', '_'),
                                                                                config["seed"],
                                                                                config["learning_rate"],
                                                                                config["discount_factor"],
                                                                                config["sampling_freq"])

    # Saving model
    policy_name = "{}_{}".format(config["policy"], config["baseline"]) if config["baseline"] is not None else config['policy']
    model_filename = os.path.join(models_path, policy_name, policy_description)
    torch.save(policy.state_dict(), "{}.pt".format(model_filename))
    # Saving rewards and loss.
    np.save(os.path.join('outputs', 'rewards', config['policy'], "{}_rewards".format(policy_description)), rewards)
    np.save(os.path.join('outputs', 'losses', config['policy'], "{}_losses".format(policy_description)), losses)

