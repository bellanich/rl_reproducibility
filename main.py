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
from time import time

# Personal imports.
from model import NNPolicy
from utils import run_episodes_policy_gradient, smooth
from configurations import grid_search_configurations, SEEDS

"""
TODO List:
"""

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# Directories to save output files in.
figures_path, models_path = os.path.join('outputs', 'figures'), os.path.join('outputs', 'models')
for folder_path in (figures_path, models_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
# Check if gpu is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

timing_filepath = os.path.join('outputs', 
                               f'timing_seed_{SEEDS[0]}_{SEEDS[-1]}.csv')
with open(timing_filepath, 'w') as t_file:
    t_file.write('policy,baseline,environment,seed,learning_rate,'
                 + 'discount_factor,sampling_freq,episode_time,total_time\n')

t_0 = time()
for config in grid_search_configurations():
    # Make environment.
    t_ep = time()
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
    trn_rewards, trn_losses = run_episodes_policy_gradient(policy, 
                                                           env,
                                                           config)

    # Save trained policy. We save the policy under the name of its hyperparameter values.
    if config['baseline'] != None:
        baseline_name = "_".join(config['baseline'].split('_')[:-1])
    else:
        baseline_name = config['baseline']
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
    if not os.path.exists(model_filename):
        os.makedirs(model_filename)
    torch.save(policy.state_dict(), "{}.pt".format(model_filename))
    # Saving rewards and loss.
    rewards_folder = os.path.join('outputs', 'rewards', config['policy'])
    losses_folder = os.path.join('outputs', 'losses', config['policy'])
    for save_folder in (rewards_folder, losses_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    np.save(os.path.join(rewards_folder, 
                         "{}_rewards".format(policy_description)), 
            trn_rewards)
    np.save(os.path.join(losses_folder, 
                         "{}_losses".format(policy_description)),
            trn_losses)

    time_data = [
        str(config["policy"]),
        str(baseline_name),
        str(config["environment"].replace('-', '_')),
        str(config["seed"]),
        str(config["learning_rate"]),
        str(config["discount_factor"]),
        str(config["sampling_freq"]),
        str(int(time() - t_ep)),
        str(int(time() - t_0))
    ]
    with open(timing_filepath, 'a') as t_file:
        t_file.write(','.join(time_data) + '\n')
