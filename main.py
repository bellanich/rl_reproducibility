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
from utils import run_episodes_policy_gradient, initialize_dirs
from configurations import grid_search_configurations, SEEDS
from GridWorld import GridworldEnv


def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# Setting up files.
# Directories to save output files in.
figures_path, models_path = os.path.join('outputs', 'figures'), os.path.join('outputs', 'models')
initialize_dirs(dir_paths=[figures_path, models_path])

# Check if gpu is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timing_filepath = os.path.join('outputs', f'timing_seed_{SEEDS[0]}_{SEEDS[-1]}.csv')
with open(timing_filepath, 'w') as t_file:
    t_file.write('policy,baseline,environment,seed,learning_rate,'
                 + 'discount_factor,sampling_freq,episode_time,total_time\n')
t_0 = time()

# For each configuration, we train the model.
for config in grid_search_configurations():
    t_ep = time()
    # Make environment.
    env_name = config["environment"]
    env = gym.make(env_name) if env_name!= 'GridWorld' else GridworldEnv(shape=[5,5])
    config['device'] = device

    print("Initializing the network for configuration:")
    for key, value in config.items():
        print(f'    {key:<15} {value}')

    # Now seed both the environment and network.
    torch.manual_seed(config["seed"])
    env.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # We make a quick check to make sure all the policies and baselines we'll use are valid before initializing our
    #   model.
    acceptable_policies = ["gpomdp", "reinforce", "normalized_gpomdp"]
    if set(config["policies"]).issubset(acceptable_policies):
        # Since GridWorld is a discrete environment, we need to define the input size a bit differently than in CartPole.
        input_dim = env.nS if env_name == 'GridWorld' else env.observation_space.shape[0]
        policy = NNPolicy(input_size=input_dim,
                        output_size=env.action_space.n,
                        num_hidden=config["hidden_layer"]).to(device)
    else:
        raise NotImplementedError

    print("Training for {} episodes.".format(config["num_episodes"]))
    # Simulate N episodes. (Code from lab.)
    policy_rewards, policy_losses, training_performance = run_episodes_policy_gradient(policy,
                                                                                       env,
                                                                                       config)

    # Save trained policy. We save the policy under the name of its hyperparameter values.
    # Hacky fix to deal with baseline names.
    model_description = "env{}_seed_{}_lr_{}_discount_{}_sampling_freq_{}".format(
                                                                                config["environment"].replace('-', '_'),
                                                                                config["seed"],
                                                                                config["learning_rate"],
                                                                                config["discount_factor"],
                                                                                config["sampling_freq"])

    # Saving model
    current_model_path = os.path.join(models_path, env_name)
    initialize_dirs(dir_paths=[current_model_path])
    torch.save(policy.state_dict(), os.path.join(current_model_path, "{}.pt".format(model_description)))

    # Saving time it took for machine to run.
    time_data = [
        str(config["policies"]),
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
