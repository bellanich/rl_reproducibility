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
from utils import run_episodes_policy_gradient, smooth, compute_reinforce_loss, compute_gpomdp_loss
from configurations import grid_search_configurations

"""
TODO List:
(1) B: Double check that it actually makes more sense to just pass the all of the config dictionary to
run_episodes_policy_gradient(). The way it's coded right now, the number of variables we would need to pass otherwise
gets too annoying.

(2) Verify what we're saving is actually the metrics we're interested in.

(3) B: Double check the two baselines and the way they're done makes sense.
"""

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# Directories to save output files in.
figures_path, models_path = os.path.join('outputs', 'figures'), os.path.join('outputs', 'models')

for config in grid_search_configurations():
    # Make environment.
    env_name = config["environment"]
    env = gym.make(env_name)

    print("Initialize {} Network for {}.".format(config["policy"], env_name))
    # Now seed both the environment and network.
    torch.manual_seed(config["seed"])
    env.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Finally, initialize network. (Needs to be reinitalized, because input dim size varies with environment.
    acceptable_policies = ["gpomdp", "reinforce", "normalized_baseline"]
    if config["policy"] in acceptable_policies:
        policy = NNPolicy(input_size=env.observation_space.shape[0],
                        output_size=env.action_space.n,
                        num_hidden=config["hidden_layer"])
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
    policy_description = "{}_seed_{}_lr_{}_discount_{}_sampling_freq_{}".format(config["environment"].replace('-', '_'),
                                                                                  config["seed"],
                                                                                  config["learning_rate"],
                                                                                  config["discount_factor"],
                                                                                  config["sampling_freq"])

    # Saving model
    model_filename = os.path.join(models_path, config['policy'], policy_description)
    torch.save(policy.state_dict(), "{}.pt".format(model_filename))
    # Saving rewards and loss.
    np.save(os.path.join('outputs', 'rewards', config['policy'],"{}_rewards".format(policy_description)), rewards)
    np.save(os.path.join('outputs', 'losses', config['policy'], "{}_losses".format(policy_description)), losses)

