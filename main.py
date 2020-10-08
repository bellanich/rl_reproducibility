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
(1) Think critically about how we're currently saving the gradients. I just found code online, but haven't run any tests
to make sure that it's correct.
 J: I checked it, I think this is correct. The view(-1) has all gradients lose dimensionality tho. We may want them to remain their original forms.
(2) Figure out if we even need a 'best_performance' dictionary anymore...
 J: No I don't think so.
(3) Subtract out baseline from REINFORCE AND GPOMDP. Do additional tricks if necessary to stabilize performance.
(4) Integrate Natural Policy Gradients into coding framework.
"""

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# Directories to save output files in.
figures_path, models_path = os.path.join('outputs', 'figures'), os.path.join('outputs', 'models')

for config in grid_search_configurations():
    env_name = config["environment"]

    # Make environment.
    env = gym.make(env_name)
    print("Initialize Network for {}.".format(env_name))
    # Now seed both the environment and network.
    torch.manual_seed(config["seed"])
    env.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Finally, initialize network. (Needs to be reinitalized, because input dim size varies with environment.
    if config["policy"] == "gpomdp":
        policy = NNPolicy(input_size=env.observation_space.shape[0],
                        output_size=env.action_space.n,
                        num_hidden=config["hidden_layer"])
    else:
        raise NotImplementedError

    print("Training for {} episodes.".format(config["num_episodes"]))
    # Simulate N episodes. (Code from lab.)
    episodes_data = run_episodes_policy_gradient(policy, 
                                                 env,
                                                 config["num_episodes"],
                                                 config["discount_factor"],
                                                 config["learning_rate"],
                                                 config["sampling_freq"])
    durations, rewards, losses, gradients = episodes_data

    # The policy gradients will be saved and used to generate plots later.
    smooth_policy_gradients = smooth(durations, 10)
    smooth_rewards = smooth(rewards, 10)

    # Save best policy. Best policy saved by hyperparameter values.
    policy_description = "{}_seed_{}_lr_{}_discount_{}_samplingfreq_{}.pt".format(config["environment"].replace('-', '_'),
                                                                                  config["seed"],
                                                                                  config["learning_rate"],
                                                                                  config["discount_factor"],
                                                                                  config["sampling_freq"])
    model_filename = os.path.join(models_path, config['policy'], policy_description)
    torch.save(policy.state_dict(), model_filename)
    # Save network gradients

    torch.save(gradients, os.path.join('outputs', 'policy_gradients', config['policy'], policy_description))

