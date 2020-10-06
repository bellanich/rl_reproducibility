import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import matplotlib.pyplot as plt
import sys
import gym
import time
import argparse

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
# Reminder to activate mini-environment
assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

# Personal imports.
from model import NNPolicy
from utils import run_episodes_policy_gradient, smooth

# Arg parser. This will make it easier when we eventually have to write a bash script to send things to Lisa.
parser = argparse.ArgumentParser(description='Model Parameters')
parser.add_option('-n', '--num-episodes', dest='num_episodes', default=500, type='int')
parser.add_option('-g', '--discount-factor', dest='discount_factor', default=0.99, type='float')
parser.add_option('-s', '--seed', dest='seed', default=42,  type='int')
parser.add_option('-l', '--learning-rate', dest='learn_rate', default=0.001, type='float', help='learning rate')
(options, args) = parser.parse_args()

# Let's sample some episodes
my_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
env = gym.make(my_envs[2])

# Initialize network.
# We need to use the shape of our observations matrix to define our input_dim size. This is a hackey way to get it.
intial_step = env.reset()
num_hidden = 128
torch.manual_seed(1234)
policy = NNPolicy(input_size=intial_step.shape[0],
                  output_size=env.action_space.n,
                  num_hidden=num_hidden)
# Now seed both the environment and network.
torch.manual_seed(args.seed)
env.seed(args.seed)

# Simulate N episodes. (Code from lab.)
episode_durations_policy_gradient = run_episodes_policy_gradient(policy,
                                                                 env,
                                                                 args.num_episodes,
                                                                 args.discount_factor,
                                                                 args.learn_rate)

# Plot learning curves.
plt.plot(smooth(episode_durations_policy_gradient, 10))
plt.title('Episode durations per episode')
plt.legend(['Policy gradient'])
plt.show()