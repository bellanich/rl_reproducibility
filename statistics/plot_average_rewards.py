import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os, sys, glob
from collections import namedtuple, defaultdict
from configurations import HIDDEN_LAYERS
from utils import smooth
sns.set()

"""
Small script to generate learning curves from best training runs.
"""

def load_reward_files(root):
    '''Load all reward arrays into a dictionary with configurations as keys.
    Identical configurations with different seeds are appended under the same
    key.
    '''
    # This named tuple will function as a hashable key.
    Configuration = namedtuple('Config', ["environment",
                                          "policy",
                                          "learning_rate",
                                          "discount_factor",
                                          "hidden_layer",
                                          "sampling_freq",
                                          "baseline"])
    config2rewards = defaultdict(list)

    # The top layer contains a map per model.
    for model_folder in os.scandir(root):
        if model_folder.is_file():
            continue

        # Loop over all different configurations per model.
        for rewards_file in os.scandir(model_folder.path):

            filename = rewards_file.name.split('_')
            if rewards_file.is_dir() or filename[0] != model_folder.name:
                continue

            while 'baseline' in filename:
                filename.remove('baseline')

            config = Configuration(
                environment=     f'{filename[2]}_{filename[3]}',
                policy=          model_folder.name,
                learning_rate=   float(filename[7]),
                discount_factor= float(filename[9]),
                hidden_layer=    HIDDEN_LAYERS,
                sampling_freq=   int(filename[12]),
                baseline=        filename[1]
            )

            reward = []
            for elem in np.load(rewards_file.path, allow_pickle=True):
                reward.append(float(elem))

            config2rewards[config].append(reward)

    return config2rewards


def pad_rewards_to_array(config2rewards):
    '''Because of the variable run-length per configuration. We need to pad the
    rewards with their last values before making arrays.'''

    for config, rewards in config2rewards.items():

        max_len = 0
        for ep_reward in rewards:
            max_len = max(len(ep_reward), max_len)

        # Pad the episode rewards untill all are of max length.
        rewards_2d = np.empty((len(rewards), max_len))
        for i, ep_rewards in enumerate(rewards):
            last_reward = ep_rewards[-1]
            for _ in range(max_len - len(ep_rewards)):
                ep_rewards.append(last_reward)
            rewards_2d[i] = ep_rewards

        config2rewards[config] = rewards_2d

    return config2rewards

root = os.path.join('..', 'outputs', 'rewards')
save_path = os.path.join('..', 'outputs', 'figures', 'cumulative_rewards', 'averaged_over_seed')
config2rewards = load_reward_files(root)
# Repeat last value so we can clearly see that our result has converged.
config2rewards = pad_rewards_to_array(config2rewards)

# Use counter just so we can see something while figures are being generated.
counter = 1

for config, rewards in config2rewards.items():
    config = config._asdict()

    # Initializing figure.
    print(f"Generating Figure {counter}")
    fig = plt.figure(1)

    # Reshaping rewards and defining x axis values.
    avg_rewards = rewards.mean(0)
    standard_dev = rewards.std(0)

    # Only smoothen rewards and std array if there are too many points.
    if avg_rewards.shape[0] > 1e3:
        smooth_factor = np.floor(avg_rewards.shape[0]/100).astype(int) # Make smoothing factor dynamic.
        avg_rewards = smooth(avg_rewards, N=smooth_factor)
        standard_dev = smooth(standard_dev, N=smooth_factor)
    episodes = np.arange(avg_rewards.shape[0])

    plt.plot(episodes, avg_rewards, label=config['policy'])
    # Visualize standard deviation.
    plt.fill_between(episodes,
                     (avg_rewards - standard_dev),
                     (avg_rewards + standard_dev), alpha=0.5)

    # Save figure.
    plt.title('Cumulative Rewards')
    # Only show legend if we choose to show several results in one plot.
    # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=len(files_list), borderaxespad=0.)

    policy_description = "{}_baseline_{}_{}_lr_{}_discount_{}_sampling_freq_{}.jpg".format(config["policy"],
                                                                                config["baseline"],
                                                                                config["environment"],
                                                                                config["learning_rate"],
                                                                                config["discount_factor"],
                                                                                config["sampling_freq"])

    fig.savefig(os.path.join(save_path, policy_description), bbox_inches='tight')
    fig.clear()

    counter += 1

