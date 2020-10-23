import os
import numpy as np
from collections import namedtuple, defaultdict


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
                                          "baseline"])
    config2rewards = defaultdict(list)

    # Loop over all different configurations per model.
    for rewards_file in os.scandir(root):

        filename = rewards_file.name.split('_')
        if rewards_file.is_dir():
            continue

        while 'baseline' in filename:
            filename.remove('baseline')

        npz = np.load(rewards_file.path, allow_pickle=True)
        policy2reward = npz[npz.files[0]].item()
        print(filename)

        for policy, reward in policy2reward.items():

            baseline = None
            if 'baseline' in policy:
                baseline = 'baseline'

            config = Configuration(
                environment=     filename[1],
                policy=          policy,
                learning_rate=   float(filename[5]),
                discount_factor= float(filename[8]),
                baseline=        baseline
            )

            config2rewards[config].append(reward)

    return config2rewards


root = os.path.join('..', 'outputs', 'rewards')
config2reward = load_reward_files(root)
print(config2reward)
