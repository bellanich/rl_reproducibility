import os
import numpy as np
from collections import namedtuple, defaultdict


HIDDEN_LAYERS = 128


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

    # Loop over all different configurations per model.
    for rewards_file in os.scandir(root):

        filename = rewards_file.name.split('_')
        if rewards_file.is_dir():
            continue

        while 'baseline' in filename:
            filename.remove('baseline')

        npz = np.load(rewards_file.path, allow_pickle=True)
        print(len(npz.files))
        values = npz[npz.files[0]].item()
        print(values)
        print(type(values))
        continue
        config = Configuration(
            environment=     filename[1],
            policy=          None,
            learning_rate=   float(filename[7]),
            discount_factor= float(filename[9]),
            hidden_layer=    HIDDEN_LAYERS,
            sampling_freq=   int(filename[12]),
            baseline=        filename[1]
        )

        reward = []

        config2rewards[config].append(reward)

    return config2rewards

root = os.path.join('..', 'outputs', 'rewards')
load_reward_files(root)
