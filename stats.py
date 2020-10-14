import numpy as np
import pickle as pkl
from scipy.stats import moment
import sys
import os

from configurations import HIDDEN_LAYERS


# TODO; These need to be set to the correct values, maybe more elegantly.
BEST_LEARNING_RATE = 0.001
BEST_DISCOUNT_FACTOR = 0.99


def gen_gradients_files(root):
    '''Yield all files and configurations containting policy gradients from 
    root.
    '''

    # Where to load data and save generated figures.
    for model_folder in os.scandir(root):
        if model_folder.is_file():
            continue
        # Note the model name and baseline from the folder name.
        policy = 'reinforce'
        baseline = None
        if model_folder.name.find('gpomdp') >= 0:
            policy = 'gpomdp'
            if model_folder.name.find('normalized_baseline') >= 0:
                baseline = 'normalized_baseline'

        # Loop over all different configurations per model + baseline.
        for config_folder in os.scandir(model_folder.path):
            folder_name = config_folder.name.split('_')
            if config_folder.is_file() or folder_name[0] != policy:
                continue
            while 'baseline' in folder_name:
                folder_name.remove('baseline')

            # Yield the hyperparameters and freeze of the gradients at path.
            for gradients_file in os.scandir(config_folder.path):
                if gradients_file.is_dir():
                    continue
                config = {
                    "environment" : f'{folder_name[2]}_{folder_name[3]}',
                    "policy" : policy,
                    "learning_rate" : float(folder_name[7]),
                    "discount_factor" : float(folder_name[9]),
                    "seed" : int(folder_name[5]),
                    "hidden_layer" : HIDDEN_LAYERS,
                    "sampling_freq": int(folder_name[12]),
                    "baseline": baseline
                }
                freeze_num = int(gradients_file.name.split('_')[1])
                yield gradients_file.path, config, freeze_num
            
            if not baseline:
                print(config)


root = os.path.join('outputs', 'policy_gradients')
stats = {}
for gradients_file_path, config, freeze in gen_gradients_files(root):
    if not (config['learning_rate'] == BEST_LEARNING_RATE
            and config['discount_factor'] == BEST_DISCOUNT_FACTOR):
        continue

    # Get name for setting + freeze point
    setting_freeze = "{}_baseline_{}_{}_freeze_{}".format(config["policy"], 
                                                          config["baseline"],
                                                          config["environment"].replace('-', '_'),
                                                          freeze)
    if setting_freeze not in stats.keys():
        stats[setting_freeze] = {'mean':[], 'var':[], 'skew':[], 'kurt':[]}

    # load gradients
    grads = np.load(gradients_file_path, allow_pickle=True)
    grads = grads['arr_0'].item()

    # 2000 gradients per parameter theta
    grads = np.hstack((grads['l1.weight'],grads['l2.weight']))

    # statistics per parameter over the 2000 roll-outs
    mean = moment(grads, moment=1)
    var = moment(grads, moment=2)
    skew = moment(grads, moment=3)
    kurt = moment(grads, moment=4)

    # averaged statistics over all parameters and add to stats dict
    mean, var, skew, kurt = np.mean(mean), np.mean(var), np.mean(skew), np.mean(kurt)
    stats[setting_freeze]['mean'].append(np.mean(mean))
    stats[setting_freeze]['var'].append(np.mean(var))
    stats[setting_freeze]['skew'].append(np.mean(skew))
    stats[setting_freeze]['kurt'].append(np.mean(kurt))


# Average over seeds
for key in stats.keys():
    stats[key]['mean'] = np.mean(stats[key]['mean'])
    stats[key]['var'] = np.mean(stats[key]['var'])
    stats[key]['skew'] = np.mean(stats[key]['skew'])
    stats[key]['kurt'] = np.mean(stats[key]['kurt'])


pkl.dump(stats, open( "stats.pkl", "wb" ) )
