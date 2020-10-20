import numpy as np
import pickle as pkl
from scipy.stats import moment
import sys
import os
from configurations import HIDDEN_LAYERS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

def create_plot(stats, env, stat):
    df = pd.DataFrame(columns=['algorithm','episode_number',stat])
    i = 0
    for key in stats.keys():
        if env in key:
            # Get algorithm name, episode_nr and stat value from key
            key_split = key.split(env)
            alg = key_split[0][:-1]
            alg = alg.split('_baseline_')
            if 'None' in alg:
                alg_name = alg[0]
            else:
                alg_name = alg[0] + '_' + alg[1]
            freeze = int(key_split[1][8:])
            values = stats[key][stat]

            # loop over all seeds in values and add row to dataframe
            for value in values:
                # Add tot dataframe
                df.loc[i] = [alg_name, freeze, value]
                i += 1

    sns.lineplot(data=df, x="episode_number", y=stat, hue="algorithm")
    #ax = sns.barplot(x="episode_number", y=stat, hue="algorithm", data=df) # ACTIVATE THIS AND LINE BELOW FOR CREATING BARPLOT INSTEAD
    #ax.plot()
    plt.legend(loc='best')

    file_name = env + '_' + stat
    plt.savefig(file_name)
    plt.clf()



# TODO; These need to be set to the correct values, maybe more elegantly.
BEST_LEARNING_RATE = 0.001
BEST_DISCOUNT_FACTOR = 0.99

# Rescale data
def scale(data, min, max):
    nom = (data-data.min(axis=0))*2
    denom = data.max(axis=0) - data.min(axis=0)
    denom[denom==0] = 1
    return -1 + nom/denom

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
        stats[setting_freeze] = {'variance':[], 'kurtosis':[]}

    # load gradients
    grads = np.load(gradients_file_path, allow_pickle=True)
    grads = grads['arr_0'].item()

    # 500 gradients per parameter theta
    grads = np.hstack((grads['l1.weight'],grads['l2.weight']))
    grads = scale(grads, -1, 1) # scale gradients to -1,1

    # statistics per parameter over the 500 roll-outs
    var = moment(grads, moment=2)
    kurt = moment(grads, moment=4)



    # averaged statistics over all parameters and add to stats dict
    var, kurt = np.mean(var), np.mean(kurt)
    stats[setting_freeze]['variance'].append(var)
    stats[setting_freeze]['kurtosis'].append(kurt)

# # Average over seeds
# for key in stats.keys():
#     stats[key]['var'] = np.mean(stats[key]['var'])
#     stats[key]['kurt'] = np.mean(stats[key]['kurt'])
#

pkl.dump(stats, open( "stats.pkl", "wb" ) )



# Plotting statistics
create_plot(stats, 'Acrobot_v1', 'variance')
create_plot(stats, 'Acrobot_v1', 'kurtosis')

# TODO Only did acrobot as this is the only env in the dummy data.
# Can you create the plots for the other envs?
