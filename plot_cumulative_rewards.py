import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os, sys, glob
sns.set()

"""
Small script to generate learning curves from best training runs.

"""

# Where to load data and save generated figures.
data_dir = os.path.join('dummy_results')
files_list = os.listdir(data_dir)
save_path = os.path.join('outputs', 'figures', 'cumulative_rewards')

# TODO: WRITE CODE HERE
# We need to somehow group files by environment type and then loop through that when generating figures.

# Initialize the one figure we'll put all of the plots in.
fig = plt.figure(1)

for filename in files_list:
    # Load info.
    env_name, model = filename.split('_')[3], " ".join(filename.split('_')[:3]).lower()
    sample_freq = filename.split('_')[-2]
    policy_rewards = np.load(os.path.join(data_dir, filename))

    # TODO: LOAD STATS HERE.
    # Once policy gradient statistics are calculated, then let's load this here
    # Should replace variance variable.
    variance = policy_rewards.var()
    # Use sample frequency to recreate the x-axis.
    episodes = np.arange(policy_rewards.shape[0]) * int(sample_freq)

    # Putting results into plot.
    plt.plot(episodes, policy_rewards, label=model)
    plt.fill_between(episodes,
                    (policy_rewards - variance),
                    (policy_rewards + variance), alpha=0.3)

plt.title('Cumulative Rewards')
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=len(files_list), borderaxespad=0.)
fig.savefig(save_path, bbox_inches='tight')
