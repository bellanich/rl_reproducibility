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
save_path = [os.path.join('outputs', 'figures', 'cumulative_rewards')]

# WRITE CODE HERE
# To somehow group files by environment type and generate figures accordingly.

# Initialize the one figure we'll put all of the plots in.
fig = plt.figure(1)

for filename in files_list:
    print(filename)
    # Load info.
    env_name, model = filename.split('_')[3], " ".join(filename.split('_')[:3]).lower()
    sample_freq = filename.split('_')[-2]
    policy_rewards = np.load(os.path.join(data_dir, filename))

    # LOAD STATS HERE.
    variance = policy_rewards.var()
    # Use sample frequency to recreate the x-axis.
    episodes = np.arange(policy_rewards.shape[0]) * int(sample_freq)

    # Initialize figure for learning curve plots.
    ax = fig.add_subplot(111)

    # Putting results into plot.
    ax.plot(episodes, policy_rewards, label=model)
    ax.fill_between(episodes,
                    (policy_rewards - variance),
                    (policy_rewards + variance), alpha=0.3)


plt.title('Cumulative Rewards')
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=len(files_list), borderaxespad=0.)
fig.savefig('samplefigure', bbox_inches='tight')

# plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
# Some tricks were used to get legend outside of plot without it getting cut off.
# handles, labels = ax.get_legend_handles_labels()
# lgd = ax.legend(handles, `                                labels, loc='upper center', bbox_to_anchor=(1.2,0.6))
# plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
# fig.clear()