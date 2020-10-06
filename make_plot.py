import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
sns.set()

"""
Small script to generate learning curves from best training runs.
"""

# Load results saved from running main.py
results_path = os.path.join('outputs', 'policy_gradients', 'best_policy_gradients.pickle')
save_path = os.path.join('outputs', 'figures')
with open(results_path, 'rb') as handle:
    results = pickle.load(handle)

# Initialize figure for learning curve plots.
fig = plt.figure(1)
ax = fig.add_subplot(111)

# Putting results into plot.
for env_name in results.keys():
    ax.plot(results[env_name], label=env_name)

# Some tricks were used to get legend outside of plot without it getting cut off.
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.2,0.6))
plt.title('Episode durations per episode')
plt.savefig(os.path.join(save_path, 'training_curves'), bbox_extra_artists=(lgd,), bbox_inches='tight')