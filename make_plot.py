import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
sns.set()

"""
Small script to generate learning curves from best training runs.
"""

result_types = ['losses', 'rewards']
figure_folder = os.path.join('outputs', 'figures')

# Iterate over all result files in either output/losses or output/rewards
for result_type in result_types:
    result_folder = os.path.join('outputs', result_type)
    for subdir, dirs, files in os.walk(result_folder):
        for filename in files:
            filepath = subdir + os.sep + filename
            with open(filepath, 'rb') as handle:
                results = np.load(handle)
            
            # Initialize figure for learning curve plots.
            fig = plt.figure(1)
            ax = fig.add_subplot(111)

            # Putting results into plot.
            ax.plot(results)

            # Some tricks were used to get legend outside of plot without it getting cut off.
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.2,0.6))
            plt.title(result_type)
            plt.show()
            save_file = os.path.join(figure_folder, filename[:-4] + '.png')
            plt.savefig(save_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.clear()