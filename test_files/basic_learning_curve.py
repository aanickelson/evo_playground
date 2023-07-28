import numpy as np
import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # File name
    base_fname = '/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/516_20230726_160858/219_run2/top_20230727_151833'
    # Get subfile names
    file_data = list(os.walk(base_fname))[0][2]
    n_gen = 3000
    all_data = np.zeros((len(file_data), n_gen))
    for i, f in enumerate(file_data):
        # Import data from 5 files
        all_data[i] = np.load(f"{base_fname}/{f}")
    avgs = np.average(all_data, axis=0)
    stds = np.std(all_data, axis=0)
    x = np.array([i for i in range(len(stds))])
    plt.plot(x, avgs)
    plt.fill_between(x, avgs-stds, avgs+stds)
    plt.show()
