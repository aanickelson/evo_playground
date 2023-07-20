import numpy as np
from matplotlib import pyplot as plt
import os



if __name__ == '__main__':

    fname = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/518_20230713_151107/112_run0/top_20230718_150159'
    ngen = 3000
    files = list(os.walk(fname))[0][2]  # returns [(root, dirs, files)]
    all_fits = np.zeros((len(files), ngen))
    for i, f in enumerate(files):
        fits = np.load(f'{fname}/{f}').transpose()
        all_fits[i] = fits
    avg = np.mean(all_fits, axis=0)
    std = np.std(all_fits, axis=0)
    x = np.arange(1, len(avg) + 1)
    plt.plot(x, avg)
    plt.fill_between(x, avg-std, avg+std)
    plt.show()
