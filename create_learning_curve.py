import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from scipy.stats import sem
from os import getcwd, path
# from parameters import PARAMS

np.set_printoptions(precision=3)


def average_every_n(n, data):
    data_len = len(data)
    n_results = ceil(data_len / n)
    avg = np.zeros(n_results)
    sterr = np.zeros(n_results)
    for i in range(n_results):
        curr_range = data[(i*n):(i+1)*n]
        avg[i] = np.mean(curr_range)
        sterr[i] = sem(curr_range)
    return avg, sterr


def plot_it(avgs, sterrs, n, fname):
    plt.clf()
    x_vals = [n*i for i in range(len(avgs))]
    plt.plot(x_vals, avgs, 'k-')
    plt.fill_between(x_vals, avgs-sterrs, avgs+sterrs, alpha=0.5)
    upper_y_lim = max(avgs)*1.01
    plt.xlabel("Epoch")
    plt.ylim([0, upper_y_lim])
    plt.margins(y=1)
    if 'false' in fname:
        plt.ylabel("Percent of time null actions chosen")
        plt.title("Percent null actions chosen")
    elif "avg" in fname:
        plt.ylabel("Percent reward captured on average by entire population")
        plt.title("Average reward captured by population")
    elif "sterr" in fname:
        plt.ylabel('Standard error across {} generations'.format(n))
        plt.title("Standard error from the mean across {} generations".format(n))
    else:
        plt.ylabel("Percent total reward captured by best policy")
        plt.title("Reward captured by best agent")
    graphs_path = path.join(getcwd(), 'graphs', '{}.png'.format(fname))
    plt.savefig(graphs_path)


if __name__ == '__main__':
    start_trial = 99
    end_trial = 99

    for num in range(start_trial, end_trial + 1):
        filename = "trial{:02d}".format(num)
        attributes = ['_avg_G', '_avg', "_max"] #, '_sterr']
        path_nm = path.join(getcwd(), 'data')
        for att in attributes:
            filename2 = filename + att
            path_to_use = path.join(path_nm, "{}.csv".format(filename2))  # Done this way for csv so we can pass the filename to make the graphs
            data = np.loadtxt(path_to_use)
            n = 20
            avgs, sterrs = average_every_n(n, data)
            plot_it(avgs, sterrs, n, filename2)
