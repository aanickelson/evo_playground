import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from scipy.stats import sem
from os import getcwd, path
import parameters as param

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


def plot_it(avgs, sterrs, n, fname, att, trial_nums):
    plt.clf()
    x_vals = [n*i for i in range(len(avgs[0]))]
    upper_y_lim = 1
    for i, avg in enumerate(avgs):
        plt.plot(x_vals, avg)
        # plt.fill_between(x_vals, avg-sterrs[i], avg+sterrs[i], alpha=0.5)
        max_avg = max(avg)
        if max_avg > upper_y_lim:
            upper_y_lim = max_avg
    plt.xlabel("Epoch")
    plt.ylim([0, upper_y_lim*1.02])
    # plt.xlim([-10, 4800])

    # labels = [f'{i}' for i in range(p_st, p_end+1)]
    plt.legend(trial_nums, loc='lower right')
    plt.margins(y=1)
    if 'false' in att:
        plt.ylabel("Percent of time null actions chosen")
        plt.title("Percent null actions chosen")
    elif 'avg_G' in att:
        plt.ylabel("Average G captured by population")
        plt.title("Average total G captured by population")

    elif "avg" in att:
        plt.ylabel("Percent reward captured on average by entire population")
        plt.title("Average reward captured by population")
    elif "sterr" in att:
        plt.ylabel('Standard error across {} generations'.format(n))
        plt.title("Standard error from the mean across {} generations".format(n))
    else:
        plt.ylabel("Percent total reward captured")
        plt.title("Reward captured by best team, normalized to greedy policy")

    graphs_path = path.join(getcwd(), 'graphs', '{}.png'.format(fname))
    plt.savefig(graphs_path)


if __name__ == '__main__':

    attributes = ['_avg_G', '_max']   #['_avg', "_max", '_avg_G']  #, '_sterr']
    preps = ['D_time']  #, 'D_binary']  #['G_b',
    # trials = param.TEST_BATCH
    # trials = [param.p318, param.p319, param.p328, param.p329, param.p402, param.p403]
    # trials = param.BIG_BATCH_01
    # trials = [param.p596, param.p597, param.p598, param.p599]
    trials = [param.p614, param.p628, param.p629]  #, param.p530, param.p531]
    trial_nums = [p.trial_num for p in trials]



    # filename = "{".format('D_b', p.trial_num)
    p_st = trial_nums[0]
    p_end = trial_nums[-1]
    n = 50

    for att in attributes:
        fname_for_plt = f'comp {p_st}_to_{p_end}{att}'
        avgs = []
        sterrs = []
        for p in trials:
            # for i in range(3):
            for pre in preps:

                filename = "{}trial{:03d}".format(pre, p.trial_num)
                path_nm = path.join(getcwd(), 'data')
                filename2 = filename + att
                path_to_use = path.join(path_nm, "{}.csv".format(filename2))  # Done this way for csv so we can pass the filename to make the graphs
                try:
                    data = np.loadtxt(path_to_use)
                except FileNotFoundError:
                    print(f"{filename2} not found")
                    continue
                a, s = average_every_n(n, data)
                avgs.append(a)
                sterrs.append(s)

        plot_it(avgs, sterrs, n, fname_for_plt, att, trial_nums)
