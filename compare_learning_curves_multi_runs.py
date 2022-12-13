import numpy as np
from scipy.stats import sem
from os import path, getcwd, mkdir
from matplotlib import pyplot as plt


def process(data):
    mu = np.mean(data, axis=0)
    ste = sem(data, axis=0)
    mu = data[0]
    ste = data[0]
    return mu, ste


def plot_data(means, stes, fpre, ext, trial, data_date):
    plt.clf()
    fname = path.join(getcwd(), 'data', data_date, 'graphs', f"trial{trial:03d}_{ext}.svg")

    x_vals = [i for i in range(len(means[0]))]
    upper_y_lim = 1
    for i, avg in enumerate(means):
        plt.plot(x_vals, avg)
        # plt.fill_between(x_vals, avg-stes[i], avg+stes[i], alpha=0.5, label='_nolegend_')
        max_avg = max(avg)
        if max_avg > upper_y_lim:
            upper_y_lim = max_avg
    plt.xlabel("Epoch")
    plt.ylim([0, upper_y_lim*1.02])
    plt.legend(fpre, loc='lower right')
    plt.margins(y=1)
    plt.savefig(fname)


def load_data(data_date, trials):
    try:
        path_nm = path.join(getcwd(), 'data', data_date, 'graphs')
        mkdir(path_nm)
    except FileExistsError:
        pass

    fpre = ['G', 'D']
    ext = ['raw_G', 'norm_G']
    path_nm = path.join(getcwd(), 'data', data_date)

    for trial in trials:
        for e in ext:
            means = []
            stes = []
            prefixes = []

            for pre in fpre:
                fname = path.join(path_nm, pre, f"{pre}_trial{trial:03d}_{e}.npy")
                try:
                    data = np.load(fname)
                except FileNotFoundError:
                    continue
                prefixes.append(pre)
                mean, ste = process(data)
                means.append(mean)
                stes.append(ste)
            if means:
                plot_data(means, stes, prefixes, e, trial, data_date)


if __name__ == '__main__':
    date = "20221213_152035"
    t0 = 2
    t1 = 2

    trials = [n for n in range(t0, t1 + 1)]

    load_data(date, trials)
