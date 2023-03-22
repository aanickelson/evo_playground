import numpy as np
from os import path, getcwd, mkdir
from scipy.stats import sem
from matplotlib import pyplot as plt

def process(d):
    mu = np.mean(d, axis=0)
    st = sem(d, axis=0)
    return mu, st

def load_data(fpath, n_stats, p_num, rew):
    all_data = []
    for stat_num in range(n_stats):
        stat_fpath = path.join(fpath, rew,  f'{p_num:03d}_{stat_num:02d}.npy')
        x = np.load(stat_fpath)
        all_data.append(x)
    all_data = np.array(all_data)
    return all_data

def plot_data(means, stes, rew, trial, base_fpath):
    plt.clf()
    fname = path.join(base_fpath, 'graphs', f"trial{trial:03d}_{rew}.svg")

    x_vals = [i for i in range(len(means[0]))]
    upper_y_lim = 1
    for i, avg in enumerate(means):
        plt.plot(x_vals, avg)
        plt.fill_between(x_vals, avg-stes[i], avg+stes[i], alpha=0.5, label='_nolegend_')
        max_avg = max(avg)
        if max_avg > upper_y_lim:
            upper_y_lim = max_avg
    plt.xlabel("Epoch")
    plt.ylim([0, upper_y_lim*1.02])
    plt.margins(y=1)
    plt.savefig(fname)


if __name__ == '__main__':
    from evo_playground.parameters.debugLP import LearnParams as lp
    from evo_playground.parameters.parameters02 import Parameters as p

    fp = path.join(getcwd(), 'base_G_002_20230322_135157')
    try:
        graphs_path_nm = path.join(fp, 'graphs')
        mkdir(graphs_path_nm)
    except FileExistsError:
        pass
    rew = 'G'
    filepath = path.join(fp, rew)
    data = load_data(fp, 5, p.param_idx, rew)
    mean, ste = process(data)
    plot_data([mean], [ste], rew, p.param_idx, fp)
