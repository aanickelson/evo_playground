import numpy as np
from scipy.stats import sem
from os import path, getcwd, mkdir
from matplotlib import pyplot as plt


def process(data):
    mu = np.mean(data, axis=0)
    ste = sem(data, axis=0)
    # mu = data[0]
    # ste = data[0]
    return mu, ste


def plot_data(means, stes, fpre, ext, trial, base_fpath):
    plt.clf()
    fname = path.join(base_fpath, 'graphs', f"trial{trial:03d}_{ext}.svg")

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


def load_data(path_nm, trial):
    try:
        graphs_path_nm = path.join(path_nm, 'graphs')
        mkdir(graphs_path_nm)
    except FileExistsError:
        pass

    fpre = ['G']  #, 'D', 'multi']
    ext = ['raw_G']  # , 'norm_G']

    for e in ext:
        means = []
        stes = []
        prefixes = []

        for pre in fpre:
            fname = path.join(path_nm, pre, f"{pre}_trial{trial:03d}_{e}.npy")
            try:
                data = np.load(fname)
            except FileNotFoundError:
                print(f'no file: {fname}')
                continue
            prefixes.append(pre)
            mean, ste = process(data)
            if pre == 'G':
                mean += 0.01
            means.append(mean)
            stes.append(ste)
        if means:
            plot_data(means, stes, prefixes, e, trial, base_fpath=path_nm)


if __name__ == '__main__':

    # prefix = 'moo'
    # top_pol = True
    # data_date = '20230113_164950'
    # data_top = '20230114_123229'
    #
    prefix = 'base'
    data_date = "20230313_184152"
    top_pol = False
    dates = ['20230310_152006', '20230310_153829', '20230310_155623', '20230310_161343', '20230310_163017']
    trial_num = 0
    # for data_date in dates:

    if top_pol:
        path_nm = path.join(getcwd(), 'data', f'{prefix}_{trial_num:03d}_{data_date}', 'top_pol', data_top)
        # path_nm = path.join(getcwd(), 'data', f'{data_date}', 'top_pol', data_top)
    else:
        path_nm = path.join(getcwd(), 'data', f'{prefix}_{trial_num:03d}_{data_date}')
        # path_nm = path.join(getcwd(), 'data', f'{data_date}')

    load_data(path_nm, trial_num)
    # path_nm = path.join(getcwd(), 'data', f'{trial_num:03d}_{data_date}')