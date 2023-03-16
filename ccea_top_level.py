# Python packages
import joblib
from os import path, getcwd, mkdir
import numpy as np
from datetime import datetime
import torch

# Custom packages
from teaming.domain_hierarchy_policies import DomainHierarchy as Domain
from ccea_binary import CCEA
# from learning.neuralnet import NeuralNetwork as NN
from learning.neuralnet_no_hid import NeuralNetwork as NN


class CCEA_Top(CCEA):
    def __init__(self, p, rew_type, fpath, fitnesses, bh, wts, reselect=1):
        self.env = Domain(p, reselect)
        super().__init__(self.env, p, rew_type, fpath)
        self.generations = range(p.n_top_gen)
        self.ll_policies = None
        self.pareto_vals = None
        self.behaviors = None
        self.unpack_data(fitnesses, bh, wts)
        self.env.setup(self.pareto_vals, self.ll_policies, self.behaviors)
        self.nn_in = self.env.global_st_size()
        self.nn_out = self.env.top_out_size()

    def unpack_data(self, fits, bhs, wts_arr):
        pols = []
        for wts in wts_arr:
            nn = NN(self.env.state_size(), self.p.hid, self.env.get_action_size())
            wts = np.reshape(wts, (8, 11))
            torch_wts = [torch.from_numpy(wts)]
            nn.set_weights(torch_wts)
            pols.append(nn)

        g_arr = np.array(fits)
        bh_arr = np.array(bhs)
        g_and_bh = np.concatenate((g_arr, bh_arr), axis=1)
        _, unique_idx = np.unique(g_and_bh, axis=0, return_index=True)
        unique_idx = np.array(unique_idx)
        self.ll_policies = [pols[i] for i in unique_idx]
        self.pareto_vals = g_arr[unique_idx]
        self.behaviors = bh_arr[unique_idx]


def make_dirs(base_fpath):
    filepath = path.join(base_fpath, 'top_pol')
    try:
        mkdir(filepath)
    except FileExistsError:
        pass

    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    fpath_now = path.join(filepath, now_str)
    mkdir(fpath_now)

    for rew in ['G']:  # , 'D']:  #
        fpath = path.join(fpath_now, rew)
        mkdir(fpath)

    return fpath_now


def main(p, date_stamp):

    base_fpath = path.join(getcwd(), 'data', f'qdpar_{p.trial_num:03d}_{date_stamp}')

    # base_fpath = path.join(getcwd(), 'data', date_stamp)
    data = load_data(base_fpath)
    fitnesses = data[:, :2]
    bh_data = data[:, 7:12]
    wts = data[:, 12:]
    is_eff = is_pareto_efficient_simple(fitnesses)

    # Keep only the data that is on the pareto front
    fitnesses_p = fitnesses[is_eff]
    bh_p = bh_data[is_eff]
    wts_p = wts[is_eff]

    # plot_data(bh_data, pareto_data)
    p.n_agents = 2
    p.thirds = False
    trials_fpath = make_dirs(base_fpath)
    p.n_policies = 100

    for _ in range(1):
        for rew in ['G']:  #, 'D']:
            evo = CCEA_Top(p, rew, trials_fpath, fitnesses_p, bh_p, wts_p)
            evo.run_evolution()

def is_pareto_efficient_simple(vals):
    """
    copied and modified from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = np.array(vals)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a lower cost
            eff_add = np.all(costs == c, axis=1)
            is_efficient += eff_add
            is_efficient[i] = True  # And keep self
    return is_efficient


def load_data(base_fpath):
    wts_fpath = path.join(base_fpath, 'weights')
    rew_str = 'qdpar'
    # gens_to_load = [i * 100 for i in range(int(p.n_gen / 100))]
    # gens_to_load.append(p.n_gen - 1)
    # gen_num = p.n_gen - 1
    # gen_num = 300

    gen_num = 500016
    pth = path.join(wts_fpath, f't{p.trial_num:03d}_{rew_str}weights_{gen_num}.dat')
    data = np.loadtxt(pth)
    return data


def plot_data(data_to_plt, pareto_data):
    import numpy as np
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection='3d')

    # Plot a sin curve using the x and y axes.
    x = [d[0] for d in data_to_plt]
    y = [d[1] for d in data_to_plt]
    z = [d[2] for d in data_to_plt]


    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x, y) points are plotted on the x and z axes.
    ax.scatter(x, y, z, zdir='y', c=pareto_data)

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # on the plane y=0
    ax.view_init()

    plt.show()


if __name__ == '__main__':
    from parameters import p06 as p
    date_stamp = '20230214_153815'
    main(p, date_stamp)

    # This plays a noise when it's done so you don't have to babysit
    # import beepy
    # beepy.beep(sound="ready")

