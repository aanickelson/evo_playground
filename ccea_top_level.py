# Python packages
import joblib
from os import path, getcwd, mkdir
import numpy as np
from datetime import datetime

# Custom packages
from teaming.domain_hierarchy_policies import DomainHierarchy as Domain
from ccea_binary import CCEA
from learning.neuralnet import NeuralNetwork as NN


class CCEA_Top(CCEA):
    def __init__(self, p, rew_type, fpath, policies, reselect=5):
        self.env = Domain(p, reselect)
        super().__init__(self.env, p, rew_type, fpath)
        self.data = policies
        self.ll_policies = None
        self.pareto_vals = None
        self.unpack_data()
        self.env.setup(self.pareto_vals, self.ll_policies)
        self.nn_in = self.env.global_st_size()
        self.nn_out = 1

    def unpack_data(self):

        g_arr = []
        pols = []
        for i, sp in enumerate(self.data):
            sorted_list = sorted(sp, key=lambda x: x[0][0], reverse=False)
            ag_g_arr = []
            ag_pols = []
            for [gs, wts] in sorted_list:
                ag_g_arr.append(gs)
                nn = NN(self.env.state_size(), self.p.hid, self.env.get_action_size())
                nn.set_weights(wts)
                ag_pols.append(nn)

            g_arr.append(ag_g_arr)
            pols.append(ag_pols)

        self.ll_policies = pols
        self.pareto_vals = g_arr


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

    for rew in ['D', 'G']:  # 'D'
        fpath = path.join(fpath_now, rew)
        mkdir(fpath)

    return fpath_now


def main(p, date_stamp):
    base_fpath = path.join(getcwd(), 'data', date_stamp)
    data = unpack_data(base_fpath)
    trials_fpath = make_dirs(base_fpath)
    rew = 'G'
    p.n_gen = 1500
    evo = CCEA_Top(p, rew, trials_fpath, data)
    evo.run_evolution()


def unpack_data(base_fpath):
    wts_fpath = path.join(base_fpath, 'weights')
    rew_str = 'multi'
    pth = path.join(wts_fpath, f't{p.trial_num:03d}_{rew_str}weights_g{p.n_gen-1}.gz')
    data = joblib.load(pth)
    return data


if __name__ == '__main__':
    from parameters import p06 as p
    date_stamp = '20221222_122731'
    main(p, date_stamp)
