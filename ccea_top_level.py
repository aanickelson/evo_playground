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
    def __init__(self, p, rew_type, fpath, data, reselect=1):
        self.env = Domain(p, reselect)
        super().__init__(self.env, p, rew_type, fpath)
        self.data = data
        self.ll_policies = None
        self.pareto_vals = None
        self.behaviors = None
        self.unpack_data()
        self.env.setup(self.pareto_vals, self.ll_policies, self.behaviors)
        self.nn_in = self.env.global_st_size()
        self.nn_out = self.env.top_out_size()

    def unpack_data(self):

        g_arr = []
        pols = []
        bh_arr = []
        for sp in self.data:
            ag_g_arr = []
            ag_pols = []
            ag_bh = []
            for [gs, wts, bh] in sp:
                ag_g_arr.append(gs)
                ag_bh.append(bh)
                nn = NN(self.env.state_size(), self.p.hid, self.env.get_action_size())
                nn.set_weights(wts)
                ag_pols.append(nn)

            g_arr.append(ag_g_arr)
            pols.append(ag_pols)
            bh_arr.append(ag_bh)

        self.ll_policies = pols
        self.pareto_vals = np.array(g_arr)
        self.behaviors = np.array(bh_arr)


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
    base_fpath = path.join(getcwd(), 'data', f'{p.trial_num:03d}_{date_stamp}')

    # base_fpath = path.join(getcwd(), 'data', date_stamp)
    data = unpack_data(base_fpath)
    p.n_agents = 2
    p.thirds = False
    if len(data) == 1 and len(data) < p.n_agents:
        data = data * p.n_agents
    trials_fpath = make_dirs(base_fpath)
    rew = 'D'
    p.n_gen = 1000
    evo = CCEA_Top(p, rew, trials_fpath, data)
    evo.run_evolution()


def unpack_data(base_fpath):
    wts_fpath = path.join(base_fpath, 'weights')
    rew_str = 'multi'
    gen_num = p.n_gen - 1
    # gen_num = 300
    pth = path.join(wts_fpath, f't{p.trial_num:03d}_{rew_str}weights_g{gen_num}.gz')
    data = joblib.load(pth)
    return data


if __name__ == '__main__':
    from parameters import p00 as p
    date_stamp = '20230107_114100'
    main(p, date_stamp)

    # This plays a noise when it's done so you don't have to babysit
    # import beepy
    # beepy.beep(sound="ready")

