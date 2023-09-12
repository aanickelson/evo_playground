from datetime import datetime
import numpy as np
from os import getcwd, path, mkdir
import copy

from AIC.aic import aic
from evo_playground.support.get_policy_from_niche import PolicyMap
from evo_playground.support.rover_wrapper import RoverWrapper
from evo_playground.ccea_base import *
from evo_playground.parameters.learningparams02 import LearnParams
import pymap_elites_multiobjective.parameters as Params
from evo_playground.radians_G import G_exp


class TopPolEnv:
    def __init__(self, p, lp, pfile, cfile, bh_sz, only_bh=False, only_obj=False):
        self.env = aic(p)
        self.wrap = RoverWrapper(self.env)
        self.wrap.use_bh = False
        self.select_only_bh = only_bh
        self.select_only_obj = only_obj
        self.pmap = PolicyMap(pfile, cfile, bh_sz)
        self.p = p
        self.lp = lp
        self.pfile = pfile
        self.cfile = cfile
        self.moo_wts = np.array([[0.5, 0.5], [0.7, 0.3], [0.3, 0.7], [1., 0.], [0., 1.]])  # [0.7, 0.3], [0.3, 0.7],

    def G(self):
        return self.env.G()

    def D(self):
        return self.env.D()

    def run(self, models):
        total_G = 0
        for wt in self.moo_wts:
            low_level_pols = []
            for i, policy in enumerate(models):
                # Get the NN output (behavior)
                bh_choice = policy(wt).detach().numpy()
                # Get a policy from a filled niche close to that behavior
                pol_choice = self.pmap.get_pol(bh_choice, self.select_only_bh, self.select_only_obj)
                if len(pol_choice) > 0:
                    low_level_pols.append(pol_choice)
                else:
                    return -3
            if len(low_level_pols) < len(models):
                print("there are fewer policies than there should be")
                return -2

            G = np.array(self.wrap._evaluate(low_level_pols))
            scalar_G = G_exp(G, wt)
            total_G += scalar_G
        # total_G = total_G / len(self.moo_wts)
        return total_G

    def reset(self):
        self.env.reset()


def setup():
    base_path = "/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/545_20230911_154331/200000_run0"
    p_base = Params.p200000
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    params = copy.deepcopy(Params.p200000b)
    params.ag_in_st = p_base.ag_in_st
    params.counter = 0
    bh_size = 2
    wts_size = 2
    out_wts_size = 2
    learnp = LearnParams
    learnp.n_stat_runs = 2
    learnp.n_gen = 5
    batch = []
    for onlybh, onlyobj in [[False, True]]:  # [False, False], [True, False],
        top_wts_path = base_path + f'/top_{now_str}_{(not onlybh)*"o"}{(not onlyobj)*"b"}/'
        print(top_wts_path)
        try:
            mkdir(top_wts_path)
        except FileExistsError:
            pass

        wts_path = base_path + "/weights_100000.dat"
        cent_path = base_path + "/centroids_1000_2.dat"

        env = TopPolEnv(params, learnp, wts_path, cent_path, bh_size, onlybh, onlyobj)
        for i in range(learnp.n_stat_runs):
            batch.append([env, params, learnp, 'G', wts_size, bh_size + out_wts_size, top_wts_path, i])
    # batch = [[env, params, learnp, 'G', wts_size, bh_size + out_wts_size, top_wts_path, i] for i in range(learnp.n_stat_runs)]
    return batch


if __name__ == '__main__':

    b = setup()
    # multiprocess_main(b)
    main(b[0])

