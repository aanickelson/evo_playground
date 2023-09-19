import os.path
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
from itertools import combinations


class TopPolEnv:
    def __init__(self, p, lp, pfile, cfile, bh_sz, bhstrs, only_bh=False, only_obj=False):
        self.env = aic(p)
        self.wrap = RoverWrapper(self.env, bhstrs)
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

    def choose_pols(self, in_val, models):
        low_level_pols = []
        for i, policy in enumerate(models):
            # Get the NN output (behavior)
            bh_choice = policy(in_val).detach().numpy()
            # Get a policy from a filled niche close to that behavior
            pol_choice = self.pmap.get_pol(bh_choice, self.select_only_bh, self.select_only_obj)
            if len(pol_choice) > 0:
                low_level_pols.append(pol_choice)
            else:
                return -3
        if len(low_level_pols) < len(models):
            print("there are fewer policies than there should be")
            return -2
        return low_level_pols

    def run(self, top_models):
        total_G = 0
        D_scores = np.zeros(self.p.n_agents)
        for wt in self.moo_wts:
            G = np.array(self.run_env(wt, top_models))
            # scalar_G = G_exp(G, wt)
            scalar_G = np.sum(wt * G)
            total_G += scalar_G
            D = np.array(self.env.D())
            for i, ag_d in enumerate(D):
                # D_scores[i] += G_exp(ag_d, wt)
                D_scores[i] += np.sum(ag_d * wt)
        # total_G = total_G / len(self.moo_wts)
        return total_G, D_scores

    def _evaluate(self, low_pol_vals):
        # out_vals = run_env(self.env, models, self.p, use_bh=self.use_bh, vis=self.vis)
        # if self.use_bh:
        #     fitness, bh = out_vals
        #     return fitness, bh[0]
        # else:
        #     return out_vals
        pass

    def run_env(self, wt_bal, top_models, n_ts_select=5):
        for i in range(self.p.time_steps):
            if not i % n_ts_select:
                input_val = np.concatenate([wt_bal, self.env.G()])
                ll_pol_vals = self.choose_pols(input_val, top_models)
                low_level_pols = self.wrap.setup(ll_pol_vals)

            state = self.env.state()
            actions = []
            for i, policy in enumerate(low_level_pols):
                action = policy(state[i]).detach().numpy()
                actions.append(action)
            self.env.action(actions)
        return self.env.G()

    def reset(self):
        self.env.reset()

def get_bh_attrs():
    bh_options = ['battery', 'distance', 'type sep', 'type combo', 'v or e', 'full act']
    bh_combos = list(combinations(bh_options, 2))
    bh_options_one = [['type sep'], ['type combo'], ['v or e'], ['full act']]
    all_options = bh_options_one + bh_combos
    bh_sizes = {'battery': 1, 'distance': 1, 'type sep': 4, 'type combo': 2,
                'v or e': 2, 'full act': 10}
    bh_strs = []
    for c in all_options:
        combo_str = ''
        n_bh = 0
        for c0 in c:
            combo_str += c0 + '_'
            n_bh += bh_sizes[c0]
        bh_strs.append([c, combo_str[:-1], n_bh])
    return bh_strs


def setup():
    base_path = "/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/554_20230914_175010"
    learnp = LearnParams
    param_num = 200000
    p_base = Params.p200000
    params = copy.deepcopy(Params.p200000b)
    ll_pol_ngen = 100000
    ll_pol_niches = 1000

    rw_type = 'D'
    params.ag_in_st = p_base.ag_in_st
    params.counter = 0
    wts_size = 2
    out_wts_size = 2
    n_runs = 2
    learnp.n_stat_runs = 2
    learnp.n_gen = 500

    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    bh_options = get_bh_attrs()
    batch = []
    for [bh_strs, bh_combo, bh_size] in bh_options:
        for runnum in range(n_runs):
            data_path = os.path.join(base_path, f'{param_num}_{bh_combo}_run{runnum}')
            wts_path = data_path + f"/weights_{ll_pol_ngen}.dat"
            cent_path = data_path + f"/centroids_{ll_pol_niches}_{bh_size}.dat"
            if not os.path.exists(wts_path) or not os.path.exists(cent_path):
                print(f'These paths do not exist \n   {wts_path} \n   {cent_path}')
                continue
            for onlybh, onlyobj in [[False, False]]:  #, [True, False], [False, True]]:
                top_wts_path = data_path + f'/top_{now_str}_{(not onlybh)*"o"}{(not onlyobj)*"b"}/'
                print(top_wts_path)
                try:
                    mkdir(top_wts_path)
                except FileExistsError:
                    pass

                env = TopPolEnv(params, learnp, wts_path, cent_path, bh_size, bh_strs, onlybh, onlyobj)
                in_size = wts_size * 2
                out_size = bh_size + out_wts_size
                if onlybh:
                    out_size = bh_size
                if onlyobj:
                    out_size = out_wts_size

                for i in range(learnp.n_stat_runs):
                    batch.append([env, params, learnp, rw_type, in_size, out_size, top_wts_path, i])
            # batch = [[env, params, learnp, 'G', wts_size, bh_size + out_wts_size, top_wts_path, i] for i in range(learnp.n_stat_runs)]
    return batch


if __name__ == '__main__':

    b = setup()
    multiprocess_main(b)
    # for bat in b:
    #     main(bat)
    # main(b[0])

